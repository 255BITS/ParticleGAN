"""Caller-owned E22 updates using the installed package.

CPU smoke: python -u examples/e22_external_loop.py --steps 5
Recovery:  python -u examples/e22_external_loop.py --steps 5 --checkpoint /tmp/e22.pt
           python -u examples/e22_external_loop.py --steps 5 --resume /tmp/e22.pt

The policy owns E22's controls and recovery state. This application owns the
networks, losses, backward calls, optimizer steps, and data cursor. The update
function is also the external-loop implementation used by the conformance tests.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json

import torch
from torch import nn

from particlegan import E22Policy, InputNoise, get_recipe, init


@dataclass
class Loop:
    generator: nn.Module
    critic: nn.Module
    prior: nn.Module
    opt_g: torch.optim.Optimizer
    opt_d: torch.optim.Optimizer
    policy: E22Policy
    penalty: object
    loss: object
    prior_regularizer: nn.Module
    noisy_critic: InputNoise


def make_loop(recipe, generator, critic, prior, *, seed=0):
    """Bind caller-owned parameters and optimizers to the complete E22 policy."""
    opt_g, opt_d = recipe.make_optimizers(
        generator, critic, prior, ema_critic=deepcopy(critic))
    penalty = recipe.make_critic_penalty(opt_d)
    policy = E22Policy(
        recipe, generator, critic, prior=prior,
        generator_optimizer=opt_g, critic_optimizer=opt_d,
        roles=[["generator", "table"], ["critic"]], seed=seed, penalty=penalty)
    return Loop(generator, critic, prior, opt_g, opt_d, policy, penalty,
                recipe.make_loss(), recipe.make_prior_regularizer(weight=1.0),
                InputNoise(critic, generator=policy.noise_generator))


def update(loop, real, *, generator_real=None, collect_stats=False):
    """One critic update followed by one generator/table update.

    Checkpoint only after ``finish_step``. For exact higher-order CUDA replay,
    wrap the entire update in ``torch.autograd.set_multithreading_enabled(False)``.
    """
    G, D, prior, policy = loop.generator, loop.critic, loop.prior, loop.policy
    opt_g, opt_d = loop.opt_g, loop.opt_d
    noise = policy.begin_step(real, game_record=opt_d.record)
    loop.noisy_critic.std = noise.input_sigma

    D.train()
    G.eval()
    with torch.no_grad():
        latent, _ = prior.sample(len(real), generator=policy.latent_generator)
        policy.observe_support(latent)
        fake = policy.generate(latent, sigma=noise.output_sigma)
    policy.observe_critic_pair(real, fake)
    adversarial_d = loop.loss.d_loss(loop.noisy_critic(real), loop.noisy_critic(fake))
    loop.penalty.collect_stats = collect_stats
    penalty = loop.penalty(loop.noisy_critic, real, fake)
    loss_d = adversarial_d + penalty
    opt_d.zero_grad()
    policy.before_critic_backward()
    loss_d.backward()
    opt_d.step()                         # KA2 guard, Adam, critic anchor update
    policy.after_critic_step()           # observe the applied critic displacement

    D.eval()
    G.train()
    flags = [parameter.requires_grad for parameter in D.parameters()]
    try:
        D.requires_grad_(False)
        latent, indices = prior.sample(len(real), generator=policy.latent_generator)
        fake_logits = loop.noisy_critic(policy.generate(latent, sigma=noise.output_sigma))
        real_g = generator_real() if callable(generator_real) else generator_real
        real_g = real if real_g is None else real_g
        real_logits = loop.noisy_critic(real_g)
        loss_gan = loop.loss.g_loss(fake_logits, real_logits)
        prior_regularization = loss_gan.new_zeros(())
        if prior.z.requires_grad:
            raw = prior.z if len(prior.z) <= 1024 else prior.z[indices.unique()]
            prior_regularization = loop.prior_regularizer(raw)
        loss_g = loss_gan + policy.recipe.prior_reg * prior_regularization
        opt_g.zero_grad()
        policy.before_generator_backward()
        loss_g.backward()
        policy.after_generator_backward(
            loss_gan=loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
        opt_g.step()                     # Adam, A2 damping and direct-particle response
        policy.after_generator_step()   # restore hot-row rates, observe displacement
    finally:
        for parameter, flag in zip(D.parameters(), flags):
            parameter.requires_grad_(flag)
    policy.finish_step()                # averages, birth/death, rebase, served choice
    result = {key: value.detach() for key, value in dict(
        loss_d=loss_d, loss_g=loss_g, loss_gan=loss_gan,
        prior_regularization=prior_regularization, penalty=penalty).items()}
    result["step"] = policy.completed_steps
    if collect_stats:
        result["penalty_stats"] = loop.penalty.last_stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5, help="Additional updates; E22 has no horizon")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", "--output", dest="checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    # These dimensions, network frequencies and observation noise belong to the task.
    recipe = get_recipe("e22", num_particles=args.particles, z_dim=2,
                        batch_size=args.batch_size, output_noise_std=0.029)
    G = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(.2), nn.Linear(32, 2)).to(device)
    D = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(.2), nn.Linear(32, 1)).to(device)
    prior = recipe.make_prior().to(device)
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    init.deterministic_orthogonal_(prior, seed=2)
    loop = make_loop(recipe, G, D, prior)
    data_rng = torch.Generator(device=device).manual_seed(42)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        loop.policy.load_state_dict(saved["policy"])
        data_rng.set_state(saved["data_rng"].cpu())
    centers = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], device=device)
    for _ in range(args.steps):
        ids = torch.randint(len(centers), (recipe.batch_size,), device=device, generator=data_rng)
        real = centers[ids] + recipe.output_noise_std * torch.randn(
            recipe.batch_size, 2, device=device, generator=data_rng)
        with torch.autograd.set_multithreading_enabled(False):
            stats = update(loop, real)
        print(json.dumps({"event": "train", "step": stats["step"],
                          "loss_d": float(stats["loss_d"]), "loss_g": float(stats["loss_g"]),
                          "output_sigma": loop.policy.output_sigma(),
                          "birth_death": loop.policy.birth_death.diagnostics()["counters"]}), flush=True)
    if args.checkpoint:
        torch.save({"policy": loop.policy.state_dict(), "data_rng": data_rng.get_state()}, args.checkpoint)

    # Frozen independent copies implement E22's current served choice, including
    # DV12 perturbation. The caller's training weights and streams stay in place.
    served = loop.policy.served_model()
    eval_rng = torch.Generator(device=device).manual_seed(100)
    samples = served.sample(16, generator=eval_rng, output_noise=True)
    print(json.dumps({"event": "complete", "served_source": served.source,
                      "sample_shape": list(samples.shape)}), flush=True)


if __name__ == "__main__":
    main()

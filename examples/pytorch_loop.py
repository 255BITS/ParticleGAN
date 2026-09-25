"""A caller-owned PyTorch loop using only Torch and the installed particlegan.

CPU smoke: python -u examples/pytorch_loop.py --steps 5 --batch-size 16
TOML:     python -u examples/pytorch_loop.py --config examples/api.toml

This small MLP demonstrates integration using the recommended (K3P) defaults,
wiring every K3P component explicitly -- the same update GANTrainer performs:

* role-wise LR schedule: ``learning_rate_scales`` (network horizon + prior),
* critic input noise / generator output noise (annealed, from one stream),
* K3P gradient penalty with an EMA-critic ``CriticAnchor`` you allocate,
* ``CriticSpikeGuard`` before each critic Adam step,
* ``after_critic_step`` after it (anchor EMA + the LR that drives the blend),
* A2 ``LatentRowDamping`` around the prior's Adam step (history you allocate).

To checkpoint, save the modules, both optimizers, ``ema_critic``,
``latent_history`` and each component's ``state_dict()``. With several
critics, build one penalty/anchor/guard set per critic optimizer.
Replace its networks and synthetic batches with your own pipeline.
"""

import argparse
import copy
import json
import time

import torch
from torch import nn

from particlegan import get_recipe, learning_rate_scales
from particlegan.training import input_noise_std, output_noise_std


@torch.no_grad()
def update_ema(average, current, decay):
    for target, source in zip(average.parameters(), current.parameters()):
        target.lerp_(source, 1.0 - decay)
    for target, source in zip(average.buffers(), current.buffers()):
        target.copy_(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="TOML file with a [particlegan] section")
    parser.add_argument("--steps", type=int, help="Override the recipe's training horizon")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")

    options = {}
    if args.config:
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10: pip install tomli
            import tomli as tomllib
        with open(args.config, "rb") as stream:
            options = tomllib.load(stream)["particlegan"]
    if args.steps is not None:
        options["total_steps"] = args.steps
    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    recipe = get_recipe(**options)
    if (recipe.model != "gan" or recipe.conditioning != "scalar"
            or recipe.encoder_mode != "none"):
        parser.error("this one-shot example requires model='gan', conditioning='scalar', encoder_mode='none'")

    device = torch.device(args.device)
    # These choices belong to this application, not the library.
    torch.manual_seed(42)
    if device.type == "cpu":
        torch.set_num_threads(1)
    generator = nn.Sequential(
        nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 2),
    ).to(device)
    critic = nn.Sequential(
        nn.Linear(2, 64), nn.LeakyReLU(0.2),
        nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1),
    ).to(device)
    prior = recipe.make_prior().to(device)
    gan = recipe.make_loss()
    # K3P: the caller allocates the EMA critic; the anchor only averages it.
    ema_critic = copy.deepcopy(critic).requires_grad_(False)
    anchor = recipe.make_critic_anchor(critic, ema_critic)
    penalty = recipe.make_gradient_penalty(anchor=anchor if recipe.reg_arm == "k3p" else None)
    guard = recipe.make_critic_guard()  # None when d_guard_ratio == 0
    spread = recipe.make_prior_regularizer()
    # Ordinary Adam optimizers ([generator, prior] groups, and the critic).
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    # A2 needs the prior table alone in its Adam group with beta1 == 0.
    latent_history = torch.zeros_like(prior.z.detach())
    damping = recipe.make_latent_damping(prior.z, latent_history)  # None when disabled
    noise = torch.Generator(device=device).manual_seed(43)

    def with_noise(x, sigma):
        if sigma == 0:
            return x
        return x + sigma * torch.randn(x.shape, generator=noise, device=x.device, dtype=x.dtype)
    ema_g = copy.deepcopy(generator).eval().requires_grad_(False)
    ema_prior = copy.deepcopy(prior).eval().requires_grad_(False)

    axis = torch.linspace(-1.0, 1.0, 10, device=device)
    centers = torch.cartesian_prod(axis, axis)
    print(json.dumps({"event": "config", **recipe.to_dict(), "device": str(device)}), flush=True)
    started = time.monotonic()
    for step in range(1, recipe.total_steps + 1):
        network, prior_scale = learning_rate_scales(step - 1, recipe)
        opt_g.param_groups[0]["lr"] = base_lrs[0][0] * network
        opt_g.param_groups[1]["lr"] = base_lrs[0][1] * prior_scale
        opt_d.param_groups[0]["lr"] = base_lrs[1][0] * network
        sigma_in = input_noise_std(recipe, step - 1)
        sigma_out = output_noise_std(recipe, step - 1)

        def noisy(fn):  # fresh critic input noise per evaluation
            return lambda x: fn(with_noise(x, sigma_in))

        # Replace this synthetic batch with a batch from your DataLoader.
        ids = torch.randint(len(centers), (recipe.batch_size,), device=device)
        real = centers[ids] + 0.015 * torch.randn(recipe.batch_size, 2, device=device)
        z, particle_ids = prior.sample(recipe.batch_size)
        fake = with_noise(generator(z), sigma_out)

        opt_d.zero_grad(set_to_none=True)
        d_loss = gan.d_loss(noisy(critic)(real), noisy(critic)(fake.detach()))
        d_loss = d_loss + penalty(noisy(critic), real, fake.detach(), step, ema_critic=noisy(anchor))
        d_loss.backward()
        if guard is not None:
            guard.apply_(opt_d)
        opt_d.step()
        penalty.after_critic_step(opt_d)  # anchor EMA, then the LR that drives s

        # Freeze critic weights while preserving gradients through critic(fake).
        flags = [parameter.requires_grad for parameter in critic.parameters()]
        critic.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            g_loss = gan.g_loss(noisy(critic)(fake), noisy(critic)(real).detach())
            prior_loss = spread(prior.z[particle_ids.unique()])
            total_g = g_loss + prior_loss  # spread already carries recipe.prior_reg
            total_g.backward()
            if damping is None:
                opt_g.step()
            else:
                with damping.around(opt_g):
                    opt_g.step()
        finally:
            for parameter, flag in zip(critic.parameters(), flags):
                parameter.requires_grad_(flag)
        update_ema(ema_g, generator, recipe.ema_decay)
        update_ema(ema_prior, prior, recipe.ema_decay)

        if step == 1 or step % args.log_every == 0 or step == recipe.total_steps:
            print(json.dumps({
                "event": "train", "step": step,
                "d_loss": d_loss.detach().item(), "g_loss": g_loss.detach().item(),
                "prior_loss": prior_loss.detach().item(), "lr_scale": network,
                "k3p_s": penalty.blend_weight(),
                "seconds": round(time.monotonic() - started, 3),
            }), flush=True)

    with torch.no_grad():
        z, _ = ema_prior.sample(256)
        samples = ema_g(z)
    if not torch.isfinite(samples).all():
        raise RuntimeError("non-finite generated samples")
    print(json.dumps({"event": "complete", "sample_shape": list(samples.shape)}), flush=True)


if __name__ == "__main__":
    main()

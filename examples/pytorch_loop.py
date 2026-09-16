"""A caller-owned PyTorch loop using only Torch and the installed particlegan.

CPU smoke: python -u examples/pytorch_loop.py --steps 5 --batch-size 16
TOML:     python -u examples/pytorch_loop.py --config examples/api.toml

This intentionally small MLP demonstrates integration. The benchmark's Fourier
critic, evaluation, and visualizations live in examples/100gaussians.py.
"""

import argparse
import copy
import json
import time

import torch
from torch import nn

from particlegan import get_recipe, learning_rate_scale


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
    if recipe.model != "gan" or recipe.conditioning != "scalar":
        parser.error("this one-shot example requires model='gan', conditioning='scalar'")

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
    penalty = recipe.make_gradient_penalty()
    spread = recipe.make_prior_regularizer()
    # Ordinary Adam optimizers; replace these with your own if desired.
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    ema_g = copy.deepcopy(generator).eval().requires_grad_(False)
    ema_prior = copy.deepcopy(prior).eval().requires_grad_(False)

    axis = torch.linspace(-1.0, 1.0, 10, device=device)
    centers = torch.cartesian_prod(axis, axis)
    print(json.dumps({"event": "config", **recipe.to_dict(), "device": str(device)}), flush=True)
    started = time.monotonic()
    for step in range(1, recipe.total_steps + 1):
        scale = learning_rate_scale(
            step - 1, recipe.total_steps, start=recipe.lr_anneal_start, floor=recipe.lr_floor,
        )
        for opt, rates in zip((opt_g, opt_d), base_lrs):
            for group, rate in zip(opt.param_groups, rates):
                group["lr"] = rate * scale

        # Replace this synthetic batch with a batch from your DataLoader.
        ids = torch.randint(len(centers), (recipe.batch_size,), device=device)
        real = centers[ids] + 0.015 * torch.randn(recipe.batch_size, 2, device=device)
        z, particle_ids = prior.sample(recipe.batch_size)
        fake = generator(z)

        opt_d.zero_grad(set_to_none=True)
        d_loss = gan.d_loss(critic(real), critic(fake.detach()))
        d_loss = d_loss + penalty(critic, real, fake.detach(), step=step)
        d_loss.backward()
        opt_d.step()

        # Freeze critic weights while preserving gradients through critic(fake).
        flags = [parameter.requires_grad for parameter in critic.parameters()]
        critic.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            g_loss = gan.g_loss(critic(fake), critic(real).detach())
            prior_loss = spread(prior(particle_ids.unique()))
            total_g = g_loss + prior_loss
            total_g.backward()
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
                "prior_loss": prior_loss.detach().item(), "lr_scale": scale,
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

"""Numerical host extracted from HyperGAN/conceptmod commit 5571213.

Training/data/evaluation logic retained; config-identity refusals removed.
See ../SOURCE.md and ../LICENSE. Candidate settings are supplied by baseline.py.
"""

from __future__ import annotations

from contextlib import nullcontext


from dataclasses import dataclass


from ..observation import checkpoint, schedule_optimizer

import torch


from torch import nn


from benchmarks.legacy.recipe import get_recipe


from benchmarks.legacy.grad_regularizers import GradRegularizer
from benchmarks.gan_v3 import gan_v3_recipe


DEMO_COVER = 1.5


PARTICLE_L2 = 0.02


N_PARTICLES = 12


STEPS = 250


BATCH = 64


LR = 2e-3


SEED = 0


DATA_STD = 0.05


ANCHORS = ((-1.5, 0.0), (1.5, 0.0))


RECON_MAX = 0.05


HOLD_MAX = 0.35




@dataclass(frozen=True)
class HoldConfig:
    name: str
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    reg_lazy: int = 1
    target_anneal: str = "none"
    fm_weight: float = 0.0
    cover_weight: float = DEMO_COVER
    particle_l2: float = PARTICLE_L2
    n_particles: int = N_PARTICLES
    reconstruction_weight: float = 1.0
    adversarial_weight: float = 1.0
    encoder_mode: str = "ae"
    steps: int = STEPS
    batch: int = BATCH
    lr: float = LR
    seed: int = SEED


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, dout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[1](self.net[0](x))


def _anchors() -> torch.Tensor:
    return torch.tensor(ANCHORS, dtype=torch.float32)


def sample_data(n: int) -> torch.Tensor:
    choice = torch.randint(0, len(ANCHORS), (n,))
    return _anchors()[choice] + DATA_STD * torch.randn(n, 2)


def _hold_distance(fake: torch.Tensor) -> float:
    distances = torch.cdist(_anchors(), fake)
    return float(distances.min(dim=1).values.mean())


@torch.no_grad()
def evaluate(encoder, decoder, prior, recipe) -> dict[str, float]:
    """Score reconstruction and unconditional hold without moving the train RNG."""
    state = torch.get_rng_state()
    try:
        data = sample_data(1024)
        query, offset = encoder(data).chunk(2, dim=1)
        encoded = recipe.encode(query, prior, offset=offset)
        recon = decoder(encoded.codes[:, 0])
        recon_mse = float((recon - data).square().mean())
        codes, _ = prior.sample(1024)
        fake = decoder(codes)
        return {"recon_mse": recon_mse, "hold": _hold_distance(fake)}
    finally:
        torch.set_rng_state(state)


def _log(arm: str, step: int, metrics: dict, extra: str = "") -> None:
    print(
        f"ae-gan-hold arm={arm} step={step} recon={metrics['recon_mse']:.4f} "
        f"hold={metrics['hold']:.4f}{extra}",
        flush=True,
    )


def make_recipe(cfg: HoldConfig):
    """Frozen AE host resources with this candidate's numerical settings."""
    return gan_v3_recipe(
        prior_kind='mog', sigma_rel=0.025, z_dim=2, total_steps=6000,
        betas=(0., .999), d_lr_mult=1.5, prior_lr_mult=10., prior_betas=(.5, .999),
        prior_reg=1., lr_floor=1.,
        num_particles=cfg.n_particles,
        reg_every=cfg.reg_lazy,
        reg_arm=cfg.reg_arm,
        reg_coeff=cfg.reg_coeff,
        reg_kappa=cfg.reg_kappa,
        loss_type=cfg.loss_type,
        gan_mode=cfg.gan_mode,
        reconstruction_weight=cfg.reconstruction_weight,
        lr=cfg.lr,
        encoder_mode=cfg.encoder_mode,
    )


def train(cfg: HoldConfig, *, noise_policy=None) -> dict:
    """Train and return reconstruction/hold measurements."""
    torch.manual_seed(cfg.seed)
    recipe = make_recipe(cfg)
    prior = recipe.make_prior()
    encoder, decoder, critic = MLP(2, 4), MLP(2, 2), MLP(2, 1)
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output
        decoder = wrap_output(decoder, noise_policy)
        critic = wrap_input(critic, noise_policy)
    opt_g, opt_d = recipe.make_optimizers(decoder, critic, prior, encoder=encoder)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    gan = recipe.make_loss()
    regularizer = recipe.make_gradient_penalty(norm=cfg.reg_norm, target_anneal=cfg.target_anneal)
    # The source's removed regularizer audit reset the CPU RNG to seed 0 and
    # consumed three uniform values (Linear(2, 1) initialization), then two
    # 8x2 normal tensors. Preserve that training-data stream for EVERY candidate
    # without constructing an audit critic or checking its loss/penalty identity.
    stream = torch.Generator().manual_seed(0)
    torch.rand(3, generator=stream)
    torch.randn(8, 2, generator=stream)
    torch.randn(8, 2, generator=stream)
    torch.set_rng_state(stream.get_state())

    def measure(step: int):
        context = noise_policy.evaluation(step) if noise_policy is not None else nullcontext()
        with context:
            return evaluate(encoder, decoder, prior, recipe)

    opened = measure(0)
    _log(cfg.name, 0, opened, extra=" phase=init")
    penalty_applied = 0
    adv_steps = 0
    for step in range(1, cfg.steps + 1):
        if noise_policy is not None:
            noise_policy.set_step(step - 1)
        data = sample_data(cfg.batch)
        if cfg.adversarial_weight > 0:
            codes, _ = prior.sample(cfg.batch)
            context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
            with context:
                fake = decoder(codes).detach()
            opt_d.zero_grad(set_to_none=True)
            d_loss = gan.d_loss(critic(data).squeeze(-1), critic(fake).squeeze(-1))
            penalty, stats = regularizer.penalty(critic, data, fake, step=step)
            if stats.get("applied"):
                penalty_applied += 1
            (d_loss + penalty).backward()
            schedule_optimizer(opt_d, step - 1)
            opt_d.step()

        query, offset = encoder(data).chunk(2, dim=1)
        encoded = recipe.encode(query, prior, offset=offset)
        reconstructed = decoder(encoded.codes[:, 0])
        recon = encoded.reconstruction_loss(reconstructed[:, None], data)
        codes, _ = prior.sample(cfg.batch)
        generated = decoder(codes)
        opt_g.zero_grad(set_to_none=True)
        for param in critic.parameters():
            param.requires_grad_(False)
        loss = cfg.reconstruction_weight * recon + cfg.particle_l2 * prior.z.square().mean()
        if cfg.adversarial_weight > 0:
            real_logits = critic(data).squeeze(-1).detach()
            fake_logits = critic(generated).squeeze(-1)
            adv = gan.g_loss(fake_logits, real_logits)
            anchors = _anchors()
            cover = torch.cdist(anchors, generated).min(dim=1).values.mean()
            loss = loss + cfg.adversarial_weight * adv + cfg.cover_weight * cover
            if cfg.fm_weight > 0:
                real_feat = critic.features(data).detach().mean(0)
                fake_feat = critic.features(generated).mean(0)
                loss = loss + cfg.fm_weight * (real_feat - fake_feat).square().mean()
            adv_steps += 1
        loss.backward()
        for param in critic.parameters():
            param.requires_grad_(True)
        schedule_optimizer(opt_g, step - 1)
        opt_g.step()
        checkpoint(step, lambda: measure(step))
        if step == 1 or (step % 50 == 0 and step != cfg.steps):
            snap = measure(step)
            _log(cfg.name, step, snap, extra=f" loss={float(loss.detach()):.4f}")

    final = measure(cfg.steps)
    row = {
        "name": cfg.name,
        "cfg": cfg,
        "recon_mse": final["recon_mse"],
        "hold": final["hold"],
        "init_recon_mse": opened["recon_mse"],
        "penalty_applied": penalty_applied,
        "adv_steps": adv_steps,
        "steps": cfg.steps,
    }
    return row

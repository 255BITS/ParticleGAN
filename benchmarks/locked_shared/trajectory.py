"""Behavioral toy extracted from HyperGAN/conceptmod at 5571213.

Original budgets, models and numerical thresholds; no configuration gates.
See SOURCE.md and LICENSE for provenance. Default losses use PR #36 builders.
"""

from __future__ import annotations

import torch
from torch import nn
from particlegan.locked_shared import LOCKED_SHARED, make_gan_loss, make_b_cap

from particlegan import ParticlePrior, ParticleRegularizer

PROTOCOL = {
    "cover_weight": LOCKED_SHARED.cover_weight,
    "particle_l2": LOCKED_SHARED.particle_l2,
    "n_particles": LOCKED_SHARED.n_particles,
    "vicreg_weight": 0.05,
    "z_dim": 4,
    "frames": 8,
    "slow_speed": 0.45,
    "fast_speed": 2.2,
    "lr": 5.0e-3,
    "beta1": 0.0,
    "beta2": 0.99,
    "steps": 400,
    "seed": 0,
    "critic_hidden": 64,
}


PAIRINGS = ("shared", "stranger", "nearest_stranger")


PASS_IDENTITY_MSE = 0.02


def trajectories(n: int | None = None, frames: int | None = None):
    """Slow and fast arcs that share a seed-specific phase and radius.

    Returns ``(slow, fast)`` with shape ``[n, frames * 2]``.
    """
    n = PROTOCOL["n_particles"] if n is None else n
    frames = PROTOCOL["frames"] if frames is None else frames
    index = torch.arange(n, dtype=torch.float32)
    phase = 2 * torch.pi * index / n
    radius = 0.7 + 0.25 * ((index % 3) - 1)
    time = torch.linspace(0, 1, frames)
    slow_angle = phase[:, None] + PROTOCOL["slow_speed"] * time[None, :]
    fast_angle = phase[:, None] + PROTOCOL["fast_speed"] * time[None, :]

    def pack(angle: torch.Tensor) -> torch.Tensor:
        xy = radius[:, None, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
        return xy.reshape(n, frames * 2)

    return pack(slow_angle), pack(fast_angle)


def pairing_index(mode: str, slow: torch.Tensor) -> torch.Tensor:
    """Row index of the fast target paired with each slow identity."""
    if mode not in PAIRINGS:
        raise ValueError(f"unknown pairing {mode!r} (expected one of {PAIRINGS})")
    n = slow.shape[0]
    identity = torch.arange(n)
    if mode == "shared":
        return identity
    if mode == "stranger":
        if n % 2:
            raise ValueError("stranger shift needs an even number of identities")
        return (identity + n // 2) % n
    distance = torch.cdist(slow, slow)
    distance.fill_diagonal_(float("inf"))
    return distance.argmin(dim=1)


def identity_mse(pred: torch.Tensor, fast: torch.Tensor) -> float:
    return float((pred - fast).pow(2).mean())


def passed(mse: float) -> bool:
    return mse <= PASS_IDENTITY_MSE


class _Generator(nn.Module):
    def __init__(self, slow_dim: int, z_dim: int, fast_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slow_dim + z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, fast_dim),
        )

    def forward(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, z), dim=-1))


class _Critic(nn.Module):
    """Scores a slow/fast pair. This toy owns the critic; nothing is swapped."""

    def __init__(self, pair_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, fast), dim=-1)).squeeze(-1)


class _FastView(nn.Module):
    """Gradient penalty sees the fast arc; the slow arc stays conditioning."""

    def __init__(self, critic: _Critic) -> None:
        super().__init__()
        self.critic = critic

    def forward(self, fast: torch.Tensor) -> torch.Tensor:
        return self.critic(self.slow, fast).unsqueeze(-1)


def _cover(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Demo cover: each true fast arc must sit near some generated arc."""
    return torch.cdist(real, fake).min(dim=1).values.square().mean()


def train(*, pairing: str = "shared", gan_factory=None, cap_factory=None, diagnostics=False) -> dict:
    """Train and measure identity error; every pairing is allowed."""
    torch.set_num_threads(1)
    torch.manual_seed(PROTOCOL["seed"])
    slow, fast = trajectories()
    index = pairing_index(pairing, slow)
    paired = fast[index]
    hidden = PROTOCOL["critic_hidden"]
    generator = _Generator(slow.shape[1], PROTOCOL["z_dim"], fast.shape[1], hidden)
    critic = _Critic(slow.shape[1] + fast.shape[1], hidden)
    view = _FastView(critic)
    prior = ParticlePrior(
        PROTOCOL["n_particles"], PROTOCOL["z_dim"], init_std=0.1,
        generator=torch.Generator().manual_seed(PROTOCOL["seed"]),
    )
    gan = (gan_factory or make_gan_loss)()
    regularizer = (cap_factory or make_b_cap)()
    spread = ParticleRegularizer(weight=PROTOCOL["vicreg_weight"])
    opt_g = torch.optim.Adam(
        list(generator.parameters()) + list(prior.parameters()),
        lr=PROTOCOL["lr"], betas=(PROTOCOL["beta1"], PROTOCOL["beta2"]),
    )
    opt_d = torch.optim.Adam(
        critic.parameters(), lr=PROTOCOL["lr"], betas=(PROTOCOL["beta1"], PROTOCOL["beta2"]),
    )
    steps = PROTOCOL["steps"]
    for step in range(1, steps + 1):
        fake = generator(slow, prior.z)
        opt_d.zero_grad(set_to_none=True)
        d_loss = gan.d_loss(critic(slow, paired), critic(slow, fake.detach()))
        view.slow = slow.detach()
        d_loss = d_loss + regularizer(view, paired, fake.detach(), step=step)
        d_loss.backward()
        opt_d.step()

        flags = [p.requires_grad for p in critic.parameters()]
        critic.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            fake = generator(slow, prior.z)
            g_loss = gan.g_loss(critic(slow, fake), critic(slow, paired).detach())
            # Cover matches the true fast cloud (set coverage). It does not
            # retarget identity; only the relativistic pair does that.
            # fm_weight is 0: no feature-matching term is added.
            g_loss = g_loss + PROTOCOL["cover_weight"] * _cover(fake, fast)
            g_loss = g_loss + PROTOCOL["particle_l2"] * prior.z.square().mean()
            g_loss = g_loss + spread(prior.z)
            g_loss.backward()
            opt_g.step()
        finally:
            for parameter, flag in zip(critic.parameters(), flags):
                parameter.requires_grad_(flag)

    with torch.no_grad():
        pred = generator(slow, prior.z)
        mse = identity_mse(pred, fast)
        paired_mse = identity_mse(pred, paired)
    result = {
        "identity_mse": mse,
        "paired_target_mse": paired_mse,
        "verdict": "PASS" if passed(mse) else "FAIL",
    }
    if diagnostics:
        with torch.no_grad():
            distances = torch.cdist(pred, fast)
            result["set_cover"] = float(_cover(pred, fast))
            result["own_nearest_fraction"] = float((distances.argmin(1) == torch.arange(len(fast))).float().mean())
            result["particle_mean_square"] = float(prior.z.square().mean())
            result["particle_std_mean"] = float(prior.z.std(0).mean())
        norms = []
        for batch in (paired, pred):
            point = batch.detach().requires_grad_(True)
            grad = torch.autograd.grad(view(point).sum(), point)[0]
            norms.append(grad.norm(dim=1).detach())
        norms = torch.cat(norms)
        result["critic_gradient_median"] = float(norms.median())
        result["critic_gradient_max"] = float(norms.max())
    return result

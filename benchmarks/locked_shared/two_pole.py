"""Behavioral toy extracted from HyperGAN/conceptmod at 5571213.

Original budgets, models and numerical thresholds; no configuration gates.
See SOURCE.md and LICENSE for provenance. Default losses use PR #36 builders.
"""

from __future__ import annotations

from .observation import checkpoint

import torch
from torch import nn
from particlegan.locked_shared import LOCKED_SHARED, make_gan_loss, make_b_cap

TOY_STEPS = 80
TOY_SEED = 0
TOY_LR = 5e-3
TOY_BETAS = (0.0, 0.99)
TRAVEL_MIN = 0.30
GRAD_MED_MAX = 1.0
POLES = (-1.0, 1.0)
_HOST_W1 = (
    -0.007487, 0.536444, -0.823045, -0.735939, -0.385154, 0.268157, -0.019813,
    0.792889, -0.088744, 0.264613, -0.302213, -0.196565, -0.955348, -0.662282,
    -0.412223, 0.037044, 0.395335, 0.600023, -0.677941, -0.435463, 0.363217,
    0.830388, -0.205800, 0.748312, -0.161183, 0.105814, 0.905476, -0.927670,
    -0.629538, -0.253165, -0.389800, 0.864001,
)


_HOST_B1 = (
    -0.648180, -0.460333, -0.698640, -0.936561, -0.583740, 0.859598, 0.446218,
    0.484673, 0.052592, -0.512684, 0.169185, -0.933695, -0.722566, -0.515530,
    0.630938, 0.586321, -0.443495, -0.036082, 0.639561, 0.994133, 0.396882,
    0.135093, 0.670486, -0.588802, 0.186344, -0.775306, -0.693086, -0.516584,
    0.452473, 0.402160, -0.592353, 0.302107,
)


_HOST_W2 = (
    0.097045, -0.022312, 0.006750, 0.040960, 0.109668, 0.169740, -0.136228,
    -0.064783, 0.069475, 0.146468, 0.153832, 0.155980, 0.035181, -0.153722,
    0.016262, -0.110592, -0.164748, 0.157065, 0.134414, -0.176340, 0.033088,
    -0.029780, -0.029091, -0.080921, 0.067981, -0.104705, 0.064805, 0.089397,
    0.126549, 0.066099, -0.174962, -0.114674,
)


_HOST_B2 = (0.088267,)


class HostCritic(nn.Module):
    """1-D critic pinned to the stored host weights. Not an arch menu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(_HOST_W1).reshape(32, 1))
            self.fc1.bias.copy_(torch.tensor(_HOST_B1))
            self.fc2.weight.copy_(torch.tensor(_HOST_W2).reshape(1, 32))
            self.fc2.bias.copy_(torch.tensor(_HOST_B2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.silu(self.fc1(x))).squeeze(-1)


def real_batch(n: int) -> torch.Tensor:
    """Balanced poles at ±1 with a fixed ±0.05 spread. No RNG."""
    half = n // 2
    offset = torch.linspace(-0.05, 0.05, half)
    return torch.cat([-1.0 + offset, 1.0 + offset]).unsqueeze(1)


def _grad_median(critic: nn.Module, real: torch.Tensor, particles: torch.Tensor) -> float:
    xs = torch.cat([real, particles.detach()]).detach().requires_grad_(True)
    grad = torch.autograd.grad(critic(xs).sum(), xs, create_graph=False)[0]
    return float(grad.flatten().abs().median())


def _nearest(particles: torch.Tensor) -> float:
    poles = particles.new_tensor(POLES)
    return float((particles.flatten().unsqueeze(1) - poles).abs().min(dim=1).values.mean())


def cell_wins(mean_abs: float, grad_med: float) -> bool:
    """Travel off the origin, and the b_cap median slope stays ≤ kappa.

    Both-pole balance and cover_score are logged elsewhere. Gating this cell
    on them would crown a thinned hinge that walks farther than locked_shared.
    """
    return mean_abs >= TRAVEL_MIN and grad_med <= GRAD_MED_MAX


def train(*, pairing="live", gan_factory=None, cap_factory=None, particle_l2=None) -> dict:
    """Run the original 80-step cloud experiment, including stranger arms."""
    torch.manual_seed(TOY_SEED)
    particle_l2 = LOCKED_SHARED.particle_l2 if particle_l2 is None else particle_l2
    critic = HostCritic()
    particles = nn.Parameter(torch.zeros(LOCKED_SHARED.n_particles, 1))
    opt_d = torch.optim.Adam(critic.parameters(), lr=TOY_LR, betas=TOY_BETAS)
    opt_p = torch.optim.Adam([particles], lr=TOY_LR, betas=TOY_BETAS)
    gan = (gan_factory or make_gan_loss)()
    regularizer = (cap_factory or make_b_cap)()
    real = real_batch(LOCKED_SHARED.n_particles)
    stranger = torch.linspace(-3.0, 3.0, LOCKED_SHARED.n_particles).unsqueeze(1)
    for step in range(1, TOY_STEPS + 1):
        opt_d.zero_grad(set_to_none=True)
        fake = particles.detach() if pairing == "live" else stranger
        d_loss = gan.d_loss(critic(real), critic(fake))
        (d_loss + regularizer(critic, real, fake, step=step)).backward()
        opt_d.step()

        opt_p.zero_grad(set_to_none=True)
        d_real = critic(real).detach()
        paired = critic(particles) if pairing == "live" else critic(stranger)
        g_loss = gan.g_loss(paired, d_real)
        g_loss = g_loss + particle_l2 * particles.square().mean()
        g_loss.backward()
        opt_p.step()
        checkpoint(step, lambda: {"mean_abs": float(particles.detach().abs().mean()),
                                 "grad_med": _grad_median(critic, real, particles)})

    with torch.no_grad():
        mean_abs = float(particles.abs().mean())
        nearest = _nearest(particles)
    grad_med = _grad_median(critic, real, particles)
    return {
        "mean_abs": mean_abs,
        "grad_med": grad_med,
        "nearest": nearest,
        "cover_score": LOCKED_SHARED.cover_weight * (1.0 - min(nearest, 1.0)),
        "verdict": "PASS" if cell_wins(mean_abs, grad_med) else "FAIL",
    }

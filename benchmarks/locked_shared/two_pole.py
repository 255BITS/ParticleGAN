"""Two-pole cloud: does a 12-particle cloud travel to the poles at +-1 while
the critic's median input slope stays bounded?

Problem only, from HyperGAN/conceptmod at 5571213 (``leaderboard_honesty.py``;
see SOURCE.md and LICENSE): the stored host critic, the deterministic
two-pole data, the particle table (the samples themselves), the live/stranger
pairing, the particle L2 pull, the travel/slope metrics and the verdict.
Everything else (optimizers and their LR schedule, loss, critic penalty,
noise, EMA, observation logging) comes from the shipped recipe through
``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.two_pole --log runs/toy-refactor/locked_two_pole.log
"""

from __future__ import annotations

import torch
from torch import nn

from particlegan import get_recipe
from benchmarks.toy_runner import Networks, ToyProblem, main, run
from .observation import checkpoint

TOY_STEPS = 80
N_PARTICLES = 12
PARTICLE_L2 = 0.02
COVER_WEIGHT = 1.5  # logged score only; not part of the verdict
TRAVEL_MIN = 0.30
GRAD_MED_MAX = 1.0
POLES = (-1.0, 1.0)
PAIRINGS = ("live", "stranger")

# Stored host critic weights (conceptmod 5571213): explicit, not an init recipe.
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


def stranger_batch(n: int) -> torch.Tensor:
    """The stranger arm's fixed fake batch: the critic never sees the particles."""
    return torch.linspace(-3.0, 3.0, n).unsqueeze(1)


def _grad_median(critic: nn.Module, real: torch.Tensor, particles: torch.Tensor) -> float:
    with torch.enable_grad():
        xs = torch.cat([real, particles.detach()]).detach().requires_grad_(True)
        grad = torch.autograd.grad(critic(xs).sum(), xs, create_graph=False)[0]
    return float(grad.flatten().abs().median())


def _nearest(particles: torch.Tensor) -> float:
    poles = particles.new_tensor(POLES)
    return float((particles.flatten().unsqueeze(1) - poles).abs().min(dim=1).values.mean())


def cell_wins(mean_abs: float, grad_med: float) -> bool:
    """Travel off the origin, and the median critic slope stays ≤ kappa.

    Both-pole balance and cover_score are logged elsewhere. Gating this cell
    on them would crown a thinned hinge that walks farther than locked_shared.
    """
    return mean_abs >= TRAVEL_MIN and grad_med <= GRAD_MED_MAX


class ParticleCloud(nn.Module):
    """The samples themselves: a 1-D particle table starting at the origin."""

    def __init__(self, n: int = N_PARTICLES) -> None:
        super().__init__()
        self.particles = nn.Parameter(torch.zeros(n, 1))


class TwoPole(ToyProblem):
    """The two-pole cloud. ``pairing="stranger"`` shows the critic a fixed
    linspace instead of the particles (so only the L2 pull reaches them);
    ``particle_l2`` weights the pull toward the origin the cloud must beat."""

    name = "locked_two_pole"

    def __init__(self, *, pairing: str = "live", particle_l2: float = PARTICLE_L2):
        if pairing not in PAIRINGS:
            raise ValueError(f"pairing must be one of {PAIRINGS}")
        self.pairing, self.particle_l2 = pairing, float(particle_l2)

    def recipe(self):
        return get_recipe(z_dim=1, num_particles=N_PARTICLES, batch_size=N_PARTICLES, total_steps=TOY_STEPS)

    def networks(self, recipe, seed):
        cloud = ParticleCloud(recipe.num_particles)
        return Networks(generator=cloud, critics=HostCritic(), prior=None, direct_particles=[cloud.particles])

    def real(self, n, stream):
        return real_batch(n)

    def fake(self, nets, n, stream, real):
        particles = nets.generator.particles
        if n != len(particles):
            raise ValueError("the two-pole cloud is scored as one full batch of its particles")
        if self.pairing == "stranger" and real is not None:
            return stranger_batch(n)
        return particles[:n]  # a view: detached under the critic step's no_grad

    def losses(self, role, nets, real, fake):
        if role != "generator" or self.particle_l2 == 0:
            return {}
        return {"particle_l2": self.particle_l2 * nets.generator.particles.square().mean()}

    def metrics(self, model):
        particles = model.nets.generator.particles
        nearest = _nearest(particles)
        return {"mean_abs": float(particles.abs().mean()),
                "grad_med": _grad_median(model.nets.critics, real_batch(len(particles)), particles),
                "nearest": nearest,
                "cover_score": COVER_WEIGHT * (1.0 - min(nearest, 1.0))}

    def verdict(self, metrics):
        return "PASS" if cell_wins(metrics["mean_abs"], metrics["grad_med"]) else "FAIL"


def train(*, pairing: str = "live", particle_l2: float | None = None, steps: int | None = None,
          log=None, log_path=None) -> dict:
    """Train the cloud on the shared runner; the final live row with its verdict
    (``ema`` and ``hold`` alongside). Observations go to
    ``benchmarks.locked_shared.observation`` recorders."""
    problem = TwoPole(pairing=pairing, particle_l2=PARTICLE_L2 if particle_l2 is None else particle_l2)
    result = run(problem, steps=steps, log=log, log_path=log_path, observer=checkpoint)
    return {**result["live"], "ema": result["ema"], "hold": result["hold"]}


if __name__ == "__main__":
    raise SystemExit(main(TwoPole()))

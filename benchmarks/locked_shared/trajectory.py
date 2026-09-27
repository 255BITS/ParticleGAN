"""Locked trajectory toy: does a conditional generator keep each identity's fast arc?

Problem only, from HyperGAN/conceptmod at 5571213 (see SOURCE.md and LICENSE):
12 identities, each a slow arc (the conditioning) and a fast arc (the target)
sharing a phase and radius; a conditional generator ``G(slow, z_i)`` with one
prior particle per identity; a conditional critic that scores (fast | slow);
the set-cover term; and the identity-MSE gate. The ``pairing`` arm chooses
which fast arc the critic treats as real for each slow arc: ``shared`` (its
own), ``stranger`` (half a turn away) or ``nearest_stranger``.

Everything else -- optimizers and their LR schedule, loss, critic penalty,
prior regularization, noise, EMA, observation logging -- comes from the
shipped recipe through ``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.trajectory --log runs/toy-refactor/locked_trajectory.log
"""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, Sample, ToyProblem, main, run
from .observation import checkpoint

N_IDENTITIES = 12
Z_DIM = 4
FRAMES = 8
SLOW_SPEED = 0.45
FAST_SPEED = 2.2
HIDDEN = 64
PARTICLE_INIT_STD = 0.1  # the host's tiny latent cloud
COVER_WEIGHT = 1.5
STEPS = 400
PAIRINGS = ("shared", "stranger", "nearest_stranger")
PASS_IDENTITY_MSE = 0.02

# The original host's optimizer/regularizer card, frozen and unused here: it
# is read only by hosts/residual_student.py (which reuses this data), and goes
# away when that host moves onto the shared runner.
PROTOCOL = MappingProxyType({
    "cover_weight": COVER_WEIGHT, "particle_l2": 0.02, "n_particles": N_IDENTITIES,
    "vicreg_weight": 0.05, "z_dim": Z_DIM, "frames": FRAMES, "slow_speed": SLOW_SPEED,
    "fast_speed": FAST_SPEED, "lr": 5.0e-3, "beta1": 0.0, "beta2": 0.99, "steps": STEPS,
    "seed": 0, "critic_hidden": HIDDEN,
})


def trajectories(n: int = N_IDENTITIES, frames: int = FRAMES):
    """Slow and fast arcs that share a seed-specific phase and radius.

    Returns ``(slow, fast)`` with shape ``[n, frames * 2]``.
    """
    index = torch.arange(n, dtype=torch.float32)
    phase = 2 * torch.pi * index / n
    radius = 0.7 + 0.25 * ((index % 3) - 1)
    time = torch.linspace(0, 1, frames)

    def pack(speed: float) -> torch.Tensor:
        angle = phase[:, None] + speed * time[None, :]
        xy = radius[:, None, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)
        return xy.reshape(n, frames * 2)

    return pack(SLOW_SPEED), pack(FAST_SPEED)


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


def _cover(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Set cover: each true fast arc must sit near some generated arc."""
    return torch.cdist(real, fake).min(dim=1).values.square().mean()


def _mlp(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, HIDDEN), nn.LeakyReLU(0.2), nn.Linear(HIDDEN, HIDDEN),
                         nn.LeakyReLU(0.2), nn.Linear(HIDDEN, outputs))


class _Generator(nn.Module):
    """Fast arc from (slow arc, identity particle)."""

    def __init__(self, slow_dim: int, fast_dim: int) -> None:
        super().__init__()
        self.net = _mlp(slow_dim + Z_DIM, fast_dim)

    def forward(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, z), dim=-1))


class _Critic(nn.Module):
    """Scores a fast arc given its slow arc: ``critic(fast, slow)``."""

    def __init__(self, fast_dim: int, slow_dim: int) -> None:
        super().__init__()
        self.net = _mlp(fast_dim + slow_dim, 1)

    def forward(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((fast, slow), dim=-1))


class Trajectory(ToyProblem):
    """Every batch is the whole set of identities: particle ``i`` generates
    identity ``i``'s fast arc from its slow arc. ``detailed`` adds set-cover,
    nearest-identity, particle-cloud and critic-slope diagnostics."""

    name = "locked_trajectory"

    def __init__(self, *, pairing: str = "shared", cover_weight: float = COVER_WEIGHT, detailed: bool = False):
        self.slow, self.fast = trajectories()
        self.pairing = pairing
        self.paired = self.fast[pairing_index(pairing, self.slow)]
        self.cover_weight, self.detailed = float(cover_weight), detailed

    def recipe(self):
        return get_recipe(z_dim=Z_DIM, num_particles=N_IDENTITIES, batch_size=N_IDENTITIES, total_steps=STEPS)

    def networks(self, recipe, seed):
        slow_dim, fast_dim = self.slow.shape[1], self.fast.shape[1]
        generator = init.deterministic_orthogonal_(_Generator(slow_dim, fast_dim), seed=seed)
        critic = init.deterministic_orthogonal_(_Critic(fast_dim, slow_dim), seed=seed + 1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(init_std=PARTICLE_INIT_STD), seed=seed)
        return Networks(generator=generator, critics=critic, prior=prior)

    def _whole_set(self, n):
        if n != N_IDENTITIES:
            raise ValueError(f"every batch is the whole set of {N_IDENTITIES} identities")

    def real(self, n, stream):
        self._whole_set(n)
        return Sample(self.paired, condition=(self.slow,))

    def fake(self, nets, n, stream, real):
        self._whole_set(n)
        return Sample(nets.generator(self.slow, nets.prior.z), condition=(self.slow,),
                      indices=torch.arange(N_IDENTITIES))

    def losses(self, role, nets, real, fake):
        # Cover matches the true fast cloud (set coverage). It does not
        # retarget identity; only the critic's pairing does that.
        if role != "generator" or self.cover_weight == 0:
            return {}
        return {"cover": self.cover_weight * _cover(fake.x, self.fast)}

    def metrics(self, model):
        pred = model.sample(N_IDENTITIES).x
        row = {"identity_mse": identity_mse(pred, self.fast), "paired_target_mse": identity_mse(pred, self.paired)}
        if self.detailed:
            z = model.nets.prior.z
            row.update(set_cover=float(_cover(pred, self.fast)),
                       own_nearest_fraction=float((torch.cdist(pred, self.fast).argmin(1)
                                                   == torch.arange(N_IDENTITIES)).float().mean()),
                       particle_mean_square=float(z.square().mean()), particle_std_mean=float(z.std(0).mean()))
            norms = []
            with torch.enable_grad():
                for batch in (self.paired, pred):
                    point = batch.detach().requires_grad_(True)
                    grad = torch.autograd.grad(model.nets.critics(point, self.slow).sum(), point)[0]
                    norms.append(grad.norm(dim=1))
            norms = torch.cat(norms)
            row.update(critic_gradient_median=float(norms.median()), critic_gradient_max=float(norms.max()))
        return row

    def verdict(self, metrics):
        return "PASS" if passed(metrics["identity_mse"]) else "FAIL"


def train(*, pairing: str = "shared", diagnostics: bool = False, steps: int | None = None, log=None) -> dict:
    """Train on the shared runner; the live row with its verdict (as the
    original host scored it), plus the EMA row and hold summary.

    Observations go to ``benchmarks.locked_shared.observation`` recorders.
    """
    torch.set_num_threads(1)  # the harness protocol runs hosts single-threaded
    result = run(Trajectory(pairing=pairing, detailed=diagnostics), steps=steps, log=log, observer=checkpoint)
    return {**result["live"], "ema": result["ema"], "hold": result["hold"]}


if __name__ == "__main__":
    raise SystemExit(main(Trajectory()))

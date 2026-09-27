"""Residual student toy: can a residual head turn each slow arc into its own fast arc?

Problem only, from HyperGAN/conceptmod at 5571213 (see ../SOURCE.md and
../LICENSE): the paired slow/fast trajectories, the residual head and the
conditional pair critic, the both-land residual target and the demo cover
term, the landing/identity metrics and the verdict. Everything else
(optimizers and their LR schedule, loss, critic penalty, prior and its
regularizer, noise, EMA, observation logging) comes from the shipped recipe
through ``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.hosts.residual_student --log runs/toy-refactor/locked_residual_student.log

Particle ``i`` of the 12-row table is identity ``i``: every update is the
full batch of identities, ``head(slow, prior.z)`` in row order.
"""

from __future__ import annotations

import torch
from torch import nn

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, Sample, ToyProblem, main, run
from ..observation import checkpoint
from ..trajectory import PASS_IDENTITY_MSE, identity_mse, pairing_index, trajectories

N_IDENTITIES = 12
FRAMES = 8
Z_DIM = 4
HIDDEN = 64
STEPS = 400
SLOW_IMPACT_MAX = 0.10
LAND_TOL = 0.25
SUCCESS_MIN = 1.0
RESIDUAL_WEIGHT = 1.0
COVER_WEIGHT = 1.5


def _xy(arc: torch.Tensor) -> torch.Tensor:
    if arc.shape[-1] != FRAMES * 2:
        raise ValueError(f"expected arc dim {FRAMES * 2}, got {arc.shape[-1]}")
    return arc.reshape(arc.shape[0], FRAMES, 2)


def endpoints(arc: torch.Tensor) -> torch.Tensor:
    """Last (x, y) of each packed arc."""
    return _xy(arc)[:, -1]


def impact(arc: torch.Tensor) -> torch.Tensor:
    """Chord of the last step. This is the touchdown speed proxy."""
    xy = _xy(arc)
    return (xy[:, -1] - xy[:, -2]).norm(dim=-1)


def both_land_mask(slow: torch.Tensor, fast: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rows whose slow teacher and paired fast teacher both succeed here.

    The slow teacher must make a gentle touchdown. The paired fast teacher
    must finish on this seed's pad. Only the same-seed fast arc does that.
    """
    if index.shape != (slow.shape[0],):
        raise ValueError("pairing index must be one entry per identity")
    slow_lands = impact(slow) <= SLOW_IMPACT_MAX
    own_pad = endpoints(fast)
    fast_lands = (own_pad[index] - own_pad).norm(dim=-1) <= LAND_TOL
    return slow_lands & fast_lands


def landing_stats(pred: torch.Tensor, fast: torch.Tensor) -> dict:
    """Own-pad success and wrong-pad crashes for a predicted fast arc."""
    pad = endpoints(fast)
    end = endpoints(pred)
    dist = (end - pad).norm(dim=-1)
    nearest = torch.cdist(end, pad).argmin(dim=1)
    return {
        "success_rate": float((dist <= LAND_TOL).float().mean()),
        "wrong_pad_rate": float((nearest != torch.arange(end.shape[0])).float().mean()),
        "endpoint_l2": float(dist.mean()),
    }


def passed(mse: float, success_rate: float, wrong_pad_rate: float) -> bool:
    """Identity, every seed on its own pad, and no wrong-pad touchdown."""
    return mse <= PASS_IDENTITY_MSE and success_rate >= SUCCESS_MIN and wrong_pad_rate == 0.0


class ResidualHead(nn.Module):
    """``slow + head(slow, z)``. A zero head leaves the slow arc untouched."""

    def __init__(self, slow_dim: int, z_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slow_dim + z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, slow_dim),
        )

    def delta(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, z), dim=-1))

    def forward(self, slow: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return slow + self.delta(slow, z)


class PairCritic(nn.Module):
    """Scores a fast arc given its slow arc: ``critic(fast, slow)``.

    The fast arc is the critic input (noise and the gradient penalty act on
    it); the slow arc is conditioning.
    """

    def __init__(self, pair_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((slow, fast), dim=-1)).squeeze(-1)


def _cover(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Demo cover: each true fast arc must sit near some generated arc."""
    return torch.cdist(real, fake).min(dim=1).values.square().mean()


class ResidualStudent(ToyProblem):
    """The residual student on paired trajectories.

    ``pairing`` picks which fast arc the critic sees as each slow arc's real
    partner ("shared", "stranger" or "nearest_stranger"). The residual term
    pulls the student to the true fast arc on both-land rows; the cover term
    asks every true fast arc to sit near some generated arc (set coverage,
    not identity). Both are part of this problem's objective.
    """

    name = "locked_residual_student"

    def __init__(self, *, pairing: str = "shared", residual_weight: float = RESIDUAL_WEIGHT,
                 cover_weight: float = COVER_WEIGHT):
        self.pairing = pairing
        self.residual_weight, self.cover_weight = float(residual_weight), float(cover_weight)
        self.slow, self.fast = trajectories(N_IDENTITIES, FRAMES)
        index = pairing_index(pairing, self.slow)
        self.paired = self.fast[index]
        self.mask = both_land_mask(self.slow, self.fast, index)

    def recipe(self):
        return get_recipe(z_dim=Z_DIM, num_particles=N_IDENTITIES, batch_size=N_IDENTITIES, total_steps=STEPS)

    def networks(self, recipe, seed):
        dim = self.slow.shape[1]
        head = init.deterministic_orthogonal_(ResidualHead(dim, Z_DIM, HIDDEN), seed=seed)
        critic = init.deterministic_orthogonal_(PairCritic(dim + self.fast.shape[1], HIDDEN), seed=seed + 1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(), seed=seed)
        return Networks(generator=head, critics=critic, prior=prior)

    def real(self, n, stream):
        if n != N_IDENTITIES:
            raise ValueError(f"{self.name} trains on the full batch of {N_IDENTITIES} identities")
        return Sample(self.paired, condition=(self.slow,))

    def fake(self, nets, n, stream, real):
        if n != N_IDENTITIES:
            raise ValueError(f"{self.name} samples the full batch of {N_IDENTITIES} identities")
        return Sample(nets.generator(self.slow, nets.prior.z), condition=(self.slow,),
                      indices=torch.arange(N_IDENTITIES))

    def losses(self, role, nets, real, fake):
        if role != "generator":
            return {}
        terms = {"cover": self.cover_weight * _cover(fake.x, self.fast)}
        if self.mask.any():
            residual = (fake.x[self.mask] - self.fast[self.mask]).pow(2).mean()
            terms["residual"] = self.residual_weight * residual
        return terms

    def metrics(self, model):
        pred = model.sample(N_IDENTITIES).x
        return {"identity_mse": identity_mse(pred, self.fast),
                "paired_target_mse": identity_mse(pred, self.paired),
                **landing_stats(pred, self.fast)}

    def verdict(self, metrics):
        ok = passed(metrics["identity_mse"], metrics["success_rate"], metrics["wrong_pad_rate"])
        return "PASS" if ok else "FAIL"


def train_residual_student(problem: ResidualStudent | None = None, *, seed: int = 0, log=None,
                           steps: int | None = None, recipe=None) -> dict:
    """Train on the shared runner; the EMA row with its verdict plus ``live``,
    ``live_curve`` and ``hold``. Observations also go to
    ``benchmarks.locked_shared.observation`` recorders."""
    problem = ResidualStudent() if problem is None else problem
    result = run(problem, recipe=recipe, seed=seed, steps=steps, log=log, observer=checkpoint)
    return {**result["ema"], "step": result["steps"], "seed": seed, "pairing": problem.pairing,
            "both_land_rows": int(problem.mask.sum()), "live": result["live"],
            "live_curve": result["curve"], "hold": result["hold"]}


if __name__ == "__main__":
    raise SystemExit(main(ResidualStudent()))

"""Unipolar toy: can a free-origin residual cover the + pole without moving scale 0?

Problem only, from HyperGAN/conceptmod at 5571213 (see ../SOURCE.md and
../LICENSE): the residual student, the scale-conditioned critic, the two-scale
targets ({0, +1}, averaged 0.5/0.5), the unipolar gates and the verdict.
Everything else (optimizers and their LR schedule, loss, critic penalty,
noise, EMA, observation logging) comes from the shipped recipe through
``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.hosts.unipolar --log runs/toy-refactor/locked_unipolar.log
    python -m benchmarks.locked_shared.hosts.unipolar --arm mse_only

Arms: ``locked_rpgan`` (the GAN arm), ``polarity_flipped`` (trains on the
- pole; the gates still read the true + pole, so it should FAIL) and
``mse_only`` (student-only: no critic, the plus-only MSE is its sole loss,
optimized by the same recipe generator optimizer).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, ToyProblem, View, main, run
from ..observation import checkpoint

PLUS_COVER_MIN = 0.85
PLUS_OFF_MAX = 0.05
PLUS_NEU_HOLD_MIN = 0.85
DIM = 4
N_ROWS = 8
SCALES = (0.0, 1.0)
STEPS = 400
CRITIC_HIDDEN = 64
PLUS = torch.tensor([1.0, 0.0, 0.0, 0.0])
OFF_AXIS = torch.tensor([0.0, 1.0, 0.0, 0.0])
ARMS = ("locked_rpgan", "mse_only", "polarity_flipped")


class FreeOriginResidual(nn.Module):
    """``delta(s) = s*odd + |s|*even + origin``. Origin is free, so hold is learned.

    Deleting ``origin`` would make neu_hold true by construction. Plus-only MSE
    can park the + pole in ``origin`` and fail hold; scale-0 GAN training cannot.
    """

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(dim))
        self.even = nn.Parameter(torch.zeros(dim))
        self.origin = nn.Parameter(torch.zeros(dim))

    def delta(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return s * self.odd + abs(s) * self.even + self.origin


# The zero start is the problem (the residual begins at the identity).
init.register(FreeOriginResidual, {"odd": init.KEEP, "even": init.KEEP, "origin": init.KEEP})


class ScaleCritic(nn.Module):
    """Two-layer LeakyReLU MLP on (delta / teacher RMS, scale label)."""

    def __init__(self, dim: int, teacher: torch.Tensor, hidden: int = CRITIC_HIDDEN) -> None:
        super().__init__()
        rms = teacher.detach().float().square().mean().sqrt()
        if not torch.isfinite(rms) or float(rms) <= 0.0:
            raise ValueError("teacher must have finite, nonzero RMS")
        self.register_buffer("input_scale", rms)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def score(self, z: torch.Tensor, scale: float) -> torch.Tensor:
        label = z.new_full((z.shape[0], 1), float(scale))
        return self.net(torch.cat([z, label], dim=-1)).squeeze(-1)

    def forward(self, delta: torch.Tensor, scale: float) -> torch.Tensor:
        return self.score(delta.float() / self.input_scale, scale)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.detach().float().unsqueeze(0), b.detach().float().unsqueeze(0)).squeeze())


def score_residual(student: FreeOriginResidual) -> dict:
    """Unipolar gates. They always read the true plus pole, never the training pole."""
    plus, off = PLUS.to(student.odd.device), OFF_AXIS.to(student.odd.device)
    d1 = student.delta(1.0).detach()
    d0 = student.delta(0.0).detach()
    d_half = student.delta(0.5).detach()
    d_minus = student.delta(-1.0).detach()
    target_norm = float(plus.norm())
    cos_plus = _cos(d1, plus)
    mag_ratio = float(d1.norm()) / target_norm
    mag_score = max(0.0, 1.0 - abs(mag_ratio - 1.0))
    cover = max(0.0, cos_plus) * mag_score
    leak = float(d1.dot(off) ** 2) / (float(d1.dot(d1)) + 1e-12)
    neu_hold = 1.0 - min(1.0, float(d0.norm()) / target_norm)
    half_ratio = float(d_half.norm()) / (float(d1.norm()) + 1e-12)
    cos_canary_plus = _cos(d_minus, plus)
    cos_canary_neg = _cos(d_minus, -plus)
    canary_off = float(d_minus.dot(off) ** 2) / (float(d_minus.dot(d_minus)) + 1e-12)
    if cos_canary_plus > 0.5:
        landed = "pos"
    elif cos_canary_neg > 0.5:
        landed = "neg"
    else:
        landed = "other"
    dangerous = bool(landed == "pos" or canary_off > PLUS_OFF_MAX)
    hit = bool(cover >= PLUS_COVER_MIN and leak <= PLUS_OFF_MAX and neu_hold >= PLUS_NEU_HOLD_MIN)
    return {
        "cover": cover,
        "off_caption": leak,
        "neu_hold": neu_hold,
        "cos_plus": cos_plus,
        "mag_ratio": mag_ratio,
        "half_norm_ratio": half_ratio,
        "origin_norm": float(student.origin.detach().norm()),
        "odd_norm": float(student.odd.detach().norm()),
        "even_norm": float(student.even.detach().norm()),
        "hit": hit,
        "canary": {
            "scored": False,
            "landed": landed,
            "cos_plus": cos_canary_plus,
            "cos_neg": cos_canary_neg,
            "off_caption": canary_off,
            "dangerous": dangerous,
        },
    }


class Unipolar(ToyProblem):
    """Scale 0 must stay at the origin while scale +1 lands on the (training) pole.

    A batch is ``N_ROWS`` copies per scale, scale 0 rows first. The critic
    reads the scale label as its condition; each scale is one view at weight
    ``1/len(SCALES)``.
    """

    def __init__(self, arm: str = "locked_rpgan"):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; this family is {', '.join(ARMS)}")
        self.arm = arm
        self.name = "unipolar" if arm == "locked_rpgan" else f"unipolar_{arm}"
        self.target = (-PLUS if arm == "polarity_flipped" else PLUS).clone()

    def recipe(self):
        return get_recipe(batch_size=N_ROWS, total_steps=STEPS)

    def networks(self, recipe, seed):
        student = init.deterministic_orthogonal_(FreeOriginResidual(DIM), seed=seed)
        if self.arm == "mse_only":
            return Networks(generator=student, critics={}, prior=None)
        teacher = self.target.unsqueeze(0).expand(N_ROWS, -1)
        critic = init.deterministic_orthogonal_(ScaleCritic(DIM, teacher), seed=seed + 1)
        return Networks(generator=student, critics=critic, prior=None)

    def real(self, n, stream):
        target = self.target.to(stream.device)
        return torch.cat([(s * target).expand(n, -1) for s in SCALES])

    def fake(self, nets, n, stream, real):
        return torch.cat([nets.generator.delta(s).unsqueeze(0).expand(n, -1) for s in SCALES])

    def views(self, nets, real, fake):
        if not nets.critics:
            return []
        n = real.x.shape[0] // len(SCALES)
        return [View("critic", real.x[i * n:(i + 1) * n], fake.x[i * n:(i + 1) * n], (s,), 1.0 / len(SCALES))
                for i, s in enumerate(SCALES)]

    def losses(self, role, nets, real, fake):
        if role != "generator" or self.arm != "mse_only":
            return {}
        # Plus-only coordinate MSE on the noise-free residual; scale 0 is not in it.
        return {"mse": F.mse_loss(nets.generator.delta(1.0), self.target.to(nets.generator.odd.device))}

    def metrics(self, model):
        return score_residual(model.nets.generator)

    def verdict(self, metrics):
        return "PASS" if metrics["hit"] else "FAIL"


def run_arm(arm: str = "locked_rpgan", *, steps: int | None = None, seed: int = 0, log=None,
            log_path=None, recipe=None) -> dict:
    """Train one arm on the shared runner. Top level is the EMA row (like
    ``mode_hold.train_mode_hold``); ``live`` holds the live row, ``curve`` the
    live observations and ``hold`` their summary. Observations also go to
    ``benchmarks.locked_shared.observation`` recorders."""
    result = run(Unipolar(arm), recipe=recipe, steps=steps, seed=seed, observe_every=50,
                 log=log, log_path=log_path, observer=checkpoint)
    return {**result["ema"], "arm": arm, "steps": result["steps"], "seed": seed, "live": result["live"],
            "live_curve": result["curve"], "hold": result["hold"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=ARMS, default="locked_rpgan")
    known, rest = parser.parse_known_args()
    raise SystemExit(main(Unipolar(known.arm), rest))

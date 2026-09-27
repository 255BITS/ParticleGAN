"""Mid-scale identity toy: does a four-scale residual keep the person at 0.5?

Problem only, from HyperGAN/conceptmod at 5571213 (see ../SOURCE.md and
../LICENSE): the smile teacher (guarded concept, identity, stranger), the
``s*odd + |s|*even + origin + bump(s)*mid`` residual student, the
scale-conditioned critic architecture, the per-scale views with their 1/4
averaging, the problem's cover constraint, and the polarity / magnitude /
identity metrics with their verdict. Everything else (optimizers and their LR
schedule, loss, critic penalty, input/output noise, EMA, observation logging)
comes from the shipped recipe through ``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.hosts.mid_scale_identity --log runs/toy-refactor/mid_scale_identity.log
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, ToyProblem, View, main, run
from ..observation import checkpoint
from .cover_leftover import (
    LOCKED_COVER,
    LeftoverField,
    faithful_guard_e,
    hold_dir,
    leftover_bipolar,
)

EVAL_SCALES = (-1.0, 0.0, 0.5, 1.0)
ANIMA_SMILE_SCALES = (0.0, 0.25, 0.5, 1.0)
CONCEPT_COS_MIN = 0.85
CONCEPT_MAG_LO = 0.75
CONCEPT_MAG_HI = 1.25
IDENTITY_KEPT_MIN = 0.85
GATE_STEPS = 800
DIM = 4
N_ROWS = 8
CRITIC_HIDDEN = 64
COVER_WEIGHT = LOCKED_COVER
"""Weight of the problem's constraint ``mean_s mse(state(s), target(s))``."""

ARMS = (
    "locked",
    "mid_collapse",
    "missing_minus",
    "polarity_flipped",
    "stranger",
)

REASON_ORDER = (
    "missing_minus",
    "incomplete_grid",
    "polarity",
    "concept",
    "identity_0",
    "identity_mid",
    "stranger_pairing",
)


def mid_bump(scale: float) -> float:
    """One-sided smile bump. 0 at ``{-1, 0, 1}``, 1 at ``0.5``.

    The negative pole is not a mid-scale. A residual can sit on the
    concept at both poles and on the person at 0, and still replace
    the person at ``+0.5``.
    """
    s = float(scale)
    if s <= 0.0 or s >= 1.0:
        return 0.0
    return 4.0 * s * (1.0 - s)


def _has_scale(scales, target: float) -> bool:
    return any(abs(float(scale) - float(target)) <= 1e-6 for scale in scales)


@dataclass(frozen=True, eq=False)
class SmileTeacher:
    """Guarded concept, the person, and an orthogonal stranger."""

    identity: torch.Tensor
    concept: torch.Tensor
    stranger: torch.Tensor
    retain_unit: torch.Tensor
    identity_amp: float
    plus: torch.Tensor
    minus: torch.Tensor

    def train_target(self, arm: str, scale: float) -> torch.Tensor:
        """Teacher state for one train scale. Eval never reads this."""
        polarity = -1.0 if arm == "polarity_flipped" else 1.0
        concept = polarity * self.concept
        if arm == "stranger" or (arm == "mid_collapse" and abs(float(scale) - 0.5) <= 1e-6):
            base = self.stranger
        else:
            base = self.identity
        return base + float(scale) * concept


def smile_teacher(field: LeftoverField | None = None) -> SmileTeacher:
    """Identity on the content axis, concept on û, stranger on the lyric axis.

    Raw poles carry leftover ê. ``faithful_guard_e`` takes ê off the odd
    part when the blend guard admits it. The concept this toy scores is
    that guarded odd direction.
    """
    field = field or LeftoverField()
    concept_axis = field.basis(0)
    content_axis = field.basis(1)
    leak_axis = field.basis(2)
    retain = hold_dir(content_axis, concept_axis)
    if retain is None:
        raise ValueError("identity retain is parallel to the concept; hold is off")
    retain_unit = retain / retain.norm().clamp_min(1e-8)
    identity = float(field.content) * content_axis
    concept_raw = float(field.slider) * concept_axis
    leak = float(field.leak) * leak_axis
    stranger = float(field.content) * field.basis(3)
    raw_plus = identity + concept_raw + leak
    raw_minus = identity - concept_raw - leak
    plus, minus = faithful_guard_e(raw_plus, raw_minus, identity, leak_axis, concept_axis)
    concept = plus - identity
    if float(concept.norm()) <= 1e-8:
        raise ValueError("guarded concept direction vanished")
    identity_amp = float(identity @ retain_unit)
    return SmileTeacher(
        identity=identity.detach(),
        concept=concept.detach(),
        stranger=stranger.detach(),
        retain_unit=retain_unit.detach(),
        identity_amp=identity_amp,
        plus=plus.detach(),
        minus=minus.detach(),
    )


class MidScaleResidual(nn.Module):
    """``s*odd + |s|*even + origin + bump(s)*mid``.

    ``bump`` is zero on ``{-1, 0, 1}``, so pole and neutral losses cannot
    see a mid-scale identity swap. Scale ``0.5`` can.
    """

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(dim))
        self.even = nn.Parameter(torch.zeros(dim))
        self.origin = nn.Parameter(torch.zeros(dim))
        self.mid = nn.Parameter(torch.zeros(dim))

    def state(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return s * self.odd + abs(s) * self.even + self.origin + mid_bump(s) * self.mid


# The residual starts at zero on purpose (the fit starts from "no edit"), so
# the explicit init keeps its constructor values.
init.register(MidScaleResidual, {"odd": init.KEEP, "even": init.KEEP, "origin": init.KEEP, "mid": init.KEEP})


class ScaleCritic(nn.Module):
    """Two-layer LeakyReLU MLP on ``(state / teacher RMS, scale)``."""

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

    def forward(self, state: torch.Tensor, scale: float) -> torch.Tensor:
        z = state.float() / self.input_scale
        label = z.new_full((z.shape[0], 1), float(scale))
        return self.net(torch.cat([z, label], dim=-1)).squeeze(-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).squeeze())


def _find(pairs: list[tuple[float, torch.Tensor]], target: float) -> torch.Tensor | None:
    for scale, state in pairs:
        if abs(scale - float(target)) <= 1e-6:
            return state
    return None


def _identity_kept(state: torch.Tensor, teacher: SmileTeacher) -> float:
    coef = float(state.float() @ teacher.retain_unit.float())
    amp = abs(float(teacher.identity_amp)) + 1e-8
    return 1.0 - min(1.0, abs(coef - float(teacher.identity_amp)) / amp)


def _concept_motion(state: torch.Tensor, origin: torch.Tensor, teacher: SmileTeacher) -> tuple[float, float]:
    motion = state.float() - origin.float()
    cos = _cos(motion, teacher.concept)
    mag = float(motion.norm()) / float(teacher.concept.norm().clamp_min(1e-8))
    return cos, mag


@torch.no_grad()
def score_hold(
    student: MidScaleResidual,
    *,
    scales=EVAL_SCALES,
    pairing: str = "matched",
    teacher: SmileTeacher | None = None,
) -> dict:
    """Score one residual on the scales the caller actually passed.

    The default grid is :data:`EVAL_SCALES` (includes ``-1``). A grid that
    omits ``-1`` is ``missing_minus`` and cannot PASS. This function does
    not probe a scale the caller left out.
    """
    if pairing not in ("matched", "stranger"):
        raise ValueError(f"pairing must be 'matched' or 'stranger', got {pairing!r}")
    teacher = teacher or smile_teacher()
    ordered: list[tuple[float, torch.Tensor]] = []
    for scale in scales:
        value = float(scale)
        if not math.isfinite(value):
            raise ValueError(f"scale must be finite, got {scale!r}")
        ordered.append((value, student.state(value).detach()))
    by_scale = ordered
    state0 = _find(by_scale, 0.0)
    state1 = _find(by_scale, 1.0)
    state_m = _find(by_scale, -1.0)
    state_mid = _find(by_scale, 0.5)

    concept_cos_plus = concept_mag_plus = None
    concept_cos_minus = concept_mag_minus = None
    if state0 is not None and state1 is not None:
        concept_cos_plus, concept_mag_plus = _concept_motion(state1, state0, teacher)
    if state0 is not None and state_m is not None:
        motion = state_m.float() - state0.float()
        concept_cos_minus = _cos(motion, -teacher.concept)
        concept_mag_minus = float(motion.norm()) / float(teacher.concept.norm().clamp_min(1e-8))
    identity_at_0 = None if state0 is None else _identity_kept(state0, teacher)
    identity_at_mid = None if state_mid is None else _identity_kept(state_mid, teacher)

    reasons: list[str] = []
    if not _has_scale(scales, -1.0):
        reasons.append("missing_minus")
    if not (_has_scale(scales, 0.0) and _has_scale(scales, 0.5) and _has_scale(scales, 1.0)):
        reasons.append("incomplete_grid")

    bad_sign = False
    bad_concept = False
    if concept_cos_plus is not None and concept_mag_plus is not None:
        if concept_cos_plus < 0.0:
            bad_sign = True
        elif concept_cos_plus < CONCEPT_COS_MIN or not (CONCEPT_MAG_LO <= concept_mag_plus <= CONCEPT_MAG_HI):
            bad_concept = True
    if concept_cos_minus is not None and concept_mag_minus is not None:
        if concept_cos_minus < 0.0:
            bad_sign = True
        elif concept_cos_minus < CONCEPT_COS_MIN or not (CONCEPT_MAG_LO <= concept_mag_minus <= CONCEPT_MAG_HI):
            bad_concept = True
    if bad_sign:
        reasons.append("polarity")
    elif bad_concept:
        reasons.append("concept")
    if identity_at_0 is not None and identity_at_0 < IDENTITY_KEPT_MIN:
        reasons.append("identity_0")
    if identity_at_mid is not None and identity_at_mid < IDENTITY_KEPT_MIN:
        reasons.append("identity_mid")
    reasons = [name for name in REASON_ORDER if name in reasons]

    bipolar = None
    if state0 is not None and state1 is not None and state_m is not None:
        bipolar = leftover_bipolar(state1 - state0, state_m - state0)

    per_scale = []
    for scale, state in by_scale:
        item = {
            "scale": float(scale),
            "identity_kept": _identity_kept(state, teacher),
            "on_concept": float(state.float() @ teacher.concept.float()) / float(teacher.concept.norm().clamp_min(1e-8)) ** 2,
        }
        per_scale.append(item)

    row = {
        "scales": [float(scale) for scale, _state in by_scale],
        "pairing": pairing,
        "concept_cos_plus": concept_cos_plus,
        "concept_cos_minus": concept_cos_minus,
        "concept_mag_plus": concept_mag_plus,
        "concept_mag_minus": concept_mag_minus,
        "identity_at_0": identity_at_0,
        "identity_at_mid": identity_at_mid,
        "pass": not reasons,
        "fail_reasons": ",".join(reasons),
        "per_scale": per_scale,
        "device": "cpu",
    }
    if bipolar is not None:
        row["same_dir"] = float(bipolar["same_dir"])
        row["leak_frac"] = float(bipolar["leak_frac"])
    else:
        row["same_dir"] = None
        row["leak_frac"] = None
    return row


def _fmt(value) -> str:
    if value is None:
        return "na"
    return f"{float(value):+.4f}"


def format_row(row: dict) -> str:
    """One line, meant to be tailed."""
    return (
        f"mid_scale arm={row['arm']} step={row.get('steps')} seed={row.get('seed')} "
        f"concept+={_fmt(row.get('concept_cos_plus'))} concept-={_fmt(row.get('concept_cos_minus'))} "
        f"id0={_fmt(row.get('identity_at_0'))} id_mid={_fmt(row.get('identity_at_mid'))} "
        f"scales={row.get('scales')} gate={'PASS' if row.get('pass') else 'FAIL'} "
        f"why={row.get('fail_reasons') or 'locked'}"
    )


def _train_arm_name(arm: str) -> str:
    """``missing_minus`` trains the locked residual and only drifts the eval grid."""
    if arm == "missing_minus":
        return "locked"
    return arm


def _eval_scales(arm: str) -> tuple[float, ...]:
    if arm == "missing_minus":
        return ANIMA_SMILE_SCALES
    return EVAL_SCALES


class MidScaleIdentity(ToyProblem):
    """One residual student (no prior) and one scale-conditioned critic.

    A batch is ``batch_size / 4`` rows per training scale, stacked in
    :data:`EVAL_SCALES` order. Each scale is its own critic view with weight
    ``1 / len(EVAL_SCALES)``, so the critic and generator losses are the
    per-scale average. The generator also carries the problem's cover
    constraint. ``arm`` picks the training target and eval grid (:data:`ARMS`).
    """

    name = "mid_scale_identity"

    def __init__(self, arm: str = "locked"):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        self.arm = arm
        self.teacher = smile_teacher()
        train_arm = _train_arm_name(arm)
        self.targets = {s: self.teacher.train_target(train_arm, s) for s in EVAL_SCALES}

    def recipe(self):
        # No prior, so z_dim / num_particles do not apply.
        return get_recipe(batch_size=N_ROWS * len(EVAL_SCALES), total_steps=GATE_STEPS)

    def networks(self, recipe, seed):
        student = init.deterministic_orthogonal_(MidScaleResidual(int(self.teacher.concept.numel())), seed=seed)
        cloud = torch.stack([self.targets[s] for s in EVAL_SCALES], dim=0)
        critic = init.deterministic_orthogonal_(ScaleCritic(student.odd.numel(), cloud), seed=seed + 1)
        return Networks(generator=student, critics=critic, prior=None)

    def _rows(self, n):
        if n % len(EVAL_SCALES):
            raise ValueError(f"batch must be a multiple of {len(EVAL_SCALES)} scales")
        return n // len(EVAL_SCALES)

    def real(self, n, stream):
        rows = self._rows(n)
        return torch.cat([self.targets[s].unsqueeze(0).expand(rows, -1) for s in EVAL_SCALES])

    def fake(self, nets, n, stream, real):
        rows = self._rows(n)
        return torch.cat([nets.generator.state(s).unsqueeze(0).expand(rows, -1) for s in EVAL_SCALES])

    def views(self, nets, real, fake):
        rows, weight = self._rows(len(real.x)), 1.0 / len(EVAL_SCALES)
        return [View("critic", real.x[i * rows:(i + 1) * rows], fake.x[i * rows:(i + 1) * rows], (s,), weight)
                for i, s in enumerate(EVAL_SCALES)]

    def losses(self, role, nets, real, fake):
        if role != "generator" or COVER_WEIGHT == 0:
            return {}
        cover = sum(F.mse_loss(nets.generator.state(s), self.targets[s]) for s in EVAL_SCALES)
        return {"cover": COVER_WEIGHT * cover / len(EVAL_SCALES)}

    def metrics(self, model):
        return score_hold(model.nets.generator, scales=_eval_scales(self.arm),
                          pairing="stranger" if self.arm == "stranger" else "matched", teacher=self.teacher)

    def verdict(self, metrics):
        return "PASS" if metrics["pass"] else "FAIL"


def run_arm(arm: str, *, steps: int = GATE_STEPS, seed: int = 0, noise_policy=None, log=None) -> dict:
    """Fit one arm on the shared runner and score its eval grid.

    Returns the live row (EMA under ``ema``) and prints a tailable line.
    Observations go to ``benchmarks.locked_shared.observation`` recorders.
    """
    if noise_policy is not None:
        raise ValueError("mid_scale_identity takes its noise from its recipe (benchmarks.toy_runner)")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    result = run(MidScaleIdentity(arm), steps=steps, seed=seed, log=log, observer=checkpoint)
    row = {**result["live"], "arm": arm, "steps": steps, "seed": seed,
           "train_scales": list(EVAL_SCALES), "cover_weight": COVER_WEIGHT,
           "ema": result["ema"], "hold": result["hold"]}
    print(format_row(row), flush=True)
    return row


if __name__ == "__main__":
    raise SystemExit(main(MidScaleIdentity()))

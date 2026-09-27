"""Cover-leftover toy: does a residual cover both caption poles without the leftover?

Problem only, from HyperGAN/conceptmod at 5571213 (see ../SOURCE.md and
../LICENSE): the one-row R^4 leftover field, the guarded teacher, the pole
clouds, the residual student plus one particle table per pole, the Fourier-2
critic, the cover constraint, the residual geometry metrics and the six
bounds. Everything else (optimizers and their LR schedule, loss, critic
penalty, particle regularizer, noise, EMA, observation logging) comes from
the shipped recipe through ``benchmarks.toy_runner``::

    python -m benchmarks.locked_shared.hosts.cover_leftover --log runs/toy-refactor/locked_cover_leftover.log
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from particlegan import get_recipe, init
from benchmarks.toy_runner import Networks, Sample, ToyProblem, main, run
from ..observation import checkpoint


LOCKED_COVER = 1.5
LOCKED_TEACHER = "faithful_guard_e"
LOCKED_N_PARTICLES = 12
U_KEPT_MIN = 0.85
CONTENT_KEPT_MIN = 0.75
LEAK_RATIO_MAX = 0.20
POLE_REL_ERR_MAX = 0.20
SAME_DIR_MAX = 0.25
GATE_STEPS = 800
BATCH = 32
CLOUD_STD = 0.03
SPAN_FRAC = 0.40
END_MARGIN = 0.60
CRITIC_HIDDEN = 64
CRITIC_N_RAND = 16
PARTICLE_INIT_STD = 0.05

# Problem arms: the cover constraint and the teacher are part of the task.
ARMS = {
    "locked": {},
    "cover_zero": {"cover_weight": 0.0},
    "teacher_drift": {"teacher": "faithful"},
}


class _Residual(nn.Module):
    """Odd/even residual ``delta(s) = s * w_odd + |s| * w_even``; starts at zero."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w_odd = nn.Parameter(torch.zeros(dim))
        self.w_even = nn.Parameter(torch.zeros(dim))

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w_odd + abs(float(scale)) * self.w_even


# The zero start is the problem: the residual begins at the neutral row.
init.register(_Residual, {"w_odd": init.KEEP, "w_even": init.KEEP})


class _FourierCritic(nn.Module):
    """Fourier-2 critic used as the training instrument.

    ParticleGAN does not ship a critic. The gate scores the residual, so
    the critic stays this one module for every arm (no architecture swap).
    """

    def __init__(self, dim: int, *, n_rand: int, hidden: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(int(seed) + 17)
        bank = 2.0 * torch.randn(int(n_rand), int(dim), generator=gen)
        self.register_buffer("bank", bank)
        feat = 4 * int(dim) + 2 * int(n_rand)
        self.net = nn.Sequential(
            nn.Linear(feat, int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden), int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        order1 = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        order2 = torch.cat([torch.sin(2.0 * x), torch.cos(2.0 * x)], dim=-1)
        proj = x @ self.bank.T
        rand = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(torch.cat([order1, order2, rand], dim=-1)).squeeze(-1)


def _unit(direction: torch.Tensor) -> torch.Tensor:
    flat = direction.flatten()
    return flat / flat.norm().clamp_min(1e-8)


def hold_dir(leak_dir: torch.Tensor, slider_dir: torch.Tensor) -> torch.Tensor | None:
    """ê perpendicular to û. Near-zero leftover turns the hold off."""
    axis = leak_dir.flatten()
    unit = _unit(slider_dir)
    out = axis - (axis @ unit) * unit
    if float(out.norm()) <= 1e-8:
        return None
    return out


def blend_guard(
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> bool:
    """True when the target is nearer its caption than the pair midpoint."""
    mid = 0.5 * (pos + neg)
    to_pole = max(float((tgt_plus - pos).norm()), float((tgt_minus - neg).norm()))
    to_mid = min(float((tgt_plus - mid).norm()), float((tgt_minus - mid).norm()))
    return to_pole < to_mid


def faithful_sub_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    slider_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Caption poles with leftover ê removed from the odd part only."""
    axis = (pos - neg) / 2.0
    held = hold_dir(leak_dir, slider_dir)
    if held is not None:
        unit = _unit(held)
        axis = axis - ((axis.flatten() @ unit) * unit).view_as(axis)
    common = (pos + neg) / 2.0 - neu
    return neu + common + axis, neu + common - axis


def faithful_guard_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    slider_dir: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract leftover ê when the blend guard admits it, else keep the captions."""
    plus, minus = faithful_sub_e(pos, neg, neu, leak_dir, slider_dir)
    if blend_guard(plus, minus, pos, neg):
        return plus, minus
    return pos, neg


@dataclass(frozen=True)
class LeftoverField:
    """One-row R^4 leftover: û, content, ê, lyric. Matches Field3D defaults."""

    slider: float = 1.0
    content: float = 0.55
    leak: float = 0.45
    lyric: float = 1.0

    @property
    def dim(self) -> int:
        return 4

    def basis(self, index: int) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[index] = 1.0
        return out

    def odd(self) -> torch.Tensor:
        return (
            float(self.slider) * self.basis(0)
            + float(self.content) * self.basis(1)
            + float(self.leak) * self.basis(2)
        )

    def poles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = float(self.lyric) * self.basis(3)
        amplitude = self.odd()
        return neu + amplitude, neu - amplitude, neu


def teacher_poles(
    field: LeftoverField,
    teacher: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos, neg, neu = field.poles()
    mode = str(teacher).strip().lower()
    if mode == "faithful":
        return pos, neg, neu
    if mode == "faithful_guard_e":
        plus, minus = faithful_guard_e(pos, neg, neu, field.basis(2), field.basis(0))
        return plus, minus, neu
    raise ValueError(f"unsupported teacher {teacher!r}")


def leftover_bipolar(d_plus: torch.Tensor, d_minus: torch.Tensor) -> dict[str, float]:
    """``leak_frac = cos(d+, d−)``, ``same_dir`` = even / (even + odd)."""
    even = 0.5 * (d_plus + d_minus)
    odd = 0.5 * (d_plus - d_minus)
    even_n = float(even.norm())
    odd_n = float(odd.norm())
    return {
        "leak_frac": float(
            F.cosine_similarity(d_plus.flatten().unsqueeze(0), d_minus.flatten().unsqueeze(0))
        ),
        "same_dir": even_n / (even_n + odd_n + 1e-8),
        "even_norm": even_n,
        "odd_norm": odd_n,
    }


def score_geometry(
    residual: _Residual,
    field: LeftoverField,
    poles_p: torch.Tensor,
    poles_m: torch.Tensor,
    neu: torch.Tensor,
) -> dict[str, float | bool]:
    d_plus = residual.delta(1.0).detach().cpu()
    d_minus = residual.delta(-1.0).detach().cpu()
    on_u = float(d_plus @ field.basis(0))
    on_c = float(d_plus @ field.basis(1))
    on_e = float(d_plus @ field.basis(2))
    u_kept = on_u / (float(field.slider) + 1e-8)
    content_kept = on_c / (float(field.content) + 1e-8)
    leak_ratio = abs(on_e) / (abs(on_u) + 1e-8)
    pred_p = neu + d_plus
    pred_m = neu + d_minus
    err_p = float((pred_p - poles_p).norm() / poles_p.norm().clamp_min(1e-8))
    err_m = float((pred_m - poles_m).norm() / poles_m.norm().clamp_min(1e-8))
    covered = bool(err_p <= POLE_REL_ERR_MAX and err_m <= POLE_REL_ERR_MAX)
    bipolar = leftover_bipolar(d_plus, d_minus)
    reasons = []
    if u_kept < U_KEPT_MIN or not covered:
        reasons.append("undershoot")
    if content_kept < CONTENT_KEPT_MIN:
        reasons.append("content")
    if leak_ratio > LEAK_RATIO_MAX:
        reasons.append("teacher_leak")
    if bipolar["same_dir"] > SAME_DIR_MAX:
        reasons.append("even_leftover")
    return {
        "u_kept": float(u_kept),
        "content_kept": float(content_kept),
        "leak_ratio": float(leak_ratio),
        "on_u": float(on_u),
        "on_content": float(on_c),
        "on_e": float(on_e),
        "pole_rel_err_plus": err_p,
        "pole_rel_err_minus": err_m,
        "covered": covered,
        "pass": not reasons,
        "fail_reasons": ",".join(reasons),
        **bipolar,
    }


def sample_pole_cloud(pole: torch.Tensor, neu: torch.Tensor, n: int, stream: torch.Generator) -> torch.Tensor:
    """Pole mass plus a short lyric-span lerp. One row, so every draw shares it."""
    n_end = int(round(END_MARGIN * int(n)))
    n_span = int(n) - n_end
    chunks = []
    if n_end:
        chunks.append(pole.expand(n_end, -1))
    if n_span:
        u = torch.rand(n_span, 1, generator=stream, device=pole.device).sqrt()
        lo = 1.0 - SPAN_FRAC
        chunks.append(neu + (lo + (1.0 - lo) * u) * (pole - neu))
    out = torch.cat(chunks, dim=0)
    return out + CLOUD_STD * torch.randn(out.shape, generator=stream, device=out.device)


class CoverLeftover(ToyProblem):
    """Residual student over the neutral row, one particle table per pole, and
    the problem's cover constraint pulling both residual poles onto the teacher.
    ``arm`` picks a problem arm from ``ARMS`` (cover off, or an unguarded teacher)."""

    name = "cover_leftover"

    def __init__(self, arm: str = "locked", *, field: LeftoverField | None = None):
        options = {"cover_weight": LOCKED_COVER, "teacher": LOCKED_TEACHER, **ARMS[arm]}
        self.arm, self.field = arm, field or LeftoverField()
        self.cover_weight, self.teacher = float(options["cover_weight"]), options["teacher"]
        self.poles_p, self.poles_m, self.neu = teacher_poles(self.field, self.teacher)

    def recipe(self):
        return get_recipe(z_dim=self.field.dim, num_particles=LOCKED_N_PARTICLES,
                          batch_size=BATCH, total_steps=GATE_STEPS)

    def networks(self, recipe, seed):
        residual = init.deterministic_orthogonal_(_Residual(self.field.dim), seed=seed)
        critic = init.deterministic_orthogonal_(
            _FourierCritic(self.field.dim, n_rand=CRITIC_N_RAND, hidden=CRITIC_HIDDEN, seed=seed), seed=seed + 1)
        priors = tuple(init.deterministic_orthogonal_(recipe.make_prior(init_std=PARTICLE_INIT_STD), seed=seed)
                       for _ in ("plus", "minus"))
        return Networks(generator=residual, critics=critic, prior=priors)

    def _targets(self, device):
        return self.poles_p.to(device), self.poles_m.to(device), self.neu.to(device)

    def real(self, n, stream):
        poles_p, poles_m, neu = self._targets(stream.device)
        half = max(1, n // 2)
        return torch.cat([sample_pole_cloud(poles_p, neu, half, stream),
                          sample_pole_cloud(poles_m, neu, half, stream)], dim=0)

    def fake(self, nets, n, stream, real):
        prior_p, prior_m = nets.priors
        residual = nets.generator
        neu = self.neu.to(prior_p.z.device)
        half = max(1, n // 2)
        z_p, _ = prior_p.sample(half, generator=stream)
        z_m, _ = prior_m.sample(half, generator=stream)
        return torch.cat([neu + residual.delta(1.0) + z_p, neu + residual.delta(-1.0) + z_m], dim=0)

    def losses(self, role, nets, real, fake):
        if role != "generator" or self.cover_weight == 0:
            return {}
        residual = nets.generator
        poles_p, poles_m, neu = self._targets(residual.w_odd.device)
        cover = (neu + residual.delta(1.0) - poles_p).pow(2).mean()
        cover = cover + (neu + residual.delta(-1.0) - poles_m).pow(2).mean()
        return {"cover": self.cover_weight * cover}

    def metrics(self, model):
        row = score_geometry(model.nets.generator, self.field, self.poles_p, self.poles_m, self.neu)
        row["particle_rms"] = float(torch.cat([p.z for p in model.nets.priors]).pow(2).mean().sqrt())
        return row

    def verdict(self, metrics):
        return "PASS" if metrics["pass"] else "FAIL"


def train_cover_leftover(problem: CoverLeftover | None = None, *, seed: int = 0, steps: int | None = None,
                         log=None, log_path=None, recipe=None) -> dict:
    """Train one arm on the shared runner. The EMA residual row (the host's
    reported score) with its verdict, plus ``live``, ``live_curve`` and ``hold``.

    Observations go to ``benchmarks.locked_shared.observation`` recorders.
    """
    problem = CoverLeftover() if problem is None else problem
    result = run(problem, recipe=recipe, seed=seed, steps=steps, log=log, log_path=log_path,
                 observer=checkpoint)
    return {**result["ema"], "arm": problem.arm, "teacher": problem.teacher,
            "cover_weight": problem.cover_weight, "steps": result["steps"], "seed": seed,
            "live": result["live"], "live_curve": result["curve"], "hold": result["hold"]}


if __name__ == "__main__":
    raise SystemExit(main(CoverLeftover()))

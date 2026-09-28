#!/usr/bin/env python3
"""Read-only, non-gating diagnostics for the native 100-Gaussian tasks.

``cloud_diagnostics`` is also called by screen.py at every native observation.
The CLI can inspect the saved noisy quality clouds and holdout of an older run:

    python harness/native100_diagnostics.py runs/CAND/grid100

The frozen coverage and accuracy scorers do not import this module. Mode IDs
and the 3-sigma radius come from the frozen evaluator geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from hosts.native100.problems import PROBLEM_NAMES, evaluation_geometry


HQ_RADIUS_SIGMAS = 3.0
MIN_HQ_MODE_MASS = .005  # frozen coverage gate
TARGET_CONDITIONAL_VARIANCE = (
    1.0 - 0.5 * HQ_RADIUS_SIGMAS**2 * math.exp(-HQ_RADIUS_SIGMAS**2 / 2.0)
    / (1.0 - math.exp(-HQ_RADIUS_SIGMAS**2 / 2.0))
)


def geometry(problem: str) -> tuple[np.ndarray, float]:
    if problem not in PROBLEM_NAMES:
        raise ValueError(f"unknown native100 problem: {problem}")
    centers, sigma = evaluation_geometry(problem, dtype=torch.float64)
    return centers.numpy(), float(sigma)


def _nearest_two(points: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest and second-nearest centre IDs, with bounded CPU memory."""
    first = np.empty(len(points), dtype=np.int32)
    second = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), 10000):
        stop = min(start + 10000, len(points))
        delta = points[start:stop, None, :] - centers[None, :, :]
        dist2 = np.einsum("nkd,nkd->nk", delta, delta)
        ids = np.argpartition(dist2, kth=1, axis=1)[:, :2]
        flip = dist2[np.arange(stop - start), ids[:, 1]] < dist2[np.arange(stop - start), ids[:, 0]]
        ids[flip] = ids[flip, ::-1]
        first[start:stop], second[start:stop] = ids[:, 0], ids[:, 1]
    return first, second


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {name: float(np.quantile(values, q)) for name, q in
            (("p50", .5), ("p90", .9), ("p99", .99))}


def cloud_diagnostics(samples: np.ndarray, problem: str) -> dict:
    """Describe mode occupancy, bridge corridors and per-mode conditional shape.

    These values are explanatory only. A corridor sample lies outside its
    nearest 3-sigma quality ball and within a 3-sigma tube around the interior
    segment connecting its two nearest target centres. The interior is bounded
    by those same 3-sigma balls. This is a lower bound on bridge stragglers;
    curved bridges and isolated particles remain in ``other_outlier_fraction``.
    """
    centers, sigma = geometry(problem)
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x):
        raise ValueError("native100 samples must have shape (N, 2), N > 0")
    valid = np.isfinite(x).all(axis=1)
    n = len(x)
    if not valid.all():
        return dict(n=n, valid_n=int(valid.sum()), invalid_n=int((~valid).sum()))

    nearest, second = _nearest_two(x, centers)
    residual = (x - centers[nearest]) / sigma
    radius = np.linalg.norm(residual, axis=1)
    hq = radius <= HQ_RADIUS_SIGMAS
    outlier = ~hq
    all_counts = np.bincount(nearest, minlength=len(centers))
    hq_counts = np.bincount(nearest[hq], minlength=len(centers))

    # The corridor uses the two nearest centres; no fitted bridge width or
    # task-specific threshold is introduced.
    a, b = centers[nearest], centers[second]
    ab = b - a
    length2 = np.einsum("nd,nd->n", ab, ab)
    fraction = np.einsum("nd,nd->n", x - a, ab) / length2
    gap = np.sqrt(length2)
    perpendicular = np.linalg.norm(x - (a + fraction[:, None] * ab), axis=1)
    corridor = (outlier & (fraction >= HQ_RADIUS_SIGMAS * sigma / gap)
                & (fraction <= 1.0 - HQ_RADIUS_SIGMAS * sigma / gap)
                & (perpendicular <= HQ_RADIUS_SIGMAS * sigma))
    corridor_counts = np.bincount(nearest[corridor], minlength=len(centers))
    outlier_counts = np.bincount(nearest[outlier], minlength=len(centers))

    center_offset = np.full(len(centers), np.nan)
    eig_low = np.full(len(centers), np.nan)
    eig_high = np.full(len(centers), np.nan)
    trace_ratio = np.full(len(centers), np.nan)
    for mode in range(len(centers)):
        r = residual[hq & (nearest == mode)]
        if len(r) < 2:
            continue
        mean = r.mean(axis=0)
        cov = (r - mean).T @ (r - mean) / len(r)
        eig = np.linalg.eigvalsh(cov) / TARGET_CONDITIONAL_VARIANCE
        center_offset[mode] = np.linalg.norm(mean)
        eig_low[mode], eig_high[mode] = eig
        trace_ratio[mode] = np.trace(cov) / (2 * TARGET_CONDITIONAL_VARIANCE)

    shape_valid = np.isfinite(center_offset)
    measured_modes = np.flatnonzero(shape_valid)
    def finite_quantiles(values):
        return _quantiles(values[np.isfinite(values)])
    def floats_or_none(values):
        return [float(v) if math.isfinite(v) else None for v in values]

    return {
        "n": n,
        "valid_n": n,
        "invalid_n": 0,
        "precision": float(hq.mean()),
        "outlier_fraction": float(outlier.mean()),
        "bridge_corridor_fraction": float(corridor.mean()),
        "other_outlier_fraction": float((outlier & ~corridor).mean()),
        "nearest_radius_sigma": _quantiles(radius),
        "outlier_radius_sigma": _quantiles(radius[outlier]),
        "covered_modes": int((hq_counts >= math.ceil(MIN_HQ_MODE_MASS * n - 1e-12)).sum()),
        "nonempty_modes": int((hq_counts > 0).sum()),
        "missing_modes": np.flatnonzero(hq_counts == 0).tolist(),
        "low_hq_mass_modes": np.flatnonzero(hq_counts / n < MIN_HQ_MODE_MASS).tolist(),
        "center_offset_sigma": finite_quantiles(center_offset),
        "eig_low_ratio": finite_quantiles(eig_low),
        "eig_high_ratio": finite_quantiles(eig_high),
        "trace_ratio": finite_quantiles(trace_ratio),
        "worst_center_modes": measured_modes[np.argsort(-center_offset[measured_modes])[:5]].tolist(),
        "worst_width_modes": measured_modes[np.argsort(-np.abs(trace_ratio[measured_modes] - 1))[:5]].tolist(),
        "modes": {
            "count": all_counts.tolist(),
            "hq_count": hq_counts.tolist(),
            "outlier_count": outlier_counts.tolist(),
            "bridge_corridor_count": corridor_counts.tolist(),
            "center_offset_sigma": floats_or_none(center_offset),
            "eig_low_ratio": floats_or_none(eig_low),
            "eig_high_ratio": floats_or_none(eig_high),
            "trace_ratio": floats_or_none(trace_ratio),
        },
    }


def prior_motion_diagnostics(previous: np.ndarray | None, current: np.ndarray,
                             problem: str) -> dict:
    """Track table rows by index between observations; BD moves count as jumps."""
    centers, sigma = geometry(problem)
    now = np.asarray(current, dtype=np.float64)
    if now.ndim != 2 or now.shape[1] != 2 or not np.isfinite(now).all():
        raise ValueError("prior centres must be finite (N, 2) coordinates")
    if previous is None:
        return dict(rows=len(now), interval=None)
    old = np.asarray(previous, dtype=np.float64)
    if old.shape != now.shape:
        raise ValueError("prior row count changed between observations")
    old_mode = _nearest_two(old, centers)[0]
    new_mode = _nearest_two(now, centers)[0]
    motion = np.linalg.norm(now - old, axis=1) / sigma
    energy = motion**2
    top_n = max(1, math.ceil(.01 * len(now)))
    energy_total = float(energy.sum())
    count = np.bincount(old_mode, minlength=len(centers))
    rms = np.sqrt(np.bincount(old_mode, weights=energy, minlength=len(centers))
                  / np.maximum(count, 1))
    return {
        "rows": len(now),
        "interval": "previous observation to current observation",
        "rms_sigma": float(np.sqrt(energy.mean())),
        "motion_sigma": _quantiles(motion),
        "top1pct_motion_energy_share": (
            float(np.partition(energy, -top_n)[-top_n:].sum() / energy_total)
            if energy_total else None),
        "mode_switch_rows": int((old_mode != new_mode).sum()),
        "mode_switch_fraction": float((old_mode != new_mode).mean()),
        "by_previous_mode": {"rows": count.tolist(), "rms_sigma": rms.tolist()},
    }


def affine_motion_diagnostics(previous: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                              current: tuple[np.ndarray, np.ndarray, np.ndarray],
                              previous_displacement: np.ndarray | None, problem: str,
                              *, lag_lineage_valid: bool = True) -> tuple[dict, np.ndarray | None]:
    """Decompose tracked affine-table motion without sampling or changing training.

    A state is ``(z, W, b)`` with clean row centres ``q = z @ W.T + b``. The
    midpoint/Shapley split is exact and treats prior and generator symmetrically:

        delta_prior = (z1 - z0) @ ((W0 + W1) / 2).T
        delta_G = ((z0 + z1) / 2) @ (W1 - W0).T + b1 - b0

    The optional lag compares row-matched output displacements of adjacent
    observation intervals. It is omitted if birth/death changed row lineage
    during either interval. All output is explanatory and evaluator-only.
    """
    centers, sigma = geometry(problem)
    z1, w1, b1 = (np.asarray(x, dtype=np.float64) for x in current)
    if (z1.ndim != 2 or w1.ndim != 2 or b1.shape != (w1.shape[0],)
            or z1.shape[1] != w1.shape[1] or w1.shape[0] != centers.shape[1]
            or not all(np.isfinite(x).all() for x in (z1, w1, b1))):
        raise ValueError("invalid current affine table state")
    if previous is None:
        return dict(rows=len(z1), interval=None, version=1), None
    z0, w0, b0 = (np.asarray(x, dtype=np.float64) for x in previous)
    if (z0.shape != z1.shape or w0.shape != w1.shape or b0.shape != b1.shape
            or not all(np.isfinite(x).all() for x in (z0, w0, b0))):
        raise ValueError("previous affine table state does not match current state")

    prior = (z1 - z0) @ ((w0 + w1) / 2).T
    generator = ((z0 + z1) / 2) @ (w1 - w0).T + b1 - b0
    displacement = prior + generator
    old_centres = z0 @ w0.T + b0
    old_mode = _nearest_two(old_centres, centers)[0]
    counts = np.bincount(old_mode, minlength=len(centers))
    mode_mean = np.stack([
        np.bincount(old_mode, weights=displacement[:, axis], minlength=len(centers))
        / np.maximum(counts, 1)
        for axis in range(displacement.shape[1])
    ], axis=1)
    within = displacement - mode_mean[old_mode]

    def rms_sigma(x):
        return float(np.sqrt(np.mean(np.sum(x * x, axis=1))) / sigma)

    lag_cosine = lag_mean_row_cosine = lag_valid_row_fraction = None
    if previous_displacement is not None and lag_lineage_valid:
        old_delta = np.asarray(previous_displacement, dtype=np.float64)
        if old_delta.shape != displacement.shape or not np.isfinite(old_delta).all():
            raise ValueError("previous displacement does not match current rows")
        old_norm = np.linalg.norm(old_delta, axis=1)
        new_norm = np.linalg.norm(displacement, axis=1)
        valid = (old_norm > np.finfo(np.float64).tiny) & (new_norm > np.finfo(np.float64).tiny)
        lag_valid_row_fraction = float(valid.mean())
        if valid.any():
            dot = np.einsum("nd,nd->n", old_delta[valid], displacement[valid])
            lag_mean_row_cosine = float(np.mean(dot / (old_norm[valid] * new_norm[valid])))
            denominator = np.linalg.norm(old_delta[valid]) * np.linalg.norm(displacement[valid])
            lag_cosine = float(dot.sum() / denominator) if denominator > 0 else None

    mode_mean_norm = np.linalg.norm(mode_mean, axis=1) / sigma
    out = dict(
        rows=len(z1), interval="previous observation to current observation", version=1,
        rms_prior_sigma=rms_sigma(prior), rms_generator_sigma=rms_sigma(generator),
        rms_total_sigma=rms_sigma(displacement),
        mean_cross_dot_sigma2=float(np.mean(np.sum(prior * generator, axis=1)) / sigma**2),
        rms_mode_mean_sigma=rms_sigma(mode_mean[old_mode]),
        rms_within_mode_sigma=rms_sigma(within),
        mode_mean_motion_sigma=_quantiles(mode_mean_norm[counts > 0]),
        mode_mean_motion_sigma_by_mode=[float(value) if counts[i] else None
                                        for i, value in enumerate(mode_mean_norm)],
        lag1_lineage_valid=bool(lag_lineage_valid),
        lag1_row_motion_cosine=lag_cosine,
        lag1_mean_row_cosine=lag_mean_row_cosine,
        lag1_valid_row_fraction=lag_valid_row_fraction,
    )
    return out, displacement


def inspect_saved_run(run_dir: Path, output: Path | None = None) -> Path:
    """Backfill cloud metrics from noisy final-five clouds and holdout."""
    task_dir = run_dir if run_dir.name in PROBLEM_NAMES else run_dir.parent
    problem = task_dir.name
    if problem not in PROBLEM_NAMES:
        raise ValueError(f"cannot infer native100 problem from {run_dir}")
    noisy = run_dir / "native-noisy" if (run_dir / "native-noisy").is_dir() else run_dir
    clouds = sorted((noisy / "quality_checks").glob("step_*.npz"))
    holdout = noisy / "holdout_samples.npz"
    if holdout.exists():
        clouds.append(holdout)
    if not clouds:
        raise FileNotFoundError(f"no saved noisy quality clouds in {noisy}")
    output = output or task_dir / "native100-cloud-backfill.jsonl"
    with output.open("w") as dst:
        for path in clouds:
            step = int(path.stem.removeprefix("step_")) if path.stem.startswith("step_") else "holdout"
            with np.load(path) as arrays:
                for model in ("live", "ema"):
                    row = dict(problem=problem, step=step, evaluation="noisy", model=model,
                               cloud=cloud_diagnostics(arrays[model], problem))
                    dst.write(json.dumps(row, allow_nan=False) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="runs/CAND/grid100 or its native-noisy directory")
    parser.add_argument("--output", type=Path, help="sidecar JSONL path")
    args = parser.parse_args()
    print(inspect_saved_run(args.run_dir, args.output))


if __name__ == "__main__":
    main()

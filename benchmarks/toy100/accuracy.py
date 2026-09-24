"""Distribution-fidelity audit for saved 100-Gaussian sample clouds.

The original coverage gate in :mod:`metrics` remains the mandatory first
condition. These additional, frozen diagnostics compare 20,000 independent
draws with the *distribution* of the target mixture. Target centers are used
only here, after generation. They are never training labels.

The four accuracy limits were set before screening accuracy candidates. Their
noise floor can be checked with :func:`oracle_reference`, which draws fresh
samples from the public target sampler at the same evaluation size. The score
is the mean of the four errors divided by their fixed limits; a smaller score
is better, but a candidate is eligible only if both gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch

from .metrics import EVAL_N, HQ_RADIUS_SIGMAS, evaluate_samples, passes
from .problems import N_MODES, PROBLEM_NAMES, evaluation_geometry, sample_real


PROTOCOL = "toy100-accuracy-v1"
ORACLE_SEED = 20260923
TARGET_PRECISION = 1.0 - math.exp(-HQ_RADIUS_SIGMAS**2 / 2.0)
# E[X_j^2 | ||X|| <= 3] for X ~ N(0, I_2).
TARGET_CONDITIONAL_VARIANCE = (
    1.0 - 0.5 * HQ_RADIUS_SIGMAS**2 * math.exp(-HQ_RADIUS_SIGMAS**2 / 2.0)
    / TARGET_PRECISION
)
LIMITS = {
    "mass_tv": 0.06,
    "center_rms_sigma": 0.20,
    "abs_cov_trace_bias": 0.10,
    "radial_ks": 0.04,
}
SCORE_FIELDS = tuple(LIMITS)


def _points(samples: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    points = np.asarray(samples)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("samples must have shape (N, 2), with N > 0")
    if not np.issubdtype(points.dtype, np.number) or np.iscomplexobj(points):
        raise ValueError("samples must contain real-valued coordinates")
    return points.astype(np.float64, copy=False)


def fidelity_metrics(
    samples: np.ndarray | torch.Tensor, problem_name: str,
) -> dict[str, float | int | bool | None]:
    """Measure mixture fidelity without drawing randomness or changing RNG state.

    Nearest-center assignments estimate mode mass. Center means, covariance,
    and radii use the points within the original 3-sigma quality radius so
    remote outliers cannot dominate conditional moments. Precision is checked
    against the target's analytic 3-sigma probability in *both* directions;
    high precision by itself earns no accuracy credit.
    """
    if problem_name not in PROBLEM_NAMES:
        raise ValueError(f"unknown toy100 problem {problem_name!r}")
    points = _points(samples)
    n = len(points)
    valid = np.isfinite(points).all(axis=1)
    result: dict[str, float | int | bool | None] = {
        "n": n,
        "valid_n": int(valid.sum()),
        "all_finite": bool(valid.all()),
        "within_radius_n": 0,
        "precision": None,
        "precision_gap": None,
        "mass_tv": None,
        "center_rms_sigma": None,
        "center_max_sigma": None,
        "cov_trace_bias": None,
        "abs_cov_trace_bias": None,
        "cov_frob_rms": None,
        "radial_ks": None,
    }
    if not valid.all():
        return result

    centers, sigma = evaluation_geometry(problem_name, dtype=torch.float64)
    center_array = centers.numpy()
    # N=20,000, K=100; vectorization is fast and has modest memory use.
    delta = points[:, None, :] - center_array[None, :, :]
    dist2 = np.einsum("nkd,nkd->nk", delta, delta)
    nearest = np.argmin(dist2, axis=1)
    residual = delta[np.arange(n), nearest] / sigma
    radii = np.linalg.norm(residual, axis=1)
    in_radius = radii <= HQ_RADIUS_SIGMAS
    counts = np.bincount(nearest, minlength=N_MODES)
    mass_tv = float(np.abs(counts / n - 1.0 / N_MODES).sum() / 2.0)
    precision = float(in_radius.mean())
    result.update(
        within_radius_n=int(in_radius.sum()),
        precision=precision,
        precision_gap=abs(precision - TARGET_PRECISION),
        mass_tv=mass_tv,
    )

    ids = nearest[in_radius]
    hq_counts = np.bincount(ids, minlength=N_MODES)
    if np.any(hq_counts < 2):
        # Missing components cannot have a meaningful conditional covariance.
        return result
    x = residual[in_radius]
    means = np.stack(
        [np.bincount(ids, weights=x[:, j], minlength=N_MODES) / hq_counts
         for j in range(2)],
        axis=1,
    )
    second = np.empty((N_MODES, 2, 2), dtype=np.float64)
    for j in range(2):
        for k in range(2):
            second[:, j, k] = (
                np.bincount(ids, weights=x[:, j] * x[:, k], minlength=N_MODES)
                / hq_counts
            )
    covariances = second - means[:, :, None] * means[:, None, :]
    q = TARGET_CONDITIONAL_VARIANCE
    trace_ratio = np.trace(covariances, axis1=1, axis2=2) / (2.0 * q)
    trace_bias = float(trace_ratio.mean() - 1.0)
    sorted_radii = np.sort(radii[in_radius])
    target_cdf = (
        1.0 - np.exp(-sorted_radii**2 / 2.0)
    ) / TARGET_PRECISION
    positions = np.arange(len(sorted_radii))
    radial_ks = float(max(
        np.max((positions + 1) / len(sorted_radii) - target_cdf),
        np.max(target_cdf - positions / len(sorted_radii)),
    ))
    result.update(
        center_rms_sigma=float(np.sqrt(np.mean(np.sum(means**2, axis=1)))),
        center_max_sigma=float(np.max(np.linalg.norm(means, axis=1))),
        cov_trace_bias=trace_bias,
        abs_cov_trace_bias=abs(trace_bias),
        cov_frob_rms=float(np.sqrt(np.mean(
            np.sum((covariances - q * np.eye(2))**2, axis=(1, 2)) / (2.0 * q**2)
        ))),
        radial_ks=radial_ks,
    )
    return result


def passes_accuracy(observation: Mapping[str, Any]) -> bool:
    """Apply the fixed fidelity limits without trusting a stored verdict."""
    try:
        n = observation["n"]
        valid_n = observation["valid_n"]
        if isinstance(n, bool) or isinstance(valid_n, bool):
            return False
        if int(n) != n or int(valid_n) != valid_n:
            return False
        if n < EVAL_N or valid_n != n or observation["all_finite"] is not True:
            return False
        for key, limit in LIMITS.items():
            value = observation[key]
            if isinstance(value, bool) or not math.isfinite(float(value)):
                return False
            if not 0.0 <= float(value) <= limit:
                return False
        bias = float(observation["cov_trace_bias"])
        if not math.isfinite(bias):
            return False
        return math.isclose(float(observation["abs_cov_trace_bias"]), abs(bias), abs_tol=1e-12)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def evaluate_accuracy(
    samples: np.ndarray | torch.Tensor,
    problem_name: str,
    *,
    gate_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return serializable fidelity metrics, fixed score, and both gate verdicts."""
    points = _points(samples)
    shape = fidelity_metrics(points, problem_name)
    if gate_metrics is None:
        gate_metrics = evaluate_samples(torch.from_numpy(points), problem_name)
    frozen_pass = passes(problem_name, gate_metrics)
    accuracy_pass = passes_accuracy(shape)
    score = (
        sum(float(shape[key]) / limit for key, limit in LIMITS.items()) / len(LIMITS)
        if all(shape[key] is not None for key in LIMITS) else None
    )
    return {
        "protocol": PROTOCOL,
        "problem": problem_name,
        **shape,
        "accuracy_score": score,
        "accuracy_pass": accuracy_pass,
        "frozen_pass": frozen_pass,
        "passed": frozen_pass and accuracy_pass,
    }


def oracle_reference(
    problem_name: str, *, n: int = EVAL_N, repetitions: int = 16,
    seed: int = ORACLE_SEED,
) -> dict[str, Any]:
    """Calibrate finite-sample errors using independent target-sampler draws."""
    if repetitions < 1 or n < 1:
        raise ValueError("n and repetitions must be positive")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    rows = [
        fidelity_metrics(sample_real(problem_name, n, generator=rng), problem_name)
        for _ in range(repetitions)
    ]
    fields = (*SCORE_FIELDS, "precision_gap", "cov_frob_rms")
    summary = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "p95": float(np.quantile([row[key] for row in rows], 0.95)),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in fields
    }
    return {
        "sampler": "benchmarks.toy100.problems.sample_real",
        "seed": seed,
        "samples_per_repetition": n,
        "repetitions": repetitions,
        "metrics": summary,
    }


def audit_npz(path: str | Path, problem_name: str, *, key: str = "live") -> dict[str, Any]:
    """Score one saved final cloud, including its original frozen gate."""
    with np.load(path) as archive:
        if key not in archive:
            raise KeyError(f"{path} has no {key!r} sample array")
        samples = archive[key]
    return evaluate_accuracy(samples, problem_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="final_samples.npz or a suite directory")
    parser.add_argument("--problem", choices=PROBLEM_NAMES,
                        help="required for an individual NPZ if its parent name is different")
    parser.add_argument("--key", default="live", help="NPZ array to audit (default: live)")
    parser.add_argument("--output", type=Path, help="write the same JSON report here")
    parser.add_argument("--oracle-repetitions", type=int, default=0,
                        help="include fixed-seed target-sampler calibration")
    parser.add_argument("--require-pass", action="store_true",
                        help="exit 1 if either fixed gate fails")
    args = parser.parse_args(argv)
    if args.path.is_file():
        problem = args.problem or args.path.parent.name
        if problem not in PROBLEM_NAMES:
            parser.error("--problem is required when the NPZ parent is not a problem name")
        problems = {problem: audit_npz(args.path, problem, key=args.key)}
    elif args.path.is_dir():
        problems = {
            name: audit_npz(args.path / name / "final_samples.npz", name, key=args.key)
            for name in PROBLEM_NAMES
        }
    else:
        parser.error(f"path does not exist: {args.path}")
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "sample_key": args.key,
        "limits": LIMITS,
        "target_precision": TARGET_PRECISION,
        "target_conditional_variance": TARGET_CONDITIONAL_VARIANCE,
        "problems": problems,
        "all_passed": all(row["passed"] for row in problems.values()),
    }
    if args.oracle_repetitions:
        report["oracle"] = {
            name: oracle_reference(name, repetitions=args.oracle_repetitions)
            for name in problems
        }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    return int(args.require_pass and not report["all_passed"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

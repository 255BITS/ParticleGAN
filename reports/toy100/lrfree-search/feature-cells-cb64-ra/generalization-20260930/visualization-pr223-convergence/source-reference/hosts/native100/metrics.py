"""Frozen sample-quality gate for the 2D, 100-Gaussian suite.

All target geometry is used here, after generation. The trainer sees only
unlabelled draws from ``problems.sample_real``. The 20,000-draw minimum makes
the per-mode lower mass bound meaningful: each mode needs at least 100 genuine
3-sigma hits, rather than a single lucky hit or a nearest-cell assignment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .problems import N_MODES, PROBLEM_NAMES, evaluation_geometry


EVAL_N = 20_000
HQ_RADIUS_SIGMAS = 3.0
MIN_HQ_MODE_MASS = 0.005  # half the expected uniform mode mass
MIN_PRECISION = 0.97  # oracle value: 1 - exp(-9/2) = 0.98889
MAX_MASS_TV = 0.10
MAX_MODE_MASS = 0.02  # twice the expected mode mass
MIN_COV_EIG_RATIO = 0.40
MAX_COV_EIG_RATIO = 1.70
MIN_RADIAL_MEDIAN_RATIO = 0.65
MAX_RADIAL_MEDIAN_RATIO = 1.40

REQUIRED_KEYS = (
    "n", "valid_n", "modes", "hq", "precision", "min_hq_count",
    "min_hq_mode_mass", "max_mode_mass", "mass_tv", "min_cov_eig_ratio",
    "max_cov_eig_ratio", "min_radial_median_ratio", "max_radial_median_ratio",
)
REQUIREMENTS = {
    "min_samples": EVAL_N,
    "min_modes": N_MODES,
    "min_hq_mode_mass": MIN_HQ_MODE_MASS,
    "min_precision": MIN_PRECISION,
    "max_mass_tv": MAX_MASS_TV,
    "max_mode_mass": MAX_MODE_MASS,
    "min_cov_eig_ratio": MIN_COV_EIG_RATIO,
    "max_cov_eig_ratio": MAX_COV_EIG_RATIO,
    "min_radial_median_ratio": MIN_RADIAL_MEDIAN_RATIO,
    "max_radial_median_ratio": MAX_RADIAL_MEDIAN_RATIO,
    "all_finite": True,
}

# Median radius for a 2D standard Gaussian conditional on radius <= 3.
_HQ_RADIAL_MEDIAN = math.sqrt(-2.0 * math.log((1.0 + math.exp(-4.5)) / 2.0))


def passes(problem_name: str, observation: Mapping) -> bool:
    """Apply the same fixed threshold set to every named problem.

    Accepts either a flat metric dictionary or an event row containing one
    under ``metrics``. Never trusts a stored ``passed`` field.
    """
    if problem_name not in PROBLEM_NAMES:
        raise ValueError(f"unknown toy100 problem {problem_name!r}")
    metrics = observation.get("metrics", observation)
    if not isinstance(metrics, Mapping):
        return False

    def number(key: str) -> float | None:
        value = metrics.get(key)
        if isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    values = {key: number(key) for key in REQUIRED_KEYS}
    if any(value is None for value in values.values()):
        return False
    if not all(values[key].is_integer() for key in ("n", "valid_n", "modes", "min_hq_count")):
        return False
    if not (
        values["n"] > 0
        and 0 <= values["modes"] <= N_MODES
        and 0 <= values["min_hq_count"] <= values["valid_n"] <= values["n"]
        and 0 <= values["hq"] <= 1
        and 0 <= values["precision"] <= 1
        and 0 <= values["min_hq_mode_mass"] <= values["max_mode_mass"] <= 1
        and 0 <= values["mass_tv"] <= 1
        and 0 <= values["min_cov_eig_ratio"] <= values["max_cov_eig_ratio"]
        and 0 <= values["min_radial_median_ratio"] <= values["max_radial_median_ratio"]
        and abs(values["min_hq_mode_mass"] - values["min_hq_count"] / values["n"]) <= 1e-12
    ):
        return False
    return bool(
        values["n"] >= EVAL_N
        and values["valid_n"] == values["n"]
        and values["modes"] == N_MODES
        and abs(values["hq"] - values["precision"]) <= 1e-7
        and values["precision"] >= MIN_PRECISION
        and values["min_hq_mode_mass"] >= MIN_HQ_MODE_MASS - 1e-12
        and values["mass_tv"] <= MAX_MASS_TV
        and values["max_mode_mass"] <= MAX_MODE_MASS
        and values["min_cov_eig_ratio"] >= MIN_COV_EIG_RATIO
        and values["max_cov_eig_ratio"] <= MAX_COV_EIG_RATIO
        and values["min_radial_median_ratio"] >= MIN_RADIAL_MEDIAN_RATIO
        and values["max_radial_median_ratio"] <= MAX_RADIAL_MEDIAN_RATIO
    )


@torch.no_grad()
def evaluate_samples(fake: torch.Tensor, problem_name: str) -> dict[str, int | float | bool]:
    """Audit every mode of independent generated draws; return scalar metrics.

    Balance uses every finite point's nearest-center cell; precision, coverage,
    and spread use only samples inside that mode's 3-sigma radius. Nonfinite
    outputs count in ``n`` and fail precision. Spread is measured on the
    in-radius population about its own mean, using covariance eigenvalues and
    the per-mode median radial distance. Both extremes across all 100 modes
    enter the pass rule, so a healthy average cannot hide one collapsed mode.
    """
    if problem_name not in PROBLEM_NAMES:
        raise ValueError(f"unknown toy100 problem {problem_name!r}")
    samples = torch.as_tensor(fake).detach()
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] == 0:
        raise ValueError("fake must have shape (N, 2), with N > 0")
    if samples.is_complex():
        raise ValueError("fake must contain real-valued coordinates")
    n = samples.shape[0]
    valid = torch.isfinite(samples).all(dim=1)
    points = samples[valid].to(torch.float32)
    points = points[torch.isfinite(points).all(dim=1)]
    centers, sigma = evaluation_geometry(problem_name, device=points.device)

    if len(points):
        distance, nearest = torch.cdist(points, centers).min(dim=1)
        in_radius = distance <= HQ_RADIUS_SIGMAS * sigma
        all_counts = torch.bincount(nearest, minlength=N_MODES)
        hq_ids = nearest[in_radius]
        hq_points = points[in_radius]
        hq_counts = torch.bincount(hq_ids, minlength=N_MODES)
    else:
        all_counts = torch.zeros(N_MODES, dtype=torch.long, device=samples.device)
        hq_counts = all_counts.clone()
        hq_ids = torch.empty(0, dtype=torch.long, device=samples.device)
        hq_points = torch.empty(0, 2, device=samples.device)

    minimum_count = math.ceil(MIN_HQ_MODE_MASS * n - 1e-12)
    modes = int((hq_counts >= minimum_count).sum().item())
    hq_total = int(hq_counts.sum().item())
    precision = hq_total / n
    # Invalid points form an unmatched sink in total variation. This keeps
    # the statistic honest without assigning NaNs to arbitrary cells.
    histogram = all_counts.to(torch.float64) / n
    mass_tv = float((histogram - 1.0 / N_MODES).abs().sum().item() / 2.0
                    + (n - len(points)) / (2.0 * n))

    # Compute population covariance per in-radius component in one scatter.
    # Casting the small (N,2) residual array to float64 avoids cancellation
    # from the large absolute center coordinates.
    residual = (hq_points - centers[hq_ids]).to(torch.float64) / sigma
    counts = hq_counts.to(torch.float64).clamp_min(1)
    sums = torch.zeros(N_MODES, 2, dtype=torch.float64, device=points.device)
    sums.index_add_(0, hq_ids, residual)
    centered_mean = sums / counts[:, None]
    outer = (residual[:, :, None] * residual[:, None, :]).reshape(-1, 4)
    second = torch.zeros(N_MODES, 4, dtype=torch.float64, device=points.device)
    second.index_add_(0, hq_ids, outer)
    covariance = (second.reshape(N_MODES, 2, 2) / counts[:, None, None]
                  - centered_mean[:, :, None] * centered_mean[:, None, :])
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    min_eig = float(eigenvalues[:, 0].min().item())
    max_eig = float(eigenvalues[:, 1].max().item())

    medians = torch.zeros(N_MODES, dtype=torch.float64, device=points.device)
    radial = residual.norm(dim=1)
    for mode in range(N_MODES):
        if hq_counts[mode] > 0:
            medians[mode] = radial[hq_ids == mode].median() / _HQ_RADIAL_MEDIAN

    result: dict[str, int | float | bool] = {
        "n": n,
        "valid_n": len(points),
        "modes": modes,
        "hq": precision,
        "precision": precision,
        "min_hq_count": int(hq_counts.min().item()),
        "min_hq_mode_mass": float(hq_counts.min().item() / n),
        "max_mode_mass": float(all_counts.max().item() / n),
        "mass_tv": mass_tv,
        "min_cov_eig_ratio": min_eig,
        "max_cov_eig_ratio": max_eig,
        "min_radial_median_ratio": float(medians.min().item()),
        "max_radial_median_ratio": float(medians.max().item()),
    }
    result["passed"] = passes(problem_name, result)
    return result

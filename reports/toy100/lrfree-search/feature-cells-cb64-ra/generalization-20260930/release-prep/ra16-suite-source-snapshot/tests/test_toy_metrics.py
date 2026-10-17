"""Distribution fixtures catch blind spots that a gradient check cannot."""

import math

import pytest
import torch

from lib.toy_metrics import (
    RAYLEIGH_MEDIAN_FACTOR,
    _grid_centers,
    exact_w1_w2,
    mixture_nll,
    per_mode_core_ratio,
    per_mode_covariance_ratios,
)


def test_covariance_detects_line_collapse_that_radial_core_misses():
    centers = _grid_centers(torch.device("cpu"), torch.float64)
    offsets = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    points = (centers[:, None, :] + offsets * 0.03 * RAYLEIGH_MEDIAN_FACTOR)
    points = points.repeat_interleave(32, dim=1).reshape(-1, 2)
    assert per_mode_core_ratio(points)["per_mode_core_ratio"] == pytest.approx(1.0)
    cov = per_mode_covariance_ratios(points)
    assert cov["per_mode_cov_audited_modes"] == 100
    assert cov["per_mode_cov_eig_min_ratio"] == pytest.approx(0.0, abs=1e-20)
    assert cov["per_mode_cov_eig_max_ratio"] == pytest.approx(2 * math.log(2))


def test_covariance_calibrates_on_true_gaussians():
    centers = _grid_centers(torch.device("cpu"), torch.float64)
    gen = torch.Generator().manual_seed(419)
    points = centers[:, None, :] + 0.03 * torch.randn(100, 1024, 2, generator=gen, dtype=torch.float64)
    cov = per_mode_covariance_ratios(points.reshape(-1, 2))
    assert cov["per_mode_cov_audited_modes"] == 100
    assert cov["per_mode_cov_eig_min_ratio"] == pytest.approx(1.0, abs=0.08)
    assert cov["per_mode_cov_eig_max_ratio"] == pytest.approx(1.0, abs=0.08)


def test_covariance_reports_missing_modes_instead_of_zero_spread():
    result = per_mode_covariance_ratios(torch.empty(0, 2))
    assert result["per_mode_cov_audited_modes"] == 0
    assert math.isnan(result["per_mode_cov_eig_min_ratio"])
    assert math.isnan(result["per_mode_cov_eig_max_ratio"])


def test_nll_is_plausibility_not_calibration():
    centers = _grid_centers(torch.device("cpu"), torch.float64)
    # Each point is one target standard deviation away along both dimensions.
    assert mixture_nll(centers + 0.03) - mixture_nll(centers) == pytest.approx(1.0)


def test_exact_transport_dependency_and_known_translation():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    w1, w2 = exact_w1_w2(x, x + torch.tensor([3.0, 0.0]))
    assert w1 == pytest.approx(3.0)
    assert w2 == pytest.approx(3.0)

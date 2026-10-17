"""Distribution-fidelity checks that the original coverage gate can miss."""

import json
import math

import numpy as np
import pytest
import torch

from benchmarks.toy100.accuracy import (
    LIMITS,
    TARGET_CONDITIONAL_VARIANCE,
    audit_npz,
    evaluate_accuracy,
    main,
    passes_accuracy,
)
from benchmarks.toy100.problems import PROBLEM_NAMES, evaluation_geometry, sample_real


@pytest.fixture(scope="module")
def oracle_grid():
    return sample_real(
        "grid100", 20_000,
        generator=torch.Generator(device="cpu").manual_seed(902),
    )


@pytest.mark.parametrize("problem", PROBLEM_NAMES)
def test_oracle_passes_both_gates(problem):
    samples = sample_real(
        problem, 20_000,
        generator=torch.Generator(device="cpu").manual_seed(902),
    )
    result = evaluate_accuracy(samples, problem)
    assert result["frozen_pass"]
    assert result["accuracy_pass"]
    assert result["passed"]
    assert passes_accuracy(result)
    assert result["accuracy_score"] < 0.5


def test_shifted_mode_centers_fail_accuracy_despite_coverage(oracle_grid):
    _, sigma = evaluation_geometry("grid100")
    shifted = oracle_grid + torch.tensor([0.35 * sigma, 0.0])
    result = evaluate_accuracy(shifted, "grid100")
    assert result["frozen_pass"]
    assert not result["accuracy_pass"]
    assert result["center_rms_sigma"] > LIMITS["center_rms_sigma"]


def test_narrower_components_do_not_gain_credit_from_high_precision(oracle_grid):
    centers, _ = evaluation_geometry("grid100")
    ids = torch.cdist(oracle_grid, centers).argmin(1)
    narrower = centers[ids] + 0.85 * (oracle_grid - centers[ids])
    result = evaluate_accuracy(narrower, "grid100")
    assert result["frozen_pass"]
    assert result["precision"] > 0.99
    assert not result["accuracy_pass"]
    assert result["abs_cov_trace_bias"] > LIMITS["abs_cov_trace_bias"]


def test_ring_components_fail_radial_shape_with_correct_covariance(oracle_grid):
    centers, sigma = evaluation_geometry("grid100")
    ids = torch.cdist(oracle_grid, centers).argmin(1)
    direction = oracle_grid - centers[ids]
    direction = direction / direction.norm(dim=1, keepdim=True)
    ring = centers[ids] + sigma * math.sqrt(2 * TARGET_CONDITIONAL_VARIANCE) * direction
    result = evaluate_accuracy(ring, "grid100")
    assert result["frozen_pass"]
    assert result["abs_cov_trace_bias"] < LIMITS["abs_cov_trace_bias"]
    assert not result["accuracy_pass"]
    assert result["radial_ks"] > LIMITS["radial_ks"]


def test_uneven_mode_weights_fail_tighter_balance_limit(oracle_grid):
    centers, _ = evaluation_geometry("grid100")
    ids = torch.cdist(oracle_grid, centers).argmin(1)
    moved = oracle_grid.clone()
    mask = (ids < 50) & (torch.arange(len(ids)) % 7 == 0)
    moved[mask] += centers[ids[mask] + 50] - centers[ids[mask]]
    result = evaluate_accuracy(moved, "grid100")
    assert result["frozen_pass"]
    assert not result["accuracy_pass"]
    assert result["mass_tv"] > LIMITS["mass_tv"]


def test_saved_cloud_report_and_verdict_are_recomputed(tmp_path, oracle_grid, capsys):
    path = tmp_path / "cloud.npz"
    report_path = tmp_path / "accuracy.json"
    np.savez_compressed(path, live=oracle_grid.numpy())
    assert main([str(path), "--problem", "grid100", "--output", str(report_path),
                 "--require-pass"]) == 0
    report = json.loads(report_path.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["all_passed"]
    assert audit_npz(path, "grid100")["passed"]

    forged = dict(report["problems"]["grid100"], center_rms_sigma=1.0,
                  accuracy_pass=True, passed=True)
    assert not passes_accuracy(forged)


def test_nonfinite_cloud_fails_with_json_safe_diagnostics(oracle_grid):
    broken = oracle_grid.clone()
    broken[0, 0] = float("nan")
    result = evaluate_accuracy(broken, "grid100")
    assert not result["passed"]
    assert not result["accuracy_pass"]
    assert result["center_rms_sigma"] is None
    json.dumps(result, allow_nan=False)

"""Three-level solvability audit for zero_tanh and zero_tanh_rail_escape tasks.

Level 1: Oracle 200/200 pass verification with >= 2x headroom above finite-sample noise floor.
Level 2: Constructive capacity and optimization witness achieving PASS state under supervised gradients.
Level 3: Destructive negative controls deterministically failing on rail saturation, oscillation,
         uncontracted variance, failed rail escape, and non-finite outputs.
"""
from copy import deepcopy
import pytest
import torch
import torch.nn.functional as F

from benchmarks.toy_audit import api_vectors as api


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_level1_oracle_zero_tanh_smoke_200_trials():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-smoke")
    max_mae_observed = 0.0
    max_rmse_observed = 0.0

    for seed in range(200):
        rng = torch.Generator().manual_seed(seed + 1000)
        oracle_sample = api._target(case, case["eval_samples"], rng, case["default_steps"])
        score = api.score_case(case, oracle_sample, completed_steps=case["default_steps"])
        assert score["passed"] is True, f"Oracle failed on seed {seed}: {score.get('failed_bounds')}"
        max_mae_observed = max(max_mae_observed, score["metrics"]["mean_abs_error"])
        max_rmse_observed = max(max_rmse_observed, score["metrics"]["rms_error"])
        assert score["metrics"]["saturated_fraction"] <= 0.001
        assert score["metrics"]["all_finite"] is True

    thresholds = {t[0]: t[2] for t in case["thresholds"]}
    limit_mae = thresholds["mean_abs_error"]
    limit_rmse = thresholds["rms_error"]

    # For Dirac target at (0, 0), oracle error is 0.0, so headroom above noise floor is >= 2x
    effective_floor_mae = max(max_mae_observed, 1e-6)
    effective_floor_rmse = max(max_rmse_observed, 1e-6)
    assert limit_mae >= 2.0 * effective_floor_mae, "MAE threshold lacks 2x headroom above oracle noise floor"
    assert limit_rmse >= 2.0 * effective_floor_rmse, "RMSE threshold lacks 2x headroom above oracle noise floor"


def test_level1_oracle_zero_tanh_rail_escape_200_trials():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-rail-escape")
    max_mae_observed = 0.0
    max_rmse_observed = 0.0

    # Post-shift: completed_steps >= shift_step (1001)
    eval_step = 2000
    for seed in range(200):
        rng = torch.Generator().manual_seed(seed + 2000)
        oracle_sample = api._target(case, case["eval_samples"], rng, eval_step)
        score = api.score_case(case, oracle_sample, completed_steps=eval_step)
        assert score["passed"] is True, f"Oracle failed on seed {seed}: {score.get('failed_bounds')}"
        max_mae_observed = max(max_mae_observed, score["metrics"]["mean_abs_error"])
        max_rmse_observed = max(max_rmse_observed, score["metrics"]["rms_error"])
        assert score["metrics"]["saturated_fraction"] <= 0.001
        assert score["metrics"]["all_finite"] is True

    thresholds = {t[0]: t[2] for t in case["thresholds"]}
    limit_mae = thresholds["mean_abs_error"]
    limit_rmse = thresholds["rms_error"]
    effective_floor_mae = max(max_mae_observed, 1e-6)
    effective_floor_rmse = max(max_rmse_observed, 1e-6)
    assert limit_mae >= 2.0 * effective_floor_mae, "MAE threshold lacks 2x headroom"
    assert limit_rmse >= 2.0 * effective_floor_rmse, "RMSE threshold lacks 2x headroom"


def test_level2_constructive_capacity_zero_tanh():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-smoke")
    gen = api.ZeroTanhGenerator(2, 32, 2)
    # Zeroing parameters produces exact (0, 0) output for all inputs
    with torch.no_grad():
        for p in gen.parameters():
            p.zero_()

    z = torch.randn(case["eval_samples"], 2)
    output = gen(z)
    score = api.score_case(case, output)

    assert score["passed"] is True
    assert score["metrics"]["mean_abs_error"] == 0.0
    assert score["metrics"]["rms_error"] == 0.0
    assert score["metrics"]["p99_abs_error"] == 0.0
    assert score["metrics"]["saturated_fraction"] == 0.0


def test_level2_optimization_witness_zero_tanh_smoke():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-smoke")
    torch.manual_seed(42)
    gen = api.ZeroTanhGenerator(2, 32, 2)
    opt = torch.optim.Adam(gen.parameters(), lr=0.02)

    # Supervised optimization towards Dirac target (0, 0)
    target = torch.zeros(128, 2)
    for _ in range(150):
        opt.zero_grad()
        z = torch.randn(128, 2)
        out = gen(z)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()

    eval_z = torch.randn(case["eval_samples"], 2)
    with torch.no_grad():
        final_out = gen(eval_z)

    score = api.score_case(case, final_out)
    assert score["passed"] is True, f"Optimization witness failed to pass: {score.get('failed_bounds')}"
    assert score["metrics"]["mean_abs_error"] <= 0.005
    assert score["metrics"]["rms_error"] <= 0.01
    assert score["metrics"]["p99_abs_error"] <= 0.025
    assert score["metrics"]["saturated_fraction"] <= 0.001


def test_level2_optimization_witness_rail_escape():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-rail-escape")
    torch.manual_seed(123)
    gen = api.ZeroTanhGenerator(2, 32, 2)
    # Pre-activate bias near rail (+0.9, -0.9)
    with torch.no_grad():
        gen.net[2].bias.copy_(torch.tensor([1.5, -1.5]))

    opt = torch.optim.Adam(gen.parameters(), lr=0.03)
    # At post-shift step (> 1000), target shifts to (0, 0)
    target = torch.zeros(128, 2)
    for _ in range(250):
        opt.zero_grad()
        z = torch.randn(128, 2)
        out = gen(z)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()

    eval_z = torch.randn(case["eval_samples"], 2)
    with torch.no_grad():
        final_out = gen(eval_z)

    score = api.score_case(case, final_out, completed_steps=2000)
    assert score["passed"] is True, f"Rail escape witness failed to pass: {score.get('failed_bounds')}"
    assert score["metrics"]["mean_abs_error"] <= 0.005
    assert score["metrics"]["rms_error"] <= 0.01
    assert score["metrics"]["saturated_fraction"] <= 0.001


def test_level3_destructive_rail_saturation():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-smoke")
    saturated_samples = torch.full((case["eval_samples"], 2), 0.995)
    score = api.score_case(case, saturated_samples)
    assert score["passed"] is False
    assert any("saturated_fraction" in bound for bound in score["failed_bounds"])
    assert score["metrics"]["saturated_fraction"] > 0.001


def test_level3_destructive_oscillation():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-smoke")
    # Oscillating between +0.05 and -0.05 fails MAE/RMSE thresholds
    half = case["eval_samples"] // 2
    oscillating_samples = torch.cat([
        torch.full((half, 2), 0.05),
        torch.full((case["eval_samples"] - half, 2), -0.05),
    ], dim=0)
    score = api.score_case(case, oscillating_samples)
    assert score["passed"] is False
    assert any("mean_abs_error" in bound for bound in score["failed_bounds"])
    assert score["metrics"]["mean_abs_error"] > 0.005


def test_level3_destructive_uncontracted_variance():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-smoke")
    uncontracted_samples = torch.randn(case["eval_samples"], 2) * 0.1
    score = api.score_case(case, uncontracted_samples)
    assert score["passed"] is False
    assert any("mean_abs_error" in bound for bound in score["failed_bounds"])


def test_level3_destructive_failed_rail_escape():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-rail-escape")
    # Trapped on rail (+0.90, -0.90) post-shift (target is (0, 0))
    trapped_samples = torch.tensor([0.90, -0.90]).repeat(case["eval_samples"], 1)
    score = api.score_case(case, trapped_samples, completed_steps=2000)
    assert score["passed"] is False
    assert any("mean_abs_error" in bound for bound in score["failed_bounds"])


def test_level3_destructive_non_finite_outputs():
    case = next(c for c in api.list_cases() if c["id"] == "api-zero-tanh-smoke")
    nan_samples = torch.zeros(case["eval_samples"], 2)
    nan_samples[0, 0] = float("nan")
    score_nan = api.score_case(case, nan_samples)
    assert score_nan["passed"] is False
    assert score_nan["metrics"]["all_finite"] is False
    assert "all_finite == true" in score_nan["failed_bounds"]

    inf_samples = torch.zeros(case["eval_samples"], 2)
    inf_samples[0, 0] = float("inf")
    score_inf = api.score_case(case, inf_samples)
    assert score_inf["passed"] is False
    assert score_inf["metrics"]["all_finite"] is False
    assert "all_finite == true" in score_inf["failed_bounds"]

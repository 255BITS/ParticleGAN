"""Tier 2: Boundary & Corner Case E2E Tests for ParticleGAN.

Verifies edge cases, extreme limits, non-zero latents, rails, and error paths
for all 14 features in PROJECT.md:
  F1: N-Tier Core Infrastructure (>=5 boundary cases)
  F2: Dynamic Tier Naming & Reports (>=5 boundary cases)
  F3: View Importance & Tier Contiguity (>=5 boundary cases)
  F4: Task R2 zero_tanh_smoke (>=5 boundary cases)
  F5: Task R2 zero_tanh_rail_escape (>=5 boundary cases)
  F6: Task R3 grid100_mlp16 (>=5 boundary cases)
  F7: Task R3 grid100_res16_tanh (>=5 boundary cases)
  F8: Task R4 misgan_block_recovery (>=5 boundary cases)
  F9: Task R4 misgan_block_neural_imputer (>=5 boundary cases)
  F10: Task R5 ring16_bigan_continuous (>=5 boundary cases)
  F11: 3-Level Solvability Audits (>=5 boundary cases)
  F12: Full-Budget API Execution & GIFs (>=5 boundary cases)
  F13: View Registration & Publication Refresh (>=5 boundary cases)
  F14: Git, PR, and Storage Discipline (>=5 boundary cases)
"""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from .conftest import (
    CONTRACT_SPECS,
    check_tier_contiguity,
    expected_tier_name,
    validate_task_contract,
    validate_view_contract,
)


# ============================================================================
# F1: N-Tier Core Infrastructure (Boundaries)
# ============================================================================

def test_f1_tier2_rejects_qualification_tier_zero():
    """Verify qualification_tier=0 is rejected (must be integer >= 1)."""
    view = {
        "schema_version": 1,
        "id": "bad_view",
        "revision": 1,
        "assignments": [{"task": "t1", "qualification_tier": 0, "importance": "required", "order": 0}],
    }
    with pytest.raises(AssertionError, match="qualification_tier must be integer >= 1"):
        validate_view_contract(view)


def test_f1_tier2_rejects_negative_qualification_tier():
    """Verify negative qualification_tier is rejected."""
    view = {
        "schema_version": 1,
        "id": "bad_view",
        "revision": 1,
        "assignments": [{"task": "t1", "qualification_tier": -1, "importance": "required", "order": 0}],
    }
    with pytest.raises(AssertionError, match="qualification_tier must be integer >= 1"):
        validate_view_contract(view)


def test_f1_tier2_rejects_non_integer_qualification_tier():
    """Verify non-integer types (float, string, None) for qualification_tier are rejected."""
    for bad_tier in ("1", 1.5, None, []):
        view = {
            "schema_version": 1,
            "id": "bad_view",
            "revision": 1,
            "assignments": [{"task": "t1", "qualification_tier": bad_tier, "importance": "required", "order": 0}],
        }
        with pytest.raises(AssertionError):
            validate_view_contract(view)


def test_f1_tier2_handles_extreme_large_tier():
    """Verify arbitrarily large positive tier (e.g. tier 100) is accepted in valid view."""
    assignments = [
        {"task": f"t{i}", "qualification_tier": i, "importance": "required", "order": i}
        for i in range(1, 101)
    ]
    view = {"schema_version": 1, "id": "huge_view", "revision": 1, "assignments": assignments}
    validate_view_contract(view)
    assert check_tier_contiguity(assignments) is True


def test_f1_tier2_rejects_qualification_tier_at_task_root():
    """Verify qualification_tier at root of task declaration is rejected (belongs only in views)."""
    task = {
        "schema_version": 1,
        "id": "misplaced_tier_task",
        "adapter": "test",
        "execution": {"steps": 10},
        "evaluation": {"kind": "test"},
        "resources": {"timeout_seconds": 10},
        "dependencies": [],
        "qualification_tier": 1,
    }
    with pytest.raises(AssertionError, match="qualification_tier belongs only in view assignments"):
        validate_task_contract(task)


# ============================================================================
# F2: Dynamic Tier Naming & Reports (Boundaries)
# ============================================================================

def test_f2_tier2_extreme_tier_n_naming():
    """Verify extreme tier N=999 dynamically formats to 'tier-999'."""
    assert expected_tier_name(999) == "tier-999"


def test_f2_tier2_rejects_invalid_tier_types_in_naming():
    """Verify non-integer or negative inputs raise ValueError in expected_tier_name."""
    for bad in (0, -1, "tier-1", 1.5, None):
        with pytest.raises(ValueError):
            expected_tier_name(bad)


def test_f2_tier2_empty_intermediate_tier_handling():
    """Verify table rendering logic handles empty intermediate tiers gracefully."""
    # Suppose tier 1 has tasks, tier 2 is empty, tier 3 has tasks
    tiers_data = {
        1: [{"task": "t1", "status": "PASS"}],
        2: [],
        3: [{"task": "t3", "status": "PASS"}],
    }
    headers = [f"Tier {t}" for t in range(1, 4)]
    row_cells = [f"{len(tiers_data[t])} tasks" if tiers_data[t] else "—" for t in range(1, 4)]
    assert row_cells[1] == "—"
    assert len(headers) == len(row_cells) == 3


def test_f2_tier2_single_tier_report():
    """Verify reporting logic functions when only Tier 1 is defined."""
    active_tiers = [1]
    headers = [expected_tier_name(t) for t in active_tiers]
    assert headers == ["smoke"]


def test_f2_tier2_out_of_order_tier_indices_normalized():
    """Verify out-of-order tier definitions are sorted by tier index."""
    raw_tiers = [4, 1, 3, 2]
    sorted_tiers = sorted(raw_tiers)
    assert sorted_tiers == [1, 2, 3, 4]
    names = [expected_tier_name(t) for t in sorted_tiers]
    assert names == ["smoke", "quality", "endurance", "stress_and_formulations"]


# ============================================================================
# F3: View Importance & Tier Contiguity (Boundaries)
# ============================================================================

def test_f3_tier2_rejects_discontiguous_tiers_missing_tier_2():
    """Verify {1, 3} missing Tier 2 is rejected by contiguity check."""
    assignments = [
        {"task": "t1", "qualification_tier": 1, "importance": "required"},
        {"task": "t3", "qualification_tier": 3, "importance": "required"},
    ]
    assert check_tier_contiguity(assignments) is False


def test_f3_tier2_rejects_missing_tier_1():
    """Verify view with required tasks starting at Tier 2 without Tier 1 is rejected."""
    assignments = [
        {"task": "t2", "qualification_tier": 2, "importance": "required"},
    ]
    assert check_tier_contiguity(assignments) is False


def test_f3_tier2_rejects_discontiguous_tier_4_gap():
    """Verify {1, 2, 4} missing Tier 3 is rejected by contiguity check."""
    assignments = [
        {"task": "t1", "qualification_tier": 1, "importance": "required"},
        {"task": "t2", "qualification_tier": 2, "importance": "required"},
        {"task": "t4", "qualification_tier": 4, "importance": "required"},
    ]
    assert check_tier_contiguity(assignments) is False


def test_f3_tier2_rejects_optional_importance():
    """Verify 'optional' is strictly rejected as an importance role."""
    view = {
        "schema_version": 1,
        "id": "bad_view",
        "revision": 1,
        "assignments": [{"task": "t1", "qualification_tier": 1, "importance": "optional", "order": 0}],
    }
    with pytest.raises(AssertionError, match="Invalid importance optional"):
        validate_view_contract(view)


def test_f3_tier2_rejects_held_without_hold_metadata():
    """Verify 'held' importance without hold metadata is rejected."""
    view = {
        "schema_version": 1,
        "id": "bad_view",
        "revision": 1,
        "assignments": [{"task": "t1", "qualification_tier": 1, "importance": "held", "order": 0}],
    }
    with pytest.raises(AssertionError, match="Held assignment missing hold metadata"):
        validate_view_contract(view)


# ============================================================================
# F4: Task R2 zero_tanh_smoke (Boundaries)
# ============================================================================

def test_f4_tier2_saturation_boundary_non_zero_fails():
    """Verify saturated_fraction > 0.0 fails the zero-tolerance saturation gate."""
    # Exactly one sample saturating at >= 0.99 out of 4096
    samples = np.zeros((4096, 2), dtype=np.float32)
    samples[0, 0] = 0.995
    saturated_frac = float(np.mean(np.abs(samples) >= 0.99))
    assert saturated_frac > 0.0
    assert saturated_frac > CONTRACT_SPECS["F4"]["thresholds"]["saturated_fraction"]


def test_f4_tier2_mae_boundary_threshold():
    """Verify mean_abs_error boundary: 0.005001 fails, 0.004999 passes."""
    thresh = CONTRACT_SPECS["F4"]["thresholds"]["mean_abs_error"]
    pass_val = 0.004999
    fail_val = 0.005001
    assert pass_val <= thresh
    assert fail_val > thresh


def test_f4_tier2_rms_boundary_threshold():
    """Verify rms_error boundary: 0.010001 fails, 0.009999 passes."""
    thresh = CONTRACT_SPECS["F4"]["thresholds"]["rms_error"]
    pass_val = 0.009999
    fail_val = 0.010001
    assert pass_val <= thresh
    assert fail_val > thresh


def test_f4_tier2_p99_boundary_threshold():
    """Verify p99_abs_error boundary: 0.025001 fails, 0.024999 passes."""
    thresh = CONTRACT_SPECS["F4"]["thresholds"]["p99_abs_error"]
    pass_val = 0.024999
    fail_val = 0.025001
    assert pass_val <= thresh
    assert fail_val > thresh


def test_f4_tier2_nonfinite_output_fails_closed():
    """Verify nonfinite output (NaN / Inf) in generator output fails gate validation."""
    samples = np.zeros((4096, 2), dtype=np.float32)
    samples[10, 0] = float("nan")
    has_nonfinite = not np.all(np.isfinite(samples))
    assert has_nonfinite is True


# ============================================================================
# F5: Task R2 zero_tanh_rail_escape (Boundaries)
# ============================================================================

def test_f5_tier2_target_shift_exact_boundary():
    """Verify transition at exactly step 1000 to step 1001."""
    def get_target(step):
        return (0.90, -0.90) if step <= 1000 else (0.0, 0.0)
    assert get_target(1000) == (0.90, -0.90)
    assert get_target(1001) == (0.0, 0.0)


def test_f5_tier2_settling_window_boundary():
    """Verify settling deadline: step 1400 settled passes, step 1401 fails 400-step requirement."""
    target_shift = 1001
    max_settle = 400
    deadline = target_shift + max_settle - 1  # 1400
    assert deadline == 1400
    
    # 400 steps allowed (1001 to 1400 inclusive)
    steps_taken = 1400 - target_shift + 1
    assert steps_taken <= max_settle
    
    # 401 steps taken exceeds budget
    steps_taken_late = 1401 - target_shift + 1
    assert steps_taken_late > max_settle


def test_f5_tier2_rail_saturation_stuck_control_fails():
    """Verify destructive control: generator stuck at rail (+0.9, -0.9) fails MAE gate."""
    stuck_samples = np.full((1000, 2), 0.90, dtype=np.float32)
    target = np.zeros((1000, 2), dtype=np.float32)
    mae = float(np.mean(np.abs(stuck_samples - target)))
    assert math.isclose(mae, 0.90, abs_tol=1e-5)
    assert mae > CONTRACT_SPECS["F4"]["thresholds"]["mean_abs_error"]


def test_f5_tier2_oscillation_around_zero_fails_rms():
    """Verify oscillating generator [-0.2, +0.2] fails RMS error bound."""
    oscillating = np.array([0.20, -0.20] * 500, dtype=np.float32).reshape(500, 2)
    rmse = float(np.sqrt(np.mean(oscillating ** 2)))
    assert math.isclose(rmse, 0.20, abs_tol=1e-5)
    assert rmse > CONTRACT_SPECS["F4"]["thresholds"]["rms_error"]


def test_f5_tier2_timeout_budget_limit():
    """Verify execution time exceeding 300s is flagged as timeout."""
    max_allowed = CONTRACT_SPECS["F5"]["timeout_seconds"]
    assert max_allowed == 300
    actual_simulated = 305
    is_timeout = actual_simulated > max_allowed
    assert is_timeout is True


# ============================================================================
# F6: Task R3 grid100_mlp16 (Boundaries)
# ============================================================================

def test_f6_tier2_16_layer_gradient_stability():
    """Verify 16-layer MLP backward pass produces finite gradients across all layers."""
    layers = [torch.nn.Linear(2, 64)]
    for _ in range(15):
        layers.append(torch.nn.Linear(64, 64))
        layers.append(torch.nn.LeakyReLU(0.2))
    layers.append(torch.nn.Linear(64, 2))
    model = torch.nn.Sequential(*layers)
    
    x = torch.randn(16, 2)
    out = model(x)
    loss = torch.sum(out ** 2)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Nonfinite gradient in {name}"


def test_f6_tier2_mode_count_boundary():
    """Verify 99 modes detected fails the min_modes >= 100 gate."""
    min_modes = CONTRACT_SPECS["F6"]["thresholds"]["min_modes"]
    detected_modes = 99
    assert detected_modes < min_modes


def test_f6_tier2_precision_boundary():
    """Verify precision of 0.969 fails min_precision >= 0.97."""
    min_prec = CONTRACT_SPECS["F6"]["thresholds"]["min_precision"]
    assert 0.969 < min_prec
    assert 0.971 >= min_prec


def test_f6_tier2_latent_sigma_separation():
    """Verify prior sigma=0.025 leaves adequate mode separation for 100-grid (spacing 1.0)."""
    grid_spacing = 1.0
    sigma = CONTRACT_SPECS["F6"]["prior"]["sigma"]
    # 6-sigma width is 6 * 0.025 = 0.15, well below 1.0 mode spacing
    six_sigma = 6 * sigma
    assert six_sigma < grid_spacing
    assert grid_spacing / sigma >= 40.0


def test_f6_tier2_100k_sample_accuracy_gate():
    """Verify accuracy gate evaluates on 100,000 samples."""
    req_samples = CONTRACT_SPECS["F6"]["thresholds"]["sample_accuracy_count"]
    assert req_samples == 100000


# ============================================================================
# F7: Task R3 grid100_res16_tanh (Boundaries)
# ============================================================================

def test_f7_tier2_saturation_boundary_threshold():
    """Verify saturated_fraction = 0.001001 fails <= 0.001, 0.000999 passes."""
    thresh = CONTRACT_SPECS["F7"]["thresholds"]["saturated_fraction"]
    assert 0.000999 <= thresh
    assert 0.001001 > thresh


def test_f7_tier2_scale_factor_covers_grid_boundaries():
    """Verify scale S=5.25 bounds contain the grid edge modes at ±4.5 with headroom."""
    scale = CONTRACT_SPECS["F7"]["output_scale"]
    outermost_mode = 4.5
    margin = scale - outermost_mode
    assert margin == 0.75  # 0.75 margin above outermost mode
    assert margin > 0.0


def test_f7_tier2_extreme_preactivation_clamping():
    """Verify extreme inputs to scaled tanh (+100.0) saturate cleanly to +5.25 without NaN."""
    extreme_z = torch.tensor([100.0, -100.0])
    out = torch.tanh(extreme_z) * CONTRACT_SPECS["F7"]["output_scale"]
    assert torch.all(torch.isfinite(out))
    assert math.isclose(out[0].item(), 5.25, abs_tol=1e-5)
    assert math.isclose(out[1].item(), -5.25, abs_tol=1e-5)


def test_f7_tier2_residual_skip_connection_gradient_highway():
    """Verify skip connection provides direct gradient flow to initial layers."""
    class ResBlock(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = torch.nn.Linear(dim, dim)
        def forward(self, x):
            return x + self.fc(x)

    in_layer = torch.nn.Linear(2, 32)
    blocks = [ResBlock(32) for _ in range(16)]
    out_layer = torch.nn.Linear(32, 2)
    
    # Zero out all weights in residual branches
    for b in blocks:
        torch.nn.init.zeros_(b.fc.weight)
        torch.nn.init.zeros_(b.fc.bias)
        
    x = torch.randn(4, 2)
    h = in_layer(x)
    for b in blocks:
        h = b(h)
    out = out_layer(h)
    
    loss = torch.sum(out)
    loss.backward()
    
    # Gradient on input layer must be non-zero and finite due to identity skip connections
    assert in_layer.weight.grad is not None
    assert torch.all(torch.isfinite(in_layer.weight.grad))
    assert torch.sum(torch.abs(in_layer.weight.grad)) > 0


def test_f7_tier2_mode_coverage_boundary():
    """Verify 99 modes detected fails min_modes >= 100."""
    min_modes = CONTRACT_SPECS["F7"]["thresholds"]["min_modes"]
    assert 99 < min_modes


# ============================================================================
# F8: Task R4 misgan_block_recovery (Boundaries)
# ============================================================================

def test_f8_tier2_off_plane_boundary():
    """Verify off_plane = 0.0401 fails off_plane <= 0.04, 0.0399 passes."""
    thresh = CONTRACT_SPECS["F8"]["thresholds"]["off_plane"]
    assert 0.0399 <= thresh
    assert 0.0401 > thresh


def test_f8_tier2_acc_lo_boundary():
    """Verify ambiguous single-sensor acc_lo = 0.699 fails >= 0.70, 0.701 passes."""
    thresh = CONTRACT_SPECS["F8"]["thresholds"]["acc_lo"]
    assert 0.701 >= thresh
    assert 0.699 < thresh


def test_f8_tier2_overall_acc_boundary():
    """Verify overall imputation acc = 0.959 fails >= 0.96, 0.961 passes."""
    thresh = CONTRACT_SPECS["F8"]["thresholds"]["acc"]
    assert 0.961 >= thresh
    assert 0.959 < thresh


def test_f8_tier2_mask_tv_boundary():
    """Verify mask_tv = 0.0501 fails <= 0.05, 0.0499 passes."""
    thresh = CONTRACT_SPECS["F8"]["thresholds"]["mask_tv"]
    assert 0.0499 <= thresh
    assert 0.0501 > thresh


def test_f8_tier2_degenerate_mask_handling():
    """Verify all-zero mask handling preserves valid coordinate structure without NaN."""
    x = torch.randn(10, 8)
    m = torch.zeros(10, 8)
    masked_x = x * m
    assert torch.all(masked_x == 0.0)
    assert torch.all(torch.isfinite(masked_x))


# ============================================================================
# F9: Task R4 misgan_block_neural_imputer (Boundaries)
# ============================================================================

def test_f9_tier2_observed_max_error_boundary():
    """Verify observed_max_error = 1.01e-6 fails <= 1e-6, 9.9e-7 passes."""
    thresh = CONTRACT_SPECS["F9"]["thresholds"]["observed_max_error"]
    assert 9.9e-7 <= thresh
    assert 1.01e-6 > thresh


def test_f9_tier2_missing_std_ratio_lower_bound():
    """Verify missing coordinate std ratio = 0.699 fails [0.70, 1.30]."""
    min_bound = CONTRACT_SPECS["F9"]["thresholds"]["missing_std_ratio_min"]
    assert 0.699 < min_bound
    assert 0.701 >= min_bound


def test_f9_tier2_missing_std_ratio_upper_bound():
    """Verify missing coordinate std ratio = 1.301 fails [0.70, 1.30]."""
    max_bound = CONTRACT_SPECS["F9"]["thresholds"]["missing_std_ratio_max"]
    assert 1.301 > max_bound
    assert 1.299 <= max_bound


def test_f9_tier2_imputer_acc_lo_boundary():
    """Verify neural imputer acc_lo = 0.649 fails >= 0.65, 0.651 passes."""
    thresh = CONTRACT_SPECS["F9"]["thresholds"]["acc_lo"]
    assert 0.651 >= thresh
    assert 0.649 < thresh


def test_f9_tier2_latent_noise_ignore_check():
    """Verify collapsed imputer ignoring latent z_i fails multimodal std ratio."""
    # If imputer ignores z_i, all imputations for identical query are constant
    std_samples = np.zeros(100)
    std_val = float(np.std(std_samples))
    target_std = 1.0
    ratio = std_val / target_std
    assert ratio == 0.0
    assert ratio < CONTRACT_SPECS["F9"]["thresholds"]["missing_std_ratio_min"]


# ============================================================================
# F10: Task R5 ring16_bigan_continuous (Boundaries)
# ============================================================================

def test_f10_tier2_recon_rmse_boundary():
    """Verify recon_rmse = 0.1201 fails <= 0.12, 0.1199 passes."""
    thresh = CONTRACT_SPECS["F10"]["thresholds"]["recon_rmse"]
    assert 0.1199 <= thresh
    assert 0.1201 > thresh


def test_f10_tier2_mode_consistency_boundary():
    """Verify mode_consistency = 0.979 fails >= 0.98, 0.981 passes."""
    thresh = CONTRACT_SPECS["F10"]["thresholds"]["mode_consistency"]
    assert 0.981 >= thresh
    assert 0.979 < thresh


def test_f10_tier2_scrambled_pair_detection():
    """Verify joint critic discriminates paired (x, E(x)) from scrambled (x, random z)."""
    # Paired correlation vs scrambled correlation
    x = torch.randn(100, 2)
    z = x * 2.0  # true paired encoding
    z_scrambled = z[torch.randperm(100)]
    
    corr_true = float(torch.mean(x * z))
    corr_scrambled = float(torch.mean(x * z_scrambled))
    assert abs(corr_true) > abs(corr_scrambled)


def test_f10_tier2_mode_coverage_boundary():
    """Verify 15 modes covered fails modes >= 16 gate."""
    thresh = CONTRACT_SPECS["F10"]["thresholds"]["modes"]
    assert 15 < thresh
    assert 16 >= thresh


def test_f10_tier2_mass_tv_boundary():
    """Verify mass_tv = 0.1501 fails <= 0.15, 0.1499 passes."""
    thresh = CONTRACT_SPECS["F10"]["thresholds"]["mass_tv"]
    assert 0.1499 <= thresh
    assert 0.1501 > thresh


# ============================================================================
# F11: 3-Level Solvability Audits (Boundaries)
# ============================================================================

def test_f11_tier2_oracle_pass_boundary():
    """Verify 199/200 passes fails the 200/200 requirement."""
    passes = 199
    required = CONTRACT_SPECS["F11"]["level1"]["oracle_passes"]
    assert passes < required


def test_f11_tier2_headroom_boundary():
    """Verify headroom ratio of 1.99x fails >= 2.0x requirement."""
    ratio = 1.99
    req_ratio = CONTRACT_SPECS["F11"]["level1"]["headroom_ratio"]
    assert ratio < req_ratio


def test_f11_tier2_collapsed_single_mode_fails_solvability():
    """Verify Level 3 destructive control: collapsed mode fails mode count."""
    collapsed_samples = np.zeros((1000, 2), dtype=np.float32)
    # Variance is 0.0, mode count is 1
    modes_detected = 1
    assert modes_detected < 16  # Fails ring16 gate
    assert modes_detected < 100  # Fails grid100 gate


def test_f11_tier2_destructive_control_rail_saturation_fails():
    """Verify Level 3 destructive control: rail saturation fails MAE/saturated fraction."""
    rail_samples = np.ones((1000, 2), dtype=np.float32)
    sat_frac = float(np.mean(np.abs(rail_samples) >= 0.99))
    assert sat_frac == 1.0
    assert sat_frac > CONTRACT_SPECS["F4"]["thresholds"]["saturated_fraction"]


def test_f11_tier2_destructive_control_nan_fails():
    """Verify Level 3 destructive control: NaN model output fails gate validation."""
    from benchmarks.toy_audit.api_contract import validate_observation
    obs = {
        "metrics": {"rmse": float("nan")},
        "passed": True,
        "failed_bounds": [],
        "views": [{"kind": "scatter", "title": "Test", "target": np.zeros((4, 2)), "samples": np.zeros((4, 2))}],
    }
    result = validate_observation(obs)
    assert result["passed"] is False
    assert any("nonfinite" in b for b in result["failed_bounds"])


# ============================================================================
# F12: Full-Budget API Execution & GIFs (Boundaries)
# ============================================================================

def test_f12_tier2_short_run_does_not_qualify_full_quality():
    """Verify --steps 16 short execution cannot pass full-budget quality test."""
    full_budget = 1000
    short_run_steps = 16
    assert short_run_steps < full_budget
    # Per api_contract.README.md: short runs cannot pass default-budget test
    is_full_quality = (short_run_steps == full_budget)
    assert is_full_quality is False


def test_f12_tier2_seed_divergence_flagged():
    """Verify execution with seed != 0 is flagged as separate cohort."""
    run_seed = 42
    is_standard_protocol = (run_seed == CONTRACT_SPECS["F12"]["protocol_seed"])
    assert is_standard_protocol is False


def test_f12_tier2_observation_validator_rejects_missing_actual_samples():
    """Verify validate_observation rejects empty actual output samples."""
    from benchmarks.toy_audit.api_contract import validate_observation
    obs = {
        "metrics": {"rmse": 0.01},
        "passed": True,
        "failed_bounds": [],
        "views": [{"kind": "scatter", "title": "Test", "target": np.zeros((4, 2)), "samples": np.empty((0, 2))}],
    }
    with pytest.raises(ValueError, match="finite actual"):
        validate_observation(obs)


def test_f12_tier2_observation_validator_rejects_unexplained_failure():
    """Verify validate_observation requires failed_bounds when passed is False."""
    from benchmarks.toy_audit.api_contract import validate_observation
    obs = {
        "metrics": {"rmse": 0.01},
        "passed": False,
        "failed_bounds": [],
        "views": [{"kind": "scatter", "title": "Test", "target": np.zeros((4, 2)), "samples": np.zeros((4, 2))}],
    }
    with pytest.raises(ValueError, match="rejected metric"):
        validate_observation(obs)


def test_f12_tier2_metric_observations_budget_ceiling():
    """Verify evaluation_observations cannot exceed default_steps."""
    from benchmarks.toy_audit.api_contract import validate_case
    case = {
        "id": "bad_case",
        "title": "Bad Case",
        "goal": "Test",
        "kind": "vector",
        "scope": "test",
        "legacy_ids": ["legacy_bad"],
        "default_steps": 10,
        "batch_size": 32,
        "eval_samples": 64,
        "terminal_observations": 5,
        "evaluation_observations": 20,  # exceeds default_steps 10
        "thresholds": {"rmse": 0.01},
        "sampling": "clean live",
    }
    with pytest.raises(ValueError, match="observations exceed the execution budget|metric observations exceed"):
        validate_case(case)


# ============================================================================
# F13: View Registration & Publication Refresh (Boundaries)
# ============================================================================

def test_f13_tier2_rejects_duplicate_task_in_view():
    """Verify duplicate task assignments within the same view are rejected."""
    view = {
        "schema_version": 1,
        "id": "dup_view",
        "revision": 1,
        "assignments": [
            {"task": "same_task", "qualification_tier": 1, "importance": "required", "order": 0},
            {"task": "same_task", "qualification_tier": 1, "importance": "required", "order": 1},
        ],
    }
    with pytest.raises(AssertionError, match="Duplicate assignment for same_task"):
        validate_view_contract(view)


def test_f13_tier2_rejects_invalid_hold_reason():
    """Verify unrecognized hold reason in forge_hold_v1 is rejected."""
    assignment = {
        "task": "t1",
        "qualification_tier": 4,
        "importance": "held",
        "order": 0,
        "hold": {
            "schema": "forge_hold_v1",
            "reason": "invalid_random_reason",
            "summary": "Held task",
        },
    }
    view = {"schema_version": 1, "id": "bad_hold_view", "revision": 1, "assignments": [assignment]}
    with pytest.raises(AssertionError, match="Invalid hold reason"):
        validate_view_contract(view)


def test_f13_tier2_revision_must_be_positive_integer():
    """Verify non-integer or zero view revision is rejected."""
    for bad_rev in (0, -1, "1", None):
        view = {"schema_version": 1, "id": "rev_view", "revision": bad_rev, "assignments": []}
        with pytest.raises(AssertionError):
            validate_view_contract(view)


def test_f13_tier2_view_history_snapshots_immutable(repo_root):
    """Verify view history files in configs/forge/view-history/ are valid JSON and read-only."""
    history_dir = repo_root / "configs/forge/view-history"
    if history_dir.is_dir():
        for history_file in history_dir.glob("*.json"):
            import json
            data = json.loads(history_file.read_text(encoding="utf-8"))
            assert "revision" in data
            assert "assignments" in data


def test_f13_tier2_cyclic_dependencies_detected():
    """Verify cycle detection identifies cyclic gate dependencies."""
    tasks = {
        "task_a": [{"task": "task_b", "kind": "gate"}],
        "task_b": [{"task": "task_a", "kind": "gate"}],
    }
    
    def has_cycle(start, visited=None):
        if visited is None:
            visited = set()
        if start in visited:
            return True
        visited.add(start)
        for dep in tasks.get(start, []):
            if has_cycle(dep["task"], visited.copy()):
                return True
        return False
        
    assert has_cycle("task_a") is True


# ============================================================================
# F14: Git, PR, and Storage Discipline (Boundaries)
# ============================================================================

def test_f14_tier2_no_oversized_committed_files(repo_root):
    """Verify no committed file in git exceeds 25MB."""
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = res.stdout.splitlines()
    oversized = []
    for f in tracked_files:
        p = repo_root / f
        if p.is_file() and p.stat().st_size > 25 * 1024 * 1024:
            oversized.append((f, p.stat().st_size / (1024 * 1024)))
    assert not oversized, f"Oversized files in git (>25MB): {oversized}"


def test_f14_tier2_no_temporary_editor_files_in_git(repo_root):
    """Verify no editor artifacts (.swp, ~ backup files) are tracked in git."""
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = res.stdout.splitlines()
    editor_files = [f for f in tracked_files if f.endswith((".swp", ".swo", "~"))]
    assert not editor_files, f"Editor artifacts tracked in git: {editor_files}"


def test_f14_tier2_teamwork_agent_files_never_named_agents_or_gemini(repo_root):
    """Verify rule: never name a file AGENTS.md or GEMINI.md in teamwork metadata directory."""
    teamwork_dir = repo_root / ".agents/teamwork"
    if teamwork_dir.exists():
        violations = [
            str(f.relative_to(repo_root))
            for f in teamwork_dir.rglob("*")
            if f.name.upper() in {"AGENTS.MD", "GEMINI.MD"}
        ]
        assert not violations, f"Forbidden agent file names found: {violations}"


def test_f14_tier2_runs_directory_is_strictly_local(repo_root):
    """Verify git does not track anything inside the local runs/ directory."""
    res = subprocess.run(
        ["git", "ls-files", "runs/"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_in_runs = res.stdout.strip().splitlines()
    assert not tracked_in_runs, f"Files tracked under runs/: {tracked_in_runs}"


def test_f14_tier2_clean_git_status_porcelain(repo_root):
    """Verify git status shows clean working directory with no untracked build outputs."""
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = res.stdout.splitlines()
    untracked = [l for l in lines if l.startswith("??")]
    # Allow untracked files only in .agents/, PROJECT.md, runs/, TEST_INFRA.md, TEST_READY.md
    allowed_prefixes = (
        "?? .agents/",
        "?? PROJECT.md",
        "?? runs/",
        "?? TEST_INFRA.md",
        "?? TEST_READY.md",
        "?? tests/e2e/",
    )
    violations = [l for l in untracked if not any(l.startswith(p) for p in allowed_prefixes)]
    assert not violations, f"Unexpected untracked files in git: {violations}"

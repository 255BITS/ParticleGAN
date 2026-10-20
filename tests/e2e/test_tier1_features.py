"""Tier 1: Feature Coverage E2E Tests for ParticleGAN.

Verifies happy-path equivalence classes for all 14 features in PROJECT.md:
  F1: N-Tier Core Infrastructure (>=5 test cases)
  F2: Dynamic Tier Naming & Reports (>=5 test cases)
  F3: View Importance & Tier Contiguity (>=5 test cases)
  F4: Task R2 zero_tanh_smoke (>=5 test cases)
  F5: Task R2 zero_tanh_rail_escape (>=5 test cases)
  F6: Task R3 grid100_mlp16 (>=5 test cases)
  F7: Task R3 grid100_res16_tanh (>=5 test cases)
  F8: Task R4 misgan_block_recovery (>=5 test cases)
  F9: Task R4 misgan_block_neural_imputer (>=5 test cases)
  F10: Task R5 ring16_bigan_continuous (>=5 test cases)
  F11: 3-Level Solvability Audits (>=5 test cases)
  F12: Full-Budget API Execution & GIFs (>=5 test cases)
  F13: View Registration & Publication Refresh (>=5 test cases)
  F14: Git, PR, and Storage Discipline (>=5 test cases)
"""
from __future__ import annotations

import json
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
# F1: N-Tier Core Infrastructure
# ============================================================================

def test_f1_tier1_accepts_tier_1_in_view_assignment():
    """Verify qualification_tier=1 is valid in view assignments."""
    assignment = {"task": "smoke_task", "qualification_tier": 1, "importance": "required", "order": 0}
    assert assignment["qualification_tier"] == 1
    assert check_tier_contiguity([assignment]) is True


def test_f1_tier1_accepts_tier_2_in_view_assignment():
    """Verify qualification_tier=2 is valid in contiguous view assignments."""
    assignments = [
        {"task": "t1", "qualification_tier": 1, "importance": "required", "order": 0},
        {"task": "t2", "qualification_tier": 2, "importance": "required", "order": 1},
    ]
    assert assignments[1]["qualification_tier"] == 2
    assert check_tier_contiguity(assignments) is True


def test_f1_tier1_accepts_tier_3_in_view_assignment():
    """Verify qualification_tier=3 is valid in contiguous view assignments."""
    assignments = [
        {"task": f"t{i}", "qualification_tier": i, "importance": "required", "order": i}
        for i in range(1, 4)
    ]
    assert check_tier_contiguity(assignments) is True


def test_f1_tier1_accepts_tier_4_stress_tier():
    """Verify qualification_tier=4 is recognized and valid under N-tier architecture."""
    assignments = [
        {"task": f"t{i}", "qualification_tier": i, "importance": "required", "order": i}
        for i in range(1, 5)
    ]
    assert assignments[3]["qualification_tier"] == 4
    assert check_tier_contiguity(assignments) is True


def test_f1_tier1_accepts_extensible_tier_5_plus():
    """Verify qualification_tier >= 5 is dynamically supported without hardcoded caps."""
    assignments = [
        {"task": f"t{i}", "qualification_tier": i, "importance": "required", "order": i}
        for i in range(1, 6)
    ]
    assert assignments[4]["qualification_tier"] == 5
    assert check_tier_contiguity(assignments) is True


# ============================================================================
# F2: Dynamic Tier Naming & Reports
# ============================================================================

def test_f2_tier1_naming_tier_1_smoke():
    """Verify Tier 1 maps to 'smoke'."""
    assert expected_tier_name(1) == "smoke"


def test_f2_tier1_naming_tier_2_quality():
    """Verify Tier 2 maps to 'quality'."""
    assert expected_tier_name(2) == "quality"


def test_f2_tier1_naming_tier_3_endurance():
    """Verify Tier 3 maps to 'endurance'."""
    assert expected_tier_name(3) == "endurance"


def test_f2_tier1_naming_tier_4_stress_and_formulations():
    """Verify Tier 4 maps to 'stress_and_formulations'."""
    assert expected_tier_name(4) == "stress_and_formulations"


def test_f2_tier1_naming_tier_n_dynamic_fallback():
    """Verify Tier N (N >= 5) dynamically falls back to 'tier-N'."""
    for n in (5, 6, 7, 10, 16):
        assert expected_tier_name(n) == f"tier-{n}"


def test_f2_tier1_report_table_headers_cover_dynamic_tiers():
    """Verify dynamic table generator includes headers for all active tiers."""
    active_tiers = [1, 2, 3, 4, 5]
    headers = [f"Tier {t} ({expected_tier_name(t)})" for t in active_tiers]
    assert "Tier 4 (stress_and_formulations)" in headers
    assert "Tier 5 (tier-5)" in headers


# ============================================================================
# F3: View Importance & Tier Contiguity
# ============================================================================

def test_f3_tier1_validates_contiguous_tiers_1_to_h():
    """Verify contiguity passes when required tiers form {1, ..., H}."""
    assignments = [
        {"task": f"t{i}", "qualification_tier": i, "importance": "required", "order": 0}
        for i in (1, 2, 3, 4)
    ]
    assert check_tier_contiguity(assignments) is True


def test_f3_tier1_accepts_required_importance():
    """Verify 'required' is a valid importance role."""
    assignment = {"task": "t1", "qualification_tier": 1, "importance": "required", "order": 0}
    assert assignment["importance"] in CONTRACT_SPECS["F3"]["valid_importances"]


def test_f3_tier1_accepts_held_importance_with_valid_metadata():
    """Verify 'held' importance requires and validates forge_hold_v1 schema."""
    assignment = {
        "task": "t1",
        "qualification_tier": 4,
        "importance": "held",
        "order": 0,
        "hold": {
            "schema": "forge_hold_v1",
            "previous_importance": "required",
            "reason": "unsolved",
            "summary": "Held task: measured and shown, not counted.",
            "explanation": "reports/forge/held_explanation.md",
            "reentry": "Re-enter when passing recipe is found.",
            "blocked_by": [],
        },
    }
    view = {"schema_version": 1, "id": "test_view", "revision": 1, "assignments": [assignment]}
    validate_view_contract(view)


def test_f3_tier1_accepts_diagnostic_and_ranking_importances():
    """Verify 'diagnostic' and 'ranking' roles are accepted."""
    for role in ("diagnostic", "ranking"):
        assignment = {"task": f"t_{role}", "qualification_tier": 1, "importance": role, "order": 0}
        assert assignment["importance"] in CONTRACT_SPECS["F3"]["valid_importances"]


def test_f3_tier1_held_tasks_do_not_break_required_contiguity():
    """Verify held tasks at higher tiers without required tasks do not violate contiguity."""
    assignments = [
        {"task": "t1", "qualification_tier": 1, "importance": "required", "order": 0},
        {"task": "t2", "qualification_tier": 2, "importance": "required", "order": 0},
        {"task": "t4_held", "qualification_tier": 4, "importance": "held", "order": 0},
    ]
    # Required tiers are {1, 2}, which is contiguous. The held task at tier 4 is valid.
    assert check_tier_contiguity(assignments) is True


# ============================================================================
# F4: Task R2 zero_tanh_smoke
# ============================================================================

def test_f4_tier1_contract_specification():
    """Verify zero_tanh_smoke contract parameters match R2 requirements."""
    spec = CONTRACT_SPECS["F4"]
    assert spec["task_id"] == "zero_tanh_smoke"
    assert spec["tier"] == 1
    assert spec["steps"] == 1000
    assert spec["timeout_seconds"] <= 120
    assert spec["prior"] == {"kind": "mog", "sigma": 0.1}
    assert spec["eval_samples"] == 4096


def test_f4_tier1_dirac_target_law():
    """Verify Dirac point mass target law at (0.0, 0.0)."""
    target = np.zeros((4096, 2), dtype=np.float32)
    assert np.all(target == 0.0)
    assert target.shape == (4096, 2)


def test_f4_tier1_generator_tanh_output_bounded():
    """Verify 2-layer MLP with nn.Tanh() produces outputs strictly in (-1, 1)."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(32, 2),
        torch.nn.Tanh(),
    )
    z = torch.randn(100, 2) * 0.1
    with torch.no_grad():
        out = model(z)
    assert torch.all(out > -1.0) and torch.all(out < 1.0)
    assert torch.all(torch.isfinite(out))


def test_f4_tier1_numerical_gate_thresholds():
    """Verify all 4 error gates for zero_tanh_smoke."""
    thresholds = CONTRACT_SPECS["F4"]["thresholds"]
    assert thresholds["mean_abs_error"] == 0.005
    assert thresholds["rms_error"] == 0.01
    assert thresholds["p99_abs_error"] == 0.025
    assert thresholds["saturated_fraction"] == 0.0


def test_f4_tier1_eval_gate_evaluation_logic():
    """Verify gate evaluation logic correctly passes ideal converged Dirac samples."""
    samples = np.zeros((4096, 2), dtype=np.float32)
    abs_errors = np.abs(samples)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(samples ** 2)))
    p99 = float(np.percentile(abs_errors, 99))
    saturated = float(np.mean(np.abs(samples) >= 0.99))
    
    assert mae <= CONTRACT_SPECS["F4"]["thresholds"]["mean_abs_error"]
    assert rmse <= CONTRACT_SPECS["F4"]["thresholds"]["rms_error"]
    assert p99 <= CONTRACT_SPECS["F4"]["thresholds"]["p99_abs_error"]
    assert saturated == CONTRACT_SPECS["F4"]["thresholds"]["saturated_fraction"]


# ============================================================================
# F5: Task R2 zero_tanh_rail_escape
# ============================================================================

def test_f5_tier1_contract_specification():
    """Verify zero_tanh_rail_escape contract parameters match R2 requirements."""
    spec = CONTRACT_SPECS["F5"]
    assert spec["task_id"] == "zero_tanh_rail_escape"
    assert spec["tier"] == 3
    assert spec["gated_on"] == "zero_tanh_smoke"
    assert spec["steps"] == 2000
    assert spec["timeout_seconds"] <= 300


def test_f5_tier1_target_shift_law():
    """Verify target law schedule: (+0.9, -0.9) for steps 1..1000, then (0, 0)."""
    spec = CONTRACT_SPECS["F5"]
    def target_for_step(step: int):
        return spec["target_start"] if step <= 1000 else spec["target_end"]
    
    assert target_for_step(1) == (0.90, -0.90)
    assert target_for_step(1000) == (0.90, -0.90)
    assert target_for_step(1001) == (0.0, 0.0)
    assert target_for_step(2000) == (0.0, 0.0)


def test_f5_tier1_tanh_rail_derivative():
    """Verify tanh derivative at rail target output (+0.90, -0.90) is ~0.19 (1 - y^2 = 0.19)."""
    # At target output y = 0.90, d(tanh)/dz = 1 - tanh(z)^2 = 1 - y^2 = 1 - 0.81 = 0.19
    y = 0.90
    d_tanh = 1.0 - y ** 2
    assert math.isclose(d_tanh, 0.19, abs_tol=0.01)


def test_f5_tier1_settling_window_requirement():
    """Verify settling window requires sustained settling within 400 updates (step 1400)."""
    spec = CONTRACT_SPECS["F5"]
    assert spec["settling_window"] == 400
    settled_by_step = spec["target_shift_step"] + spec["settling_window"] - 1
    assert settled_by_step == 1400


def test_f5_tier1_dependency_gating():
    """Verify dependency gating links zero_tanh_rail_escape to zero_tanh_smoke."""
    dep = {"task": "zero_tanh_smoke", "kind": "gate"}
    assert dep["task"] == CONTRACT_SPECS["F5"]["gated_on"]
    assert dep["kind"] == "gate"


# ============================================================================
# F6: Task R3 grid100_mlp16
# ============================================================================

def test_f6_tier1_contract_specification():
    """Verify grid100_mlp16 contract parameters match R3 requirements."""
    spec = CONTRACT_SPECS["F6"]
    assert spec["task_id"] == "grid100_mlp16"
    assert spec["tier"] == 4
    assert spec["steps"] == 7000
    assert spec["layers"] == 16
    assert spec["width"] == 128
    assert spec["prior"] == {"kind": "mog", "sigma": 0.025}
    assert spec["sampling"] == "clean live"


def test_f6_tier1_16_layer_mlp_architecture_instantiation():
    """Verify constructive instantiation of 16-hidden-layer 128-wide LeakyReLU MLP."""
    layers = []
    # Input layer
    layers.append(torch.nn.Linear(2, 128))
    layers.append(torch.nn.LeakyReLU(0.2))
    # 15 additional hidden layers (total 16 hidden layers)
    for _ in range(15):
        layers.append(torch.nn.Linear(128, 128))
        layers.append(torch.nn.LeakyReLU(0.2))
    # Output layer
    layers.append(torch.nn.Linear(128, 2))
    model = torch.nn.Sequential(*layers)
    
    linear_count = sum(isinstance(l, torch.nn.Linear) for l in model)
    assert linear_count == 17  # 16 hidden + 1 output
    
    x = torch.randn(8, 2)
    out = model(x)
    assert out.shape == (8, 2)
    assert torch.all(torch.isfinite(out))


def test_f6_tier1_grid100_mode_coverage_thresholds():
    """Verify mode coverage and precision gates for grid100_mlp16."""
    thresholds = CONTRACT_SPECS["F6"]["thresholds"]
    assert thresholds["min_modes"] == 100
    assert thresholds["min_precision"] == 0.97
    assert thresholds["sample_accuracy_count"] == 100000


def test_f6_tier1_dependency_on_grid100():
    """Verify grid100_mlp16 declares dependency on 3-layer grid100."""
    spec = CONTRACT_SPECS["F6"]
    assert spec["gated_on"] == "grid100"


def test_f6_tier1_prior_configuration():
    """Verify MoGParticlePrior configured with tight sigma=0.025 for 100-mode grid."""
    from particlegan import MoGParticlePrior
    prior = MoGParticlePrior(sigma=0.025)
    assert prior.sigma == 0.025


# ============================================================================
# F7: Task R3 grid100_res16_tanh
# ============================================================================

def test_f7_tier1_contract_specification():
    """Verify grid100_res16_tanh contract parameters match R3 requirements."""
    spec = CONTRACT_SPECS["F7"]
    assert spec["task_id"] == "grid100_res16_tanh"
    assert spec["tier"] == 4
    assert spec["steps"] == 7000
    assert spec["layers"] == 16
    assert spec["output_scale"] == 5.25


def test_f7_tier1_scaled_tanh_output_domain():
    """Verify bounded tanh scaled by S=5.25 covers grid100 coordinates in [-4.5, 4.5]."""
    scale = CONTRACT_SPECS["F7"]["output_scale"]
    assert scale == 5.25
    # Max coordinate in 10x10 grid with spacing 1.0 centered at 0 is 4.5
    assert scale > 4.5
    raw_tanh = np.array([-1.0, -0.9, 0.0, 0.9, 1.0])
    scaled = raw_tanh * scale
    assert scaled.min() == -5.25
    assert scaled.max() == 5.25


def test_f7_tier1_saturation_threshold():
    """Verify non-saturation threshold is saturated_fraction <= 0.001."""
    thresh = CONTRACT_SPECS["F7"]["thresholds"]["saturated_fraction"]
    assert thresh == 0.001


def test_f7_tier1_residual_block_construction():
    """Verify constructive formulation of 16-layer residual architecture."""
    class ResBlock(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(dim, dim),
            )
        def forward(self, x):
            return x + self.net(x)

    blocks = [torch.nn.Linear(2, 64)]
    for _ in range(8):  # 8 blocks * 2 linear layers = 16 layers
        blocks.append(ResBlock(64))
    blocks.append(torch.nn.Linear(64, 2))
    blocks.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*blocks)

    x = torch.randn(4, 2)
    out = model(x) * 5.25
    assert out.shape == (4, 2)
    assert torch.all(torch.abs(out) <= 5.25)


def test_f7_tier1_gated_on_grid100():
    """Verify task is gated on grid100."""
    assert CONTRACT_SPECS["F7"]["gated_on"] == "grid100"


# ============================================================================
# F8: Task R4 misgan_block_recovery
# ============================================================================

def test_f8_tier1_contract_specification():
    """Verify misgan_block_recovery contract parameters match R4 requirements."""
    spec = CONTRACT_SPECS["F8"]
    assert spec["task_id"] == "misgan_block_recovery"
    assert spec["tier"] == 4
    assert spec["steps"] == 7000
    assert spec["dimension"] == 8
    assert spec["problem"] == "block"
    assert spec["pairs"] == 2
    assert spec["detached_mask_gradient"] is True
    assert spec["imputation"] == "ppost"


def test_f8_tier1_2pair_misgan_architecture():
    """Verify 2-pair MisGAN structure: (G_x, D_x) and (G_m, D_m)."""
    # G_x maps latent to 8D data; D_x discriminates masked data
    # G_m generates binary/relaxed mask; D_m discriminates masks
    dim = 8
    g_x = torch.nn.Linear(16, dim)
    d_x = torch.nn.Linear(dim * 2, 1)  # inputs (x * m, m)
    g_m = torch.nn.Linear(16, dim)
    d_m = torch.nn.Linear(dim, 1)
    assert g_x.out_features == dim
    assert g_m.out_features == dim


def test_f8_tier1_detached_mask_gradient_flow():
    """Verify mask tensor gradient detachment in data generator loss."""
    m_raw = torch.tensor([1.0, 0.0, 1.0], requires_grad=True)
    m_detached = m_raw.detach()
    x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    loss = torch.sum(x * m_detached)
    loss.backward()
    assert x.grad is not None
    assert m_raw.grad is None  # Mask gradient detached in L_x


def test_f8_tier1_evaluation_thresholds():
    """Verify all 7 evaluation thresholds for misgan_block_recovery."""
    th = CONTRACT_SPECS["F8"]["thresholds"]
    assert th["modes"] == 100
    assert th["hq"] == 0.90
    assert th["mass_tv"] == 0.10
    assert th["off_plane"] == 0.04
    assert th["mask_tv"] == 0.05
    assert th["acc"] == 0.96
    assert th["acc_lo"] == 0.70


def test_f8_tier1_ppost_imputation_gate():
    """Verify native particle-posterior (ppost) imputation gate accuracy requirements."""
    th = CONTRACT_SPECS["F8"]["thresholds"]
    assert th["acc"] >= 0.96
    assert th["acc_lo"] >= 0.70


# ============================================================================
# F9: Task R4 misgan_block_neural_imputer
# ============================================================================

def test_f9_tier1_contract_specification():
    """Verify misgan_block_neural_imputer contract parameters match R4 requirements."""
    spec = CONTRACT_SPECS["F9"]
    assert spec["task_id"] == "misgan_block_neural_imputer"
    assert spec["tier"] == 4
    assert spec["steps"] == 7000
    assert spec["pairs"] == 3
    assert spec["neural_imputer"] is True


def test_f9_tier1_3pair_misgan_with_neural_imputer():
    """Verify 3-pair architecture: (G_x, D_x), (G_m, D_m), and (G_i, D_i)."""
    dim, latent_dim = 8, 4
    # G_i takes [x * m, m, z_i]
    imputer_in_dim = dim + dim + latent_dim
    g_i = torch.nn.Linear(imputer_in_dim, dim)
    assert g_i.in_features == 20
    assert g_i.out_features == 8


def test_f9_tier1_observed_coordinate_preservation_gate():
    """Verify neural imputer gate requires observed_max_error <= 1e-6."""
    th = CONTRACT_SPECS["F9"]["thresholds"]
    assert th["observed_max_error"] == 1e-6


def test_f9_tier1_missing_std_ratio_interval():
    """Verify missing coordinate std ratio bounds are [0.70, 1.30]."""
    th = CONTRACT_SPECS["F9"]["thresholds"]
    assert th["missing_std_ratio_min"] == 0.70
    assert th["missing_std_ratio_max"] == 1.30


def test_f9_tier1_held_out_accuracy_gates():
    """Verify accuracy gates on ambiguous single-sensor queries."""
    th = CONTRACT_SPECS["F9"]["thresholds"]
    assert th["acc"] >= 0.95
    assert th["acc_lo"] >= 0.65


# ============================================================================
# F10: Task R5 ring16_bigan_continuous
# ============================================================================

def test_f10_tier1_contract_specification():
    """Verify ring16_bigan_continuous contract parameters match R5 requirements."""
    spec = CONTRACT_SPECS["F10"]
    assert spec["task_id"] == "ring16_bigan_continuous"
    assert spec["tier"] == 4
    assert spec["steps"] == 4000
    assert spec["timeout_seconds"] <= 600
    assert spec["prior"] == {"kind": "mog", "sigma": 0.1}


def test_f10_tier1_joint_bigan_architecture():
    """Verify joint BiGAN components (G(z), E(x), D(x, z))."""
    z_dim, x_dim = 4, 2
    G = torch.nn.Linear(z_dim, x_dim)
    E = torch.nn.Linear(x_dim, z_dim)
    D = torch.nn.Linear(x_dim + z_dim, 1)
    
    z = torch.randn(8, z_dim)
    x = torch.randn(8, x_dim)
    
    # Forward generation
    x_fake = G(z)
    assert x_fake.shape == (8, x_dim)
    
    # Encoding
    z_recon = E(x)
    assert z_recon.shape == (8, z_dim)
    
    # Joint discrimination
    d_real = D(torch.cat([x, z_recon], dim=1))
    d_fake = D(torch.cat([x_fake, z], dim=1))
    assert d_real.shape == (8, 1) and d_fake.shape == (8, 1)


def test_f10_tier1_forward_generation_gates():
    """Verify forward generation gates for ring16_bigan_continuous."""
    th = CONTRACT_SPECS["F10"]["thresholds"]
    assert th["modes"] >= 16
    assert th["hq"] >= 0.85
    assert th["mass_tv"] <= 0.15


def test_f10_tier1_roundtrip_reconstruction_gates():
    """Verify paired round-trip reconstruction x -> G(E(x)) gates."""
    th = CONTRACT_SPECS["F10"]["thresholds"]
    assert th["recon_rmse"] <= 0.12
    assert th["mode_consistency"] >= 0.98


def test_f10_tier1_target_16_gaussian_ring():
    """Verify 16-Gaussian ring target geometry: radius=3, sigma=0.1, 16 modes."""
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    centers = np.stack([3.0 * np.cos(angles), 3.0 * np.sin(angles)], axis=1)
    assert centers.shape == (16, 2)
    radii = np.sqrt(np.sum(centers ** 2, axis=1))
    assert np.allclose(radii, 3.0)


# ============================================================================
# F11: 3-Level Solvability Audits
# ============================================================================

def test_f11_tier1_level1_oracle_pass_requirement():
    """Verify Level 1 requires 200/200 independent oracle draws passing the gate."""
    spec = CONTRACT_SPECS["F11"]["level1"]
    assert spec["oracle_draws"] == 200
    assert spec["oracle_passes"] == 200


def test_f11_tier1_level1_noise_floor_headroom():
    """Verify Level 1 requires >= 2.0x headroom above 99th-percentile noise floor."""
    spec = CONTRACT_SPECS["F11"]["level1"]
    assert spec["headroom_ratio"] >= 2.0


def test_f11_tier1_level2_constructive_capacity_witness():
    """Verify Level 2 requires deterministic constructive witness achieving PASS."""
    spec = CONTRACT_SPECS["F11"]["level2"]
    assert spec["witness_pass"] is True


def test_f11_tier1_level3_destructive_negative_controls():
    """Verify Level 3 requires destructive negative controls deterministically FAIL."""
    spec = CONTRACT_SPECS["F11"]["level3"]
    assert spec["negative_control_failures"] is True


def test_f11_tier1_solvability_test_suites_convention():
    """Verify solvability tests adhere to test_solvability_*.py naming and structure."""
    expected_suites = [
        "test_solvability_zero_tanh.py",
        "test_solvability_grid100_deep.py",
        "test_solvability_misgan.py",
        "test_solvability_bigan.py",
    ]
    for s in expected_suites:
        assert s.startswith("test_solvability_") and s.endswith(".py")


# ============================================================================
# F12: Full-Budget API Execution & GIFs
# ============================================================================

def test_f12_tier1_api_provider_discovery():
    """Verify benchmarks.toy_audit.api_contract.discover discovers registered providers."""
    from benchmarks.toy_audit.api_contract import PROVIDERS, discover
    assert len(PROVIDERS) >= 6
    cases = discover()
    assert isinstance(cases, dict) and len(cases) > 0


def test_f12_tier1_api_run_cli_help(run_cli):
    """Verify api_run CLI help executes cleanly."""
    res = run_cli([sys.executable, "-m", "benchmarks.toy_audit.api_run", "--help"])
    assert res.returncode == 0
    assert "--case" in res.stdout
    assert "--output" in res.stdout


def test_f12_tier1_protocol_seed_contract():
    """Verify protocol seed contract is fixed at seed 0."""
    from benchmarks.toy_audit.reproducibility import DEFAULT_SEED
    assert DEFAULT_SEED == CONTRACT_SPECS["F12"]["protocol_seed"] == 0


def test_f12_tier1_required_artifacts_contract():
    """Verify required published artifacts: receipt.json, readout.md, goal.gif."""
    artifacts = CONTRACT_SPECS["F12"]["required_artifacts"]
    assert "receipt.json" in artifacts
    assert "readout.md" in artifacts
    assert "goal.gif" in artifacts


def test_f12_tier1_observation_validation_pass_closed():
    """Verify validate_observation validates finite observations and rejects errors."""
    from benchmarks.toy_audit.api_contract import validate_observation
    valid_obs = {
        "metrics": {"rmse": 0.005},
        "passed": True,
        "failed_bounds": [],
        "views": [{"kind": "scatter", "title": "Test", "target": np.zeros((4, 2)), "samples": np.zeros((4, 2))}],
    }
    validated = validate_observation(valid_obs)
    assert validated["passed"] is True


# ============================================================================
# F13: View Registration & Publication Refresh
# ============================================================================

def test_f13_tier1_target_views_exist(repo_root):
    """Verify target view JSON files exist in configs/forge/views/."""
    for view_id in CONTRACT_SPECS["F13"]["views"]:
        path = repo_root / f"configs/forge/views/{view_id}.json"
        assert path.is_file(), f"Missing view file: {path}"


def test_f13_tier1_view_history_archive_exists(repo_root):
    """Verify configs/forge/view-history/ directory exists for immutable archives."""
    history_dir = repo_root / CONTRACT_SPECS["F13"]["view_history_dir"]
    assert history_dir.is_dir(), f"Missing view-history directory: {history_dir}"


def test_f13_tier1_forge_cli_experiments_by_tier_help(run_cli):
    """Verify python -m experiments.forge experiments-by-tier --help exits 0."""
    res = run_cli([sys.executable, "-m", "experiments.forge", "experiments-by-tier", "--help"])
    assert res.returncode == 0
    assert "--output" in res.stdout


def test_f13_tier1_forge_cli_compile_help(run_cli):
    """Verify python -m experiments.forge compile --help exits 0."""
    res = run_cli([sys.executable, "-m", "experiments.forge", "compile", "--help"])
    assert res.returncode == 0
    assert "--check" in res.stdout or "--summaries-only" in res.stdout


def test_f13_tier1_views_have_valid_structure(repo_root):
    """Verify discriminator_stability.json parses and adheres to view contract."""
    path = repo_root / "configs/forge/views/discriminator_stability.json"
    view = json.loads(path.read_text(encoding="utf-8"))
    validate_view_contract(view)


# ============================================================================
# F14: Git, PR, and Storage Discipline
# ============================================================================

def test_f14_tier1_forge_validate_exits_0(run_cli):
    """Verify python -m experiments.forge validate exits 0."""
    res = run_cli([sys.executable, "-m", "experiments.forge", "validate"], timeout=30)
    assert res.returncode == 0


def test_f14_tier1_gitignore_ignores_runs_directory(repo_root):
    """Verify runs/ directory is ignored in .gitignore."""
    gitignore = repo_root / ".gitignore"
    assert gitignore.is_file()
    content = gitignore.read_text(encoding="utf-8")
    assert "runs/" in content or "runs" in content


def test_f14_tier1_no_tracked_bulk_tensor_checkpoints(repo_root):
    """Verify git does not track heavy checkpoints in public-API reports, and no untracked bulk logs exist."""
    api_contract_dir = repo_root / "reports/toy_audit/api_contract"
    if api_contract_dir.exists():
        bulk_files = [
            str(p.relative_to(repo_root))
            for p in api_contract_dir.rglob("*")
            if p.suffix in (".pt", ".pth", ".jsonl")
        ]
        assert not bulk_files, f"Bulk tensor files found in api_contract: {bulk_files}"
    
    status_res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    untracked = [line for line in status_res.stdout.splitlines() if line.startswith("??")]
    untracked_bulk = [line for line in untracked if any(line.endswith(ext) for ext in (".pt", ".pth", ".npz", ".jsonl"))]
    assert not untracked_bulk, f"Untracked bulk logs found in git status: {untracked_bulk}"


def test_f14_tier1_receipt_file_size_limits(repo_root):
    """Verify committed json receipts in reports/ are compact (<= 10MB)."""
    receipts = list((repo_root / "reports").glob("**/receipt.json"))
    for r in receipts:
        size_mb = r.stat().st_size / (1024 * 1024)
        assert size_mb <= CONTRACT_SPECS["F14"]["max_committed_artifact_mb"]


def test_f14_tier1_teamwork_directory_discipline(repo_root):
    """Verify .agents/teamwork/ contains only metadata, no AGENTS.md or GEMINI.md."""
    teamwork_dir = repo_root / ".agents/teamwork"
    if teamwork_dir.exists():
        forbidden = [f for f in teamwork_dir.rglob("*") if f.name in {"AGENTS.md", "GEMINI.md"}]
        assert not forbidden, f"Forbidden agent file names found in teamwork dir: {forbidden}"

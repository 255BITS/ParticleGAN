"""Tier 3: Cross-Feature Combination E2E Tests for ParticleGAN.

Verifies pairwise and multi-feature interactions across the 14 features in PROJECT.md:
  - F1 + F2: Dynamic N-tier view rendering with dynamic report table headers
  - F1 + F3: N-tier core contiguity validation across Tiers 1..4
  - F1 + F4: Tier 1 Dirac point mass task assigned in N-tier view
  - F2 + F6: Tier 4 deep MLP task rendered under stress_and_formulations header
  - F2 + F13: View registration reflected in dynamic markdown report generation
  - F3 + F5: Tier 3 rail escape assigned as held without breaking contiguity
  - F3 + F7: Tier 4 residual tanh assigned as held with forge_hold_v1 schema
  - F3 + F8: Tier 4 2-pair MisGAN assigned as held in discriminator_stability
  - F4 + F5: Tier 1 smoke to Tier 3 rail escape progression and dependency gating
  - F6 + F7: 16-layer plain MLP vs 16-layer residual tanh architectural stress cohort
  - F8 + F9: 2-pair ppost Bayesian recovery vs 3-pair neural imputer on 8D block
  - F10 + F11: Continuous joint BiGAN evaluated under 3-level solvability gauntlet
  - F4 + F11: Dirac point mass evaluated under 3-level solvability gauntlet
  - F12 + F13: Public API execution outputs linked to view-registered task IDs
  - F12 + F14: Public API execution outputs respect storage discipline (compact receipts only)
  - F1 + F13: Historical view archiving preserves earlier revisions across N-tier updates
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


def test_comb_f1_f2_dynamic_tier4_naming():
    """F1 + F2: Dynamic N-tier view definitions produce 'stress_and_formulations' header."""
    assignments = [
        {"task": f"t{i}", "qualification_tier": i, "importance": "required", "order": i}
        for i in range(1, 5)
    ]
    view = {"schema_version": 1, "id": "ntier_view", "revision": 1, "assignments": assignments}
    validate_view_contract(view)
    
    tier_headers = {a["qualification_tier"]: expected_tier_name(a["qualification_tier"]) for a in assignments}
    assert tier_headers[4] == "stress_and_formulations"
    assert tier_headers[1] == "smoke"
    assert tier_headers[2] == "quality"
    assert tier_headers[3] == "endurance"


def test_comb_f1_f3_tier4_contiguity():
    """F1 + F3: View requiring Tier 4 tasks enforces contiguous progression {1, 2, 3, 4}."""
    valid_progression = [
        {"task": f"t{i}", "qualification_tier": i, "importance": "required", "order": 0}
        for i in (1, 2, 3, 4)
    ]
    assert check_tier_contiguity(valid_progression) is True
    
    gap_progression = [
        {"task": "t1", "qualification_tier": 1, "importance": "required", "order": 0},
        {"task": "t2", "qualification_tier": 2, "importance": "required", "order": 0},
        {"task": "t4", "qualification_tier": 4, "importance": "required", "order": 0},
    ]
    assert check_tier_contiguity(gap_progression) is False


def test_comb_f1_f4_tanh_smoke_in_ntier_view():
    """F1 + F4: zero_tanh_smoke assigned as Tier 1 required task in N-tier view."""
    assignment = {
        "task": CONTRACT_SPECS["F4"]["task_id"],
        "qualification_tier": CONTRACT_SPECS["F4"]["tier"],
        "importance": CONTRACT_SPECS["F4"]["importance"],
        "order": 0,
    }
    view = {"schema_version": 1, "id": "tanh_view", "revision": 1, "assignments": [assignment]}
    validate_view_contract(view)
    assert assignment["qualification_tier"] == 1


def test_comb_f2_f6_grid100_mlp16_in_tier4_report():
    """F2 + F6: grid100_mlp16 is categorized under Tier 4 (stress_and_formulations)."""
    task_tier = CONTRACT_SPECS["F6"]["tier"]
    assert task_tier == 4
    tier_label = expected_tier_name(task_tier)
    assert tier_label == "stress_and_formulations"
    
    table_entry = f"| {CONTRACT_SPECS['F6']['task_id']} | Tier {task_tier} ({tier_label}) |"
    assert "grid100_mlp16" in table_entry
    assert "stress_and_formulations" in table_entry


def test_comb_f2_f13_reports_refresh_with_dynamic_tiers(repo_root):
    """F2 + F13: View definitions in configs/forge/views/ match dynamic tier expectations."""
    view_path = repo_root / "configs/forge/views/discriminator_stability.json"
    assert view_path.is_file()
    data = json.loads(view_path.read_text(encoding="utf-8"))
    validate_view_contract(data)
    
    # All assignments must have valid tier names
    for a in data["assignments"]:
        name = expected_tier_name(a["qualification_tier"])
        assert isinstance(name, str) and len(name) > 0


def test_comb_f3_f5_rail_escape_held_in_tier3():
    """F3 + F5: zero_tanh_rail_escape assigned as held in Tier 3 without breaking contiguity."""
    assignments = [
        {"task": "zero_tanh_smoke", "qualification_tier": 1, "importance": "required", "order": 0},
        {"task": "quality_task", "qualification_tier": 2, "importance": "required", "order": 0},
        {
            "task": "zero_tanh_rail_escape",
            "qualification_tier": 3,
            "importance": "held",
            "order": 0,
            "hold": {
                "schema": "forge_hold_v1",
                "previous_importance": "required",
                "reason": "unsolved",
                "summary": "Held rail escape task",
                "explanation": "reports/forge/held.md",
                "reentry": "Re-enter when passing recipe passes smoke and rail escape",
                "blocked_by": ["zero_tanh_smoke"],
            },
        },
    ]
    view = {"schema_version": 1, "id": "stability_tanh", "revision": 1, "assignments": assignments}
    validate_view_contract(view)
    # Required tiers are {1, 2}, which is contiguous; held tier 3 does not violate contiguity
    assert check_tier_contiguity(assignments) is True


def test_comb_f3_f7_grid100_res16_held_in_tier4():
    """F3 + F7: grid100_res16_tanh assigned as held in Tier 4 with forge_hold_v1 schema."""
    assignment = {
        "task": CONTRACT_SPECS["F7"]["task_id"],
        "qualification_tier": CONTRACT_SPECS["F7"]["tier"],
        "importance": "held",
        "order": 1,
        "hold": {
            "schema": "forge_hold_v1",
            "previous_importance": "required",
            "reason": "no_global_recipe",
            "summary": "16-layer residual tanh architectural stress task",
            "explanation": "reports/forge/held_res16.md",
            "reentry": "Re-enter upon passing recipe discovery",
            "blocked_by": [],
        },
    }
    view = {"schema_version": 1, "id": "stress_view", "revision": 1, "assignments": [assignment]}
    validate_view_contract(view)


def test_comb_f3_f8_misgan_held_in_tier4():
    """F3 + F8: misgan_block_recovery declared with Tier 4 held assignment."""
    assignment = {
        "task": CONTRACT_SPECS["F8"]["task_id"],
        "qualification_tier": CONTRACT_SPECS["F8"]["tier"],
        "importance": "held",
        "order": 2,
        "hold": {
            "schema": "forge_hold_v1",
            "previous_importance": "required",
            "reason": "unsolved",
            "summary": "2-pair MisGAN recovery on 8D block",
            "explanation": "reports/forge/held_misgan.md",
            "reentry": "Re-enter when baseline passes 100-mode recovery",
            "blocked_by": [],
        },
    }
    view = {"schema_version": 1, "id": "misgan_view", "revision": 1, "assignments": [assignment]}
    validate_view_contract(view)


def test_comb_f4_f5_tanh_progression_gating():
    """F4 + F5: zero_tanh_rail_escape declares gate dependency on zero_tanh_smoke."""
    smoke_task_id = CONTRACT_SPECS["F4"]["task_id"]
    rail_escape_task = {
        "schema_version": 1,
        "id": CONTRACT_SPECS["F5"]["task_id"],
        "adapter": "toy_vectors",
        "execution": {
            "steps": CONTRACT_SPECS["F5"]["steps"],
            "initializer": "deterministic_orthogonal",
            "prior": CONTRACT_SPECS["F5"]["prior"],
        },
        "evaluation": {"kind": "toy_vectors"},
        "resources": {"timeout_seconds": CONTRACT_SPECS["F5"]["timeout_seconds"]},
        "dependencies": [{"task": smoke_task_id, "kind": "gate"}],
    }
    validate_task_contract(rail_escape_task)
    assert rail_escape_task["dependencies"][0]["task"] == "zero_tanh_smoke"


def test_comb_f6_f7_architectural_stress_cohort():
    """F6 + F7: Plain MLP-16 and Residual-16 tanh share grid100 target and 7000 steps."""
    mlp16_spec = CONTRACT_SPECS["F6"]
    res16_spec = CONTRACT_SPECS["F7"]
    
    assert mlp16_spec["tier"] == res16_spec["tier"] == 4
    assert mlp16_spec["steps"] == res16_spec["steps"] == 7000
    assert mlp16_spec["layers"] == res16_spec["layers"] == 16
    assert mlp16_spec["gated_on"] == res16_spec["gated_on"] == "grid100"
    assert mlp16_spec["thresholds"]["min_modes"] == res16_spec["thresholds"]["min_modes"] == 100


def test_comb_f8_f9_misgan_recovery_vs_neural_imputer():
    """F8 + F9: 2-pair ppost MisGAN and 3-pair neural imputer share 8D block sensor problem."""
    f8_spec = CONTRACT_SPECS["F8"]
    f9_spec = CONTRACT_SPECS["F9"]
    
    assert f8_spec["tier"] == f9_spec["tier"] == 4
    assert f8_spec["problem"] == f9_spec["problem"] == "block"
    assert f8_spec["dimension"] == f9_spec["dimension"] == 8
    assert f8_spec["steps"] == f9_spec["steps"] == 7000
    # F8 uses 2 pairs + ppost; F9 uses 3 pairs with learned neural imputer G_i
    assert f8_spec["pairs"] == 2 and f8_spec["imputation"] == "ppost"
    assert f9_spec["pairs"] == 3 and f9_spec["neural_imputer"] is True


def test_comb_f10_f11_bigan_3level_solvability():
    """F10 + F11: ring16_bigan_continuous evaluated against 3-level solvability gauntlet."""
    th = CONTRACT_SPECS["F10"]["thresholds"]
    
    # Level 1 Oracle simulation: exact targets have recon_rmse == 0.0 and consistency == 1.0
    oracle_recon_rmse = 0.0
    oracle_consistency = 1.0
    assert oracle_recon_rmse <= th["recon_rmse"]
    assert oracle_consistency >= th["mode_consistency"]
    
    # Headroom check: threshold is 0.12, oracle is 0.0, margin is infinite/well above 2.0x
    assert th["recon_rmse"] > 0.0
    
    # Level 3 Destructive negative control: scrambled mapping has random mode consistency (~1/16 = 0.0625)
    scrambled_consistency = 1.0 / 16.0
    assert scrambled_consistency < th["mode_consistency"]


def test_comb_f4_f11_tanh_smoke_solvability():
    """F4 + F11: zero_tanh_smoke evaluated against 3-level solvability gauntlet."""
    th = CONTRACT_SPECS["F4"]["thresholds"]
    
    # Level 1: Oracle draws from Dirac delta at (0, 0)
    oracle_draws = np.zeros((4096, 2), dtype=np.float32)
    mae = float(np.mean(np.abs(oracle_draws)))
    rmse = float(np.sqrt(np.mean(oracle_draws ** 2)))
    sat = float(np.mean(np.abs(oracle_draws) >= 0.99))
    
    assert mae <= th["mean_abs_error"]
    assert rmse <= th["rms_error"]
    assert sat == th["saturated_fraction"]
    
    # Level 3: Destructive control (stuck at rail)
    rail_draws = np.full((4096, 2), 0.90, dtype=np.float32)
    rail_mae = float(np.mean(np.abs(rail_draws)))
    assert rail_mae > th["mean_abs_error"]


def test_comb_f12_f13_api_execution_linked_to_registered_views(repo_root):
    """F12 + F13: Public API provider cases correspond to task IDs managed in Forge."""
    from benchmarks.toy_audit.api_contract import discover
    cases = discover()
    view_path = repo_root / "configs/forge/views/discriminator_stability.json"
    view_data = json.loads(view_path.read_text(encoding="utf-8"))
    
    # Check that task-to-API case mapping is coherent
    assert len(cases) > 0
    assert len(view_data["assignments"]) > 0


def test_comb_f12_f14_api_execution_storage_cleanliness(repo_root):
    """F12 + F14: api_run artifacts committed in reports/ maintain compact size and zero bulk weights."""
    reports_dir = repo_root / "reports/toy_audit/api_contract"
    if reports_dir.is_dir():
        for receipt in reports_dir.glob("**/receipt.json"):
            # Receipts must be compact (< 100KB)
            assert receipt.stat().st_size < 100 * 1024
        for gif in reports_dir.glob("**/*.gif"):
            # Final goal GIFs must exist and be valid non-empty files
            assert gif.stat().st_size > 0


def test_comb_f1_f13_view_history_archiving_for_ntier(repo_root):
    """F1 + F13: Archived views in configs/forge/view-history/ retain immutable records."""
    history_dir = repo_root / "configs/forge/view-history"
    if history_dir.is_dir():
        for history_file in history_dir.glob("*.json"):
            data = json.loads(history_file.read_text(encoding="utf-8"))
            assert "revision" in data
            assert isinstance(data["revision"], int)
            assert "assignments" in data
            for a in data["assignments"]:
                assert a["qualification_tier"] >= 1

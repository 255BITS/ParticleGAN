"""Shared test configuration, fixtures, and contract validators for ParticleGAN E2E suite."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Authoritative interface contracts derived from ORIGINAL_REQUEST.md and PROJECT.md
CONTRACT_SPECS: Dict[str, Any] = {
    "F1": {
        "name": "N-Tier Core Infrastructure",
        "min_tier": 1,
        "supported_tiers": [1, 2, 3, 4, 5, 10],
    },
    "F2": {
        "name": "Dynamic Tier Naming & Reports",
        "tier_names": {
            1: "smoke",
            2: "quality",
            3: "endurance",
            4: "stress_and_formulations",
        },
        "fallback_template": "tier-{n}",
    },
    "F3": {
        "name": "View Importance & Tier Contiguity",
        "valid_importances": {"required", "ranking", "held", "diagnostic"},
        "hold_schema": "forge_hold_v1",
        "hold_reasons": {"no_global_recipe", "unsolved", "held_prerequisite"},
        "removal_schema": "forge_count_removal_v1",
    },
    "F4": {
        "task_id": "zero_tanh_smoke",
        "tier": 1,
        "importance": "required",
        "steps": 1000,
        "timeout_seconds": 120,
        "prior": {"kind": "mog", "sigma": 0.1},
        "target": "Dirac point mass at (0.0, 0.0)",
        "output_head": "nn.Tanh()",
        "eval_samples": 4096,
        "thresholds": {
            "mean_abs_error": 0.005,
            "rms_error": 0.01,
            "p99_abs_error": 0.025,
            "saturated_fraction": 0.0,
        },
    },
    "F5": {
        "task_id": "zero_tanh_rail_escape",
        "tier": 3,
        "importance": "held",
        "gated_on": "zero_tanh_smoke",
        "steps": 2000,
        "timeout_seconds": 300,
        "prior": {"kind": "mog", "sigma": 0.1},
        "target_start": (0.90, -0.90),
        "target_shift_step": 1001,
        "target_end": (0.0, 0.0),
        "settling_window": 400,
    },
    "F6": {
        "task_id": "grid100_mlp16",
        "tier": 4,
        "importance": "held",
        "gated_on": "grid100",
        "steps": 7000,
        "layers": 16,
        "width": 128,
        "activation": "LeakyReLU(0.2)",
        "prior": {"kind": "mog", "sigma": 0.025},
        "sampling": "clean live",
        "thresholds": {
            "min_modes": 100,
            "min_precision": 0.97,
            "sample_accuracy_count": 100000,
        },
    },
    "F7": {
        "task_id": "grid100_res16_tanh",
        "tier": 4,
        "importance": "held",
        "gated_on": "grid100",
        "steps": 7000,
        "layers": 16,
        "architecture": "residual MLP",
        "output_scale": 5.25,
        "thresholds": {
            "min_modes": 100,
            "min_precision": 0.97,
            "saturated_fraction": 0.001,
        },
    },
    "F8": {
        "task_id": "misgan_block_recovery",
        "tier": 4,
        "importance": "held",
        "steps": 7000,
        "problem": "block",
        "dimension": 8,
        "pairs": 2,
        "detached_mask_gradient": True,
        "imputation": "ppost",
        "thresholds": {
            "modes": 100,
            "hq": 0.90,
            "mass_tv": 0.10,
            "off_plane": 0.04,
            "mask_tv": 0.05,
            "acc": 0.96,
            "acc_lo": 0.70,
        },
    },
    "F9": {
        "task_id": "misgan_block_neural_imputer",
        "tier": 4,
        "importance": "held",
        "steps": 7000,
        "problem": "block",
        "dimension": 8,
        "pairs": 3,
        "neural_imputer": True,
        "thresholds": {
            "observed_max_error": 1e-6,
            "acc": 0.95,
            "acc_lo": 0.65,
            "missing_std_ratio_min": 0.70,
            "missing_std_ratio_max": 1.30,
        },
    },
    "F10": {
        "task_id": "ring16_bigan_continuous",
        "tier": 4,
        "importance": "held",
        "steps": 4000,
        "timeout_seconds": 600,
        "prior": {"kind": "mog", "sigma": 0.1},
        "target": "16-Gaussian ring",
        "thresholds": {
            "modes": 16,
            "hq": 0.85,
            "mass_tv": 0.15,
            "recon_rmse": 0.12,
            "mode_consistency": 0.98,
        },
    },
    "F11": {
        "name": "3-Level Solvability Audits",
        "level1": {"oracle_draws": 200, "oracle_passes": 200, "headroom_ratio": 2.0},
        "level2": {"witness_pass": True},
        "level3": {"negative_control_failures": True},
    },
    "F12": {
        "name": "Full-Budget API Execution & GIFs",
        "protocol_seed": 0,
        "required_artifacts": ["receipt.json", "readout.md", "goal.gif"],
    },
    "F13": {
        "name": "View Registration & Publication Refresh",
        "views": ["discriminator_stability", "formulation_comparison"],
        "target_revision": 13,
        "view_history_dir": "configs/forge/view-history",
    },
    "F14": {
        "name": "Git, PR, and Storage Discipline",
        "prohibited_tracked_extensions": [".pt", ".pth", ".npz", ".jsonl", ".h5"],
        "max_committed_artifact_mb": 10.0,
    },
}


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the absolute path to ParticleGAN repository root."""
    return ROOT


@pytest.fixture(scope="session")
def contract_specs() -> Dict[str, Any]:
    """Provide the authoritative specification contracts."""
    return CONTRACT_SPECS


@pytest.fixture
def run_cli():
    """Execute a CLI command within the repository root environment."""
    def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        return subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    return _run


def expected_tier_name(tier: int) -> str:
    """Derive expected tier name dynamically following F2 contract."""
    known = CONTRACT_SPECS["F2"]["tier_names"]
    if tier in known:
        return known[tier]
    if isinstance(tier, int) and tier >= 5:
        return f"tier-{tier}"
    raise ValueError(f"Invalid tier index: {tier}")


def validate_task_contract(task_dict: Dict[str, Any]) -> None:
    """Validate that a task definition adheres to Forge task contract schema."""
    assert isinstance(task_dict, dict), "Task must be a dictionary"
    assert "schema_version" in task_dict, "Task missing schema_version"
    assert "id" in task_dict and isinstance(task_dict["id"], str) and task_dict["id"], "Task missing valid id"
    assert "adapter" in task_dict, "Task missing adapter"
    assert "execution" in task_dict, "Task missing execution"
    assert "evaluation" in task_dict, "Task missing evaluation"
    assert "resources" in task_dict, "Task missing resources"
    assert "dependencies" in task_dict and isinstance(task_dict["dependencies"], list), "Task missing dependencies list"
    assert "qualification_tier" not in task_dict, "qualification_tier belongs only in view assignments"


def validate_view_contract(view_dict: Dict[str, Any]) -> None:
    """Validate that a view definition adheres to Forge view contract schema."""
    assert isinstance(view_dict, dict), "View must be a dictionary"
    assert "schema_version" in view_dict, "View missing schema_version"
    assert "id" in view_dict and isinstance(view_dict["id"], str), "View missing id"
    assert "revision" in view_dict and isinstance(view_dict["revision"], int) and view_dict["revision"] >= 1, "View revision must be integer >= 1"
    assert "assignments" in view_dict and isinstance(view_dict["assignments"], list), "View missing assignments"
    
    tasks_seen = set()
    for assignment in view_dict["assignments"]:
        assert "task" in assignment, "Assignment missing task"
        assert assignment["task"] not in tasks_seen, f"Duplicate assignment for {assignment['task']}"
        tasks_seen.add(assignment["task"])
        
        tier = assignment.get("qualification_tier")
        assert isinstance(tier, int) and tier >= 1, f"qualification_tier must be integer >= 1, got {tier}"
        
        importance = assignment.get("importance")
        assert importance in CONTRACT_SPECS["F3"]["valid_importances"], f"Invalid importance {importance}"
        
        if importance == "held":
            assert "hold" in assignment, "Held assignment missing hold metadata"
            hold = assignment["hold"]
            assert hold.get("schema") == CONTRACT_SPECS["F3"]["hold_schema"], "Invalid hold schema"
            assert hold.get("reason") in CONTRACT_SPECS["F3"]["hold_reasons"], f"Invalid hold reason {hold.get('reason')}"


def check_tier_contiguity(assignments: List[Dict[str, Any]]) -> bool:
    """Authoritative reference contiguity check for required tiers."""
    required_tiers = {
        a["qualification_tier"]
        for a in assignments
        if a.get("importance") == "required"
    }
    if not required_tiers:
        return True
    highest = max(required_tiers)
    expected = set(range(1, highest + 1))
    return required_tiers == expected

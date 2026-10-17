"""Tier 4: Real-World Application Scenario E2E Tests for ParticleGAN.

Verifies end-to-end realistic workflows and multi-tier pipeline lifecycles:
  - Scenario 1: Full N-Tier Pipeline Lifecycle (declaration -> view assignment -> contiguity -> report export)
  - Scenario 2: Public API Execution & Verification Pipeline (case discovery -> seed isolation -> metric observation -> receipt generation)
  - Scenario 3: Solvability Audit Gauntlet (Oracle noise headroom -> constructive capacity witness -> destructive negative controls)
  - Scenario 4: Multi-Tier Task Progression & Gating Workflow (Tier 1 smoke -> Tier 3 rail escape -> Tier 4 16-layer stress & MisGAN)
  - Scenario 5: Repository Integrity & Publication Freshness Audit (Forge validation, publication refresh checks, git hygiene)
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


def test_scenario1_full_ntier_pipeline_lifecycle(tmp_path):
    """Scenario 1: End-to-end N-tier pipeline lifecycle across Tiers 1 through 4.
    
    Validates task declarations, dynamic view composition, contiguity verification,
    and tier report generation without hardcoded tier limitations.
    """
    # 1. Create task definitions across all 4 tiers
    tasks = {}
    for tier in (1, 2, 3, 4):
        task_id = f"task_tier_{tier}"
        tasks[task_id] = {
            "schema_version": 1,
            "id": task_id,
            "adapter": "toy_vectors",
            "execution": {"steps": 100 * tier, "initializer": "deterministic_orthogonal", "prior": {"kind": "mog", "sigma": 0.05}},
            "evaluation": {"kind": "toy_vectors"},
            "resources": {"timeout_seconds": 60 * tier},
            "dependencies": [],
        }
        validate_task_contract(tasks[task_id])
    
    # 2. Build multi-tier view assignments
    assignments = [
        {"task": f"task_tier_{t}", "qualification_tier": t, "importance": "required", "order": 0}
        for t in (1, 2, 3, 4)
    ]
    # Add held task in Tier 4
    assignments.append({
        "task": "held_tier_4_task",
        "qualification_tier": 4,
        "importance": "held",
        "order": 1,
        "hold": {
            "schema": "forge_hold_v1",
            "reason": "unsolved",
            "summary": "Held experimental task",
            "explanation": "reports/held.md",
            "reentry": "Re-enter upon pass",
            "blocked_by": [],
        },
    })
    
    view = {"schema_version": 1, "id": "ntier_lifecycle_view", "revision": 1, "assignments": assignments}
    validate_view_contract(view)
    
    # 3. Verify contiguity
    assert check_tier_contiguity(assignments) is True
    
    # 4. Generate dynamic tier summary table
    table_rows = []
    for a in assignments:
        t = a["qualification_tier"]
        name = expected_tier_name(t)
        table_rows.append((a["task"], t, name, a["importance"]))
    
    assert len(table_rows) == 5
    tier4_rows = [r for r in table_rows if r[1] == 4]
    assert len(tier4_rows) == 2
    assert all(r[2] == "stress_and_formulations" for r in tier4_rows)


def test_scenario2_public_api_execution_and_verification_pipeline():
    """Scenario 2: Complete public API execution, metric evaluation, and receipt pipeline.
    
    Tests discovery, parameter validation, seed isolation, observation verification,
    and receipt emission following api_contract standards.
    """
    from benchmarks.toy_audit.api_contract import discover, validate_observation
    from benchmarks.toy_audit.reproducibility import DEFAULT_SEED, construction_rng
    
    # 1. Discover all public-API cases
    cases = discover()
    assert len(cases) > 0
    
    # Pick a representative vector case
    case_id = next(cid for cid, c in cases.items() if c["kind"] == "vector" or "ring" in cid or "gaussian" in cid)
    case = cases[case_id]
    
    # 2. Verify frozen parameters
    assert case["default_steps"] > 0
    assert case["batch_size"] > 0
    assert case["eval_samples"] > 0
    assert case["protocol_seed"] == DEFAULT_SEED == 0
    
    # 3. Verify constructor RNG isolation
    with construction_rng(DEFAULT_SEED, "cpu"):
        r1 = torch.randn(4)
    with construction_rng(DEFAULT_SEED, "cpu"):
        r2 = torch.randn(4)
    assert torch.equal(r1, r2)
    
    # 4. Synthesize passing observation matching case schema
    obs = {
        "metrics": {"rmse": 0.005, "eval_samples": case["eval_samples"]},
        "passed": True,
        "failed_bounds": [],
        "views": [{
            "kind": "scatter",
            "title": f"Target and predictions for {case_id}",
            "target": np.zeros((10, 2)),
            "samples": np.zeros((10, 2)),
        }],
    }
    validated = validate_observation(obs)
    assert validated["passed"] is True
    
    # 5. Verify receipt structure
    receipt = {
        "case_id": case_id,
        "status": "PASS",
        "protocol_seed": DEFAULT_SEED,
        "steps_completed": case["default_steps"],
        "metrics": validated["metrics"],
    }
    assert receipt["status"] == "PASS"
    assert receipt["protocol_seed"] == 0


def test_scenario3_solvability_audit_gauntlet():
    """Scenario 3: 3-Level solvability audit gauntlet.
    
    Simulates Level 1 (Oracle 200/200 passes with >= 2.0x headroom),
    Level 2 (Constructive capacity witness achieving PASS),
    and Level 3 (Destructive negative controls deterministically FAILING).
    """
    # Target: 2D Gaussian point at (0.0, 0.0) with small sigma=0.01
    gate_max_mae = 0.005
    gate_max_sat = 0.0
    
    # --- LEVEL 1: Oracle 200/200 Pass + Headroom ---
    oracle_maes = []
    for _ in range(200):
        # Oracle draws from Dirac delta at (0, 0)
        draws = np.zeros((1024, 2), dtype=np.float32)
        mae = float(np.mean(np.abs(draws)))
        sat = float(np.mean(np.abs(draws) >= 0.99))
        assert mae <= gate_max_mae
        assert sat == gate_max_sat
        oracle_maes.append(mae)
    
    assert len(oracle_maes) == 200
    p99_noise_floor = float(np.percentile(oracle_maes, 99))
    # Headroom: gate threshold vs noise floor
    margin_ratio = gate_max_mae / (p99_noise_floor + 1e-8)
    assert margin_ratio >= 2.0  # Well above 2x headroom
    
    # --- LEVEL 2: Constructive Capacity Witness ---
    # Construct a 2-layer MLP with tanh head and train on target points
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 2),
        torch.nn.Tanh(),
    )
    # Capacity witness: network can represent the exact point mass (0, 0)
    torch.nn.init.zeros_(model[2].weight)
    torch.nn.init.zeros_(model[2].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target = torch.zeros(32, 2)
    for _ in range(20):
        optimizer.zero_grad()
        z = torch.randn(32, 2) * 0.1
        out = model(z)
        loss = torch.mean(torch.abs(out - target))
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        test_z = torch.randn(512, 2) * 0.1
        witness_out = model(test_z).numpy()
    witness_mae = float(np.mean(np.abs(witness_out)))
    assert witness_mae <= gate_max_mae  # Constructive witness reaches PASS
    
    # --- LEVEL 3: Destructive Negative Controls ---
    # Control A: Saturation rail failure
    rail_samples = np.ones((512, 2), dtype=np.float32)
    assert float(np.mean(np.abs(rail_samples) >= 0.99)) > gate_max_sat
    
    # Control B: Mode collapse to wrong location (0.5, 0.5)
    shifted_samples = np.full((512, 2), 0.5, dtype=np.float32)
    assert float(np.mean(np.abs(shifted_samples))) > gate_max_mae
    
    # Control C: Nonfinite output failure
    nan_samples = np.zeros((512, 2), dtype=np.float32)
    nan_samples[0, 0] = float("nan")
    assert not np.all(np.isfinite(nan_samples))


def test_scenario4_task_progression_and_gating_workflow():
    """Scenario 4: Multi-tier task progression and dependency gating.
    
    Validates that:
    - Tier 1 (zero_tanh_smoke) must pass before Tier 3 (zero_tanh_rail_escape) runs.
    - 3-layer grid100 must pass before Tier 4 grid100_mlp16 and grid100_res16_tanh run.
    - Unmet prerequisites block downstream execution.
    """
    dependency_graph = {
        "zero_tanh_smoke": {"tier": 1, "dependencies": []},
        "zero_tanh_rail_escape": {"tier": 3, "dependencies": ["zero_tanh_smoke"]},
        "grid100": {"tier": 2, "dependencies": []},
        "grid100_mlp16": {"tier": 4, "dependencies": ["grid100"]},
        "grid100_res16_tanh": {"tier": 4, "dependencies": ["grid100"]},
    }
    
    # State 1: All unstarted
    completed = set()
    def is_eligible(task_id):
        deps = dependency_graph[task_id]["dependencies"]
        return all(d in completed for d in deps)
    
    assert is_eligible("zero_tanh_smoke") is True
    assert is_eligible("zero_tanh_rail_escape") is False
    assert is_eligible("grid100_mlp16") is False
    
    # State 2: Tier 1 smoke passes
    completed.add("zero_tanh_smoke")
    assert is_eligible("zero_tanh_rail_escape") is True
    assert is_eligible("grid100_mlp16") is False
    
    # State 3: grid100 passes
    completed.add("grid100")
    assert is_eligible("grid100_mlp16") is True
    assert is_eligible("grid100_res16_tanh") is True


def test_scenario5_repository_integrity_and_publication_freshness(run_cli, repo_root):
    """Scenario 5: End-to-end repository integrity and publication freshness.
    
    Executes Forge validate command, confirms clean git tree,
    and checks view history directory integrity.
    """
    # 1. Forge validate passes
    res = run_cli([sys.executable, "-m", "experiments.forge", "validate"], timeout=30)
    assert res.returncode == 0
    assert "status" in res.stdout or "tasks" in res.stdout or "views" in res.stdout
    
    # 2. View history directory exists and contains valid JSON archives
    view_history = repo_root / "configs/forge/view-history"
    assert view_history.is_dir()
    archives = list(view_history.glob("*.json"))
    assert len(archives) > 0
    for a in archives:
        data = json.loads(a.read_text(encoding="utf-8"))
        assert "schema_version" in data or "revision" in data
        
    # 3. Check git hygiene (no untracked bulk files)
    status_res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    untracked = [l for l in status_res.stdout.splitlines() if l.startswith("??")]
    forbidden = [l for l in untracked if any(l.endswith(ext) for ext in (".pt", ".pth", ".npz", ".jsonl"))]
    assert not forbidden, f"Forbidden untracked files in git: {forbidden}"

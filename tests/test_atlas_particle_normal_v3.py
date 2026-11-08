"""Regression tests for atlas-particle-normal-v3 and clean summaries-only Forge compilation."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from experiments.forge import knowledge
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import read_json, validate_idea
from experiments.forge.planning import load_idea
from experiments.forge.studies import load_study


ROOT = Path(__file__).resolve().parents[1]


def test_atlas_particle_normal_v3_idea_and_study_contracts():
    idea = load_idea(ROOT, "atlas-particle-normal-v3")
    validate_idea(idea)
    assert idea["id"] == "atlas-particle-normal-v3"
    assert idea["parent"] == "atlas-particle-normal-v2"
    assert idea["recipe_preset"] == "atlas"
    assert idea["recipe_overrides"] == {
        "birth_death_cells": 256,
        "d_guard_min_steps": 0,
        "d_guard_ratio": 0.0,
        "output_noise_mode": "fixed",
        "output_noise_std": 0.0,
    }
    assert idea["prior_mechanism"] == {
        "kind": "particles",
        "manipulators": [{"kind": "normal_noise", "sigma": 0.0065}],
        "standardize": False,
    }

    study = load_study(ROOT, "atlas-particle-normal-replication-study-v3")
    assert study["candidate"] == "atlas-particle-normal-v3"
    assert study["control"]["candidate_id"] == "atlas-particle-normal-v2"
    assert study["scope"]["view"] == "prior-ablation-full-suite-v1"
    assert study["scope"]["execution_backend"] == "cuda"
    assert study["scope"]["through_tier"] == 1


def test_atlas_particle_normal_v3_cell_threshold_and_prior_noise_bindings():
    idea = load_idea(ROOT, "atlas-particle-normal-v3")

    for vector_task_id in ("vector_unequal_mass", "vector_unequal_width"):
        task = read_json(ROOT / f"configs/forge/tasks/{vector_task_id}.json")
        ctx = task_formulation_context(idea, task, {"seed": 0}, device="cpu", root=ROOT)
        assert ctx.recipe.num_particles == 256
        assert ctx.recipe.birth_death_cells == 256
        assert ctx.prior_config["manipulators"] == [{"kind": "normal_noise", "sigma": 0.0065}]

    img_task = read_json(ROOT / "configs/forge/tasks/img_intensity2.json")
    img_ctx = task_formulation_context(idea, img_task, {"seed": 0}, device="cpu", root=ROOT)
    assert img_ctx.prior_config["manipulators"] == []
    assert img_ctx.recipe.prior_kind == "particles"

    grid_task = read_json(ROOT / "configs/forge/tasks/grid100.json")
    grid_ctx = task_formulation_context(idea, grid_task, {"seed": 0}, device="cpu", root=ROOT)
    assert grid_ctx.recipe.num_particles == 20000
    assert grid_ctx.recipe.birth_death_cells == 256 < grid_ctx.recipe.num_particles


def test_atlas_particle_normal_v3_readout_surpasses_v2_and_bispf_with_full_vector_suite():
    idea = load_idea(ROOT, "atlas-particle-normal-v3")
    view = json.loads((ROOT / "configs/forge/views/prior-ablation-full-suite-v1.json").read_text())
    assert len(view["assignments"]) == 48
    for assignment in view["assignments"]:
        task = json.loads((ROOT / f"configs/forge/tasks/{assignment['task']}.json").read_text())
        ctx = task_formulation_context(idea, task)
        assert ctx.recipe is not None
        assert ctx.prior_config is not None

    v3_readout = read_json(ROOT / "reports/forge/records/readout-efb8bacf93e19fc2b0d7f7bf.json")
    v2_readout = read_json(ROOT / "reports/forge/records/readout-4dc982d111099c9b0f33a18e.json")
    bispf_readout = read_json(ROOT / "reports/forge/records/readout-7b017c8ed900b677ebf6bf55.json")
    v1_readout = read_json(ROOT / "reports/forge/records/readout-9f0d62f87d1ff351cfef7f98.json")

    assert v3_readout["candidate_id"] == "atlas-particle-normal-v3"
    assert v3_readout["study_id"] == "atlas-particle-normal-replication-study-v3"
    assert all(item["valid_receipt"] for item in v3_readout["provenance"]["attempts"])

    decision = v3_readout["decision_outcomes"][0]
    assert decision["binding_errors"] == []
    assert decision["observed"]["prediction"]["satisfied"] is True
    assert decision["observed"]["prediction"]["value"] <= 0.85
    assert decision["observed"]["falsifier"]["satisfied"] is False

    v3_by_task = {row["task_id"]: row["gate_status"] for row in v3_readout["task_results"]}
    v3_counts = Counter(v3_by_task.values())
    v2_counts = Counter(row["gate_status"] for row in v2_readout["task_results"])
    bispf_counts = Counter(row["gate_status"] for row in bispf_readout["task_results"])
    v1_counts = Counter(row["gate_status"] for row in v1_readout["task_results"])

    assert v3_counts["INCOMPLETE"] == 0
    assert v3_counts["PASS"] == 23
    assert v3_counts["PASS"] > v2_counts["PASS"] == 22
    assert v3_counts["PASS"] > bispf_counts["PASS"] == 20
    assert v3_counts["PASS"] > v1_counts["PASS"] == 17

    for expected_pass in (
        "vector_anisotropic",
        "vector_overlap",
        "vector_spiral",
        "vector_two_broad",
        "vector_unequal_mass",
        "vector_unequal_width",
        "img_bars4_residual16",
        "img_blobs4_residual16",
        "img_intensity2",
        "img_stripes2_residual16",
        "clockfree_audit",
        "clockfree_audit_measurement_v1",
        "model_vae_categorical",
        "model_vae_hard",
        "model_ae_reconstruction",
        "zero_tanh_smoke",
        "gaussian1d_smoke",
    ):
        assert v3_by_task[expected_pass] == "PASS", f"Expected {expected_pass} to PASS in v3 readout"


def test_compilation_manifest_excludes_untracked_raw_attempts():
    manifest = read_json(ROOT / "reports/forge/compilation.json")
    assert not any(
        key.startswith("reports/forge/attempts/") for key in manifest["input_hashes"]
    ), "compilation.json must not depend on gitignored reports/forge/attempts/ files"
    status = knowledge.freshness(ROOT)
    assert status["fresh"], status

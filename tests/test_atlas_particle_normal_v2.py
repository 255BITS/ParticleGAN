"""Regression tests for atlas-particle-normal-v2 and full-suite Forge integration fixes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from particlegan import Recipe, get_recipe
from particlegan.continuous import LossSurprise
from experiments.forge.api import task_formulation_context, task_recipe_overrides
from experiments.forge.clockfree import learning_state, run_clockfree, source_audit, state_digest
from experiments.forge.contracts import atomic_json, validate_idea
from experiments.forge.planning import load_idea
from experiments.forge.sources import inspect_source, snapshot_source, source_files
from experiments.forge.studies import load_study
from experiments.forge.views import _adaptation, grade_result


ROOT = Path(__file__).resolve().parents[1]


def test_atlas_particle_normal_v2_idea_and_study_contracts():
    idea = load_idea(ROOT, "atlas-particle-normal-v2")
    validate_idea(idea)
    assert idea["id"] == "atlas-particle-normal-v2"
    assert idea["parent"] == "atlas-particle-normal-v1"
    assert idea["recipe_preset"] == "atlas"
    assert idea["recipe_overrides"] == {
        "d_guard_min_steps": 0,
        "d_guard_ratio": 0.0,
        "output_noise_mode": "fixed",
        "output_noise_std": 0.0,
    }
    assert idea["prior_mechanism"] == {
        "kind": "particles",
        "manipulators": [{"kind": "normal_noise", "sigma": 0.005}],
        "standardize": False,
    }

    study = load_study(ROOT, "atlas-particle-normal-replication-study-v2")
    assert study["candidate"] == "atlas-particle-normal-v2"
    assert study["control"]["candidate_id"] == "atlas-particle-normal-v1"
    assert study["scope"]["view"] == "prior-ablation-full-suite-v1"
    assert study["scope"]["execution_backend"] == "cuda"
    assert study["scope"]["through_tier"] == 1


def test_loss_surprise_equilibrium_separation_thresholds():
    assert LossSurprise.Z_THRESH == 3.5
    assert LossSurprise.Z_CALM == 1.5


def test_worker_snapshot_includes_frozen_task_configs_and_excludes_mutable_forge_configs(tmp_path):
    files = {str(p.relative_to(ROOT)) for p in source_files(ROOT)}
    assert "configs/forge/tasks/model_ae_reconstruction.json" in files
    assert "configs/forge/tasks/clockfree_audit.json" in files
    assert not any(path.startswith("configs/forge/ideas/") for path in files)
    assert not any(path.startswith("configs/forge/studies/") for path in files)
    assert not any(path.startswith("configs/forge/views/") for path in files)

    manifest = inspect_source(ROOT)
    snap = snapshot_source(ROOT, tmp_path / "queue", manifest)
    assert (snap / "configs/forge/tasks/model_ae_reconstruction.json").is_file()
    assert not (snap / "configs/forge/ideas").exists()


def test_word_joint_clean_live_public_components_binding():
    idea = load_idea(ROOT, "atlas-particle-normal-v2")
    for task_id in ("five_word_joint_smoke", "five_word_joint_hold"):
        task = json.loads((ROOT / f"configs/forge/tasks/{task_id}.json").read_text())
        overrides = task_recipe_overrides(idea, task)
        assert overrides["continuous_policy"] is None
        assert overrides["lr_control"] == "mobility"
        assert overrides["serve_average"] == 0.0
        assert overrides["particle_birth_death"] is False
        assert overrides["row_evidence_gate"] is False
        assert overrides["total_steps"] == 20000

        ctx = task_formulation_context(idea, task, {"seed": 0}, device="cpu", root=ROOT)
        assert ctx.recipe.prior_kind == "particles"
        assert ctx.recipe.num_particles == 5
        assert ctx.recipe.z_dim == 2
        assert ctx.recipe.batch_size == 256
        assert ctx.recipe.serve_average == 0.0
        assert ctx.recipe.continuous_policy is None
        receipt = ctx.receipt()
        assert receipt["field_ownership"]["task_contract"]["prior"]["code_path"] == "ParticlePrior"


def test_dv12_clockfree_audit_passes_and_blocks_scheduled_guards(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        idea = load_idea(ROOT, "atlas-particle-normal-v2")
        task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
        request = {"candidate": idea, "protocol": {"seed": 0}}
        raw = run_clockfree(request, task, tmp_path / "pass", "cpu")
        grade = grade_result(task, raw)
        assert grade["status"] == "PASS", grade
        assert grade["metrics"]["parity_comparisons"] == 4
        assert raw["evidence"]["source_audit"]["unexplained_clock_dependencies"] == []

        # Scheduled critic guard must remain blocked
        guarded = deepcopy(idea)
        guarded["recipe_overrides"]["d_guard_ratio"] = 2.0
        guarded["recipe_overrides"]["d_guard_min_steps"] = 200
        raw_guarded = run_clockfree({"candidate": guarded, "protocol": {"seed": 0}}, task, tmp_path / "guarded", "cpu")
        assert grade_result(task, raw_guarded)["status"] == "BLOCKED"

        # Scheduled beta2_end or non-stationarity continuous policy must be flagged by source_audit
        dv12_dict = get_recipe("atlas", **idea["recipe_overrides"]).to_dict()
        assert source_audit({**dv12_dict, "beta2_end": 0.9}, {})["unexplained_clock_dependencies"] != []
        assert source_audit({**dv12_dict, "total_steps": 1000}, {})["unexplained_clock_dependencies"] != []
        assert source_audit({**dv12_dict, "lr_control": "mobility"}, {})["unexplained_clock_dependencies"] != []
    finally:
        torch.set_num_threads(previous)


def test_clockfree_learning_state_sanitizes_diagnostic_step_stamps_only():
    fake_state = {
        "streams": {
            "manifest": {"bindings": {"train": {"family": "prior"}, "eval_s": {"family": "eval"}}},
            "states": {"train": torch.ByteTensor([1, 2, 3]), "eval_s": torch.ByteTensor([9, 9])},
        },
        "trainer": {
            "completed_steps": 104,
            "max_steps": 2000,
            "recipe": {"name": "atlas"},
            "streams": {},
            "lr_settle": [[{"s": 0.5, "last_look": {"step": 104, "stat": 1.25}}]],
            "birth_death": {
                "fill": 256,
                "iso_log": [[104, 2, 1]],
                "last": {"step": 104, "moves": 2, "mean_transport": {"step": 104, "accepted": 1}},
                "paired_average": {"step": 104, "eligible": True, "snapshot": 3},
            },
            "surprise": {"fires": 1, "log": [[104, 2.5]]},
            "reopen_guard": {"excursion": {"step": 104, "network": {"0.0": {"role": "generator", "scale": 0.5}}}},
        },
    }
    offset_state = deepcopy(fake_state)
    offset_state["trainer"]["completed_steps"] = 1104
    offset_state["trainer"]["lr_settle"][0][0]["last_look"]["step"] = 1104
    offset_state["trainer"]["birth_death"]["iso_log"][0][0] = 1104
    offset_state["trainer"]["birth_death"]["last"]["step"] = 1104
    offset_state["trainer"]["birth_death"]["last"]["mean_transport"]["step"] = 1104
    offset_state["trainer"]["birth_death"]["paired_average"]["step"] = 1104
    offset_state["trainer"]["surprise"]["log"][0][0] = 1104
    offset_state["trainer"]["reopen_guard"]["excursion"]["step"] = 1104

    assert state_digest(learning_state(fake_state)) == state_digest(learning_state(offset_state))
    # Changing actual training statistics still changes learning_state
    mutated = deepcopy(offset_state)
    mutated["trainer"]["birth_death"]["last"]["moves"] = 3
    assert state_digest(learning_state(fake_state)) != state_digest(learning_state(mutated))


def test_paired_adaptation_allows_none_noise_horizon_only_for_schedule_free_recipe(tmp_path, monkeypatch):
    import experiments.forge.adaptation as adaptation

    monkeypatch.setattr(adaptation, "verify_pair_artifacts", lambda evidence: None)
    task = json.loads((ROOT / "configs/forge/tasks/target_shift_recovery.json").read_text())
    steps = task["evaluation"]["steps"]
    shift_step = task["evaluation"]["shift_step"]
    curve = [{"step": s, "modes": 8, "hq": 1.0} for s in range(10, steps + 1, 10)]
    frozen_curve = [
        {"step": s, "modes": 8 if s <= shift_step else 0, "hq": 1.0 if s <= shift_step else 0.0}
        for s in range(10, steps + 1, 10)
    ]

    def make_run(points, frozen=False):
        return {
            "config_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "runtime": "cuda",
            "mode": "public",
            "noise_horizon": None,
            "diagnostic_every": 10,
            "dense_after": None,
            "dense_until": None,
            "steps": steps,
            "shift_step": shift_step,
            "freeze_after_shift": frozen,
            "diagnostic": points,
            "shift": {"kind": "translate"},
            "shift_pair": {"optimizer_at_shift": [{"updates": shift_step}]},
            "optimizer_final": [{"updates": shift_step if frozen else steps}],
        }

    evidence = {
        "active": make_run(curve, frozen=False),
        "frozen": make_run(frozen_curve, frozen=True),
        "artifact_root": str(tmp_path),
        "artifact_manifest": {"files": {}},
    }

    # Schedule-free recipe (total_steps: None) -> should PASS on noise_horizon=None
    atomic_json(tmp_path / "config.json", {"recipe": {"total_steps": None}})
    verdict_free = _adaptation(task, evidence)
    assert verdict_free["status"] == "PASS", verdict_free

    # Horizon-bound recipe (total_steps: 3600) with noise_horizon=None -> MUST be INCOMPLETE
    atomic_json(tmp_path / "config.json", {"recipe": {"total_steps": 3600}})
    verdict_bound = _adaptation(task, evidence)
    assert verdict_bound["status"] == "INCOMPLETE"


def test_atlas_particle_normal_v2_all_48_tasks_unblocked_and_readout_beats_bispf():
    from collections import Counter

    idea = load_idea(ROOT, "atlas-particle-normal-v2")
    view = json.loads((ROOT / "configs/forge/views/prior-ablation-full-suite-v1.json").read_text())
    assert len(view["assignments"]) == 48
    for assignment in view["assignments"]:
        task = json.loads((ROOT / f"configs/forge/tasks/{assignment['task']}.json").read_text())
        ctx = task_formulation_context(idea, task)
        assert ctx.recipe is not None
        assert ctx.prior_config is not None

    v2_readout = json.loads((ROOT / "reports/forge/records/readout-e97f5dbf54d7493604c4689f.json").read_text())
    v1_readout = json.loads((ROOT / "reports/forge/records/readout-9f0d62f87d1ff351cfef7f98.json").read_text())
    bispf_readout = json.loads((ROOT / "reports/forge/records/readout-7b017c8ed900b677ebf6bf55.json").read_text())

    assert v2_readout["candidate_id"] == "atlas-particle-normal-v2"
    assert v2_readout["study_id"] == "atlas-particle-normal-replication-study-v2"

    v2_by_task = {row["task_id"]: row["gate_status"] for row in v2_readout["task_results"]}
    v1_counts = Counter(row["gate_status"] for row in v1_readout["task_results"])
    bispf_counts = Counter(row["gate_status"] for row in bispf_readout["task_results"])
    v2_counts = Counter(v2_by_task.values())

    assert v2_counts["INCOMPLETE"] == 0
    assert v2_counts["PASS"] == 22
    assert v2_counts["PASS"] > bispf_counts["PASS"] == 20
    assert v2_counts["PASS"] > v1_counts["PASS"] == 17

    for newly_passing in (
        "clockfree_audit_measurement_v1",
        "clockfree_audit",
        "mode_hold",
        "img_intensity2",
        "model_ae_reconstruction",
    ):
        assert v2_by_task[newly_passing] == "PASS", f"Expected {newly_passing} to PASS in v2 readout"


"""Public adapter execution, independent timing grading and bounded admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge import adapters, scaling
from experiments.forge.api import task_formulation_context, task_policy_blockers
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.sampling import expected_policy
from experiments.forge.trainer_families import current_family_candidates
from experiments.forge.views import grade_result, load_tasks, load_view, qualify, validate_view


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_matrix_preserves_selected_recipes_and_clears_all_family_preflights():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "dcgan_performance")
    assert view["evidence_scope"] == "research_diagnostic"
    assert len(view["assignments"]) == 6
    selected = current_family_candidates(ROOT)
    from experiments.forge.planning import load_idea
    global_bindings = {}
    for choice in selected:
        candidate = load_idea(ROOT, choice["candidate_id"])
        for assignment in view["assignments"]:
            task = tasks[assignment["task"]]
            assert adapters.adapter_preflight(task, candidate, root=ROOT) == []
            context = task_formulation_context(candidate, task)
            recipe = context.recipe.to_dict()
            global_recipe = {key: value for key, value in recipe.items()
                             if key not in {"batch_size", "z_dim", "num_particles", "total_steps"}}
            if choice["family"] in global_bindings:
                assert global_recipe == global_bindings[choice["family"]]
            global_bindings[choice["family"]] = global_recipe
            assert context.prior_config["kind"] == "particle_cloud"
            assert context.prior_config["sigma"] == 0
            assert task["dependencies"] == []
            assert assignment["importance"] == "diagnostic"
    assert len(global_bindings) == 8


def test_policy_exception_requires_explicit_performance_contract():
    from experiments.forge.planning import load_idea
    candidate = load_idea(ROOT, "e22")
    task = load_tasks(ROOT)["dcgan_performance_r32"]
    assert task_policy_blockers(task, candidate) == []
    task["evaluation"]["qualification_input"] = True
    assert task_policy_blockers(task, candidate)


def test_public_cpu_adapter_actual_training_and_independent_trace_grade(tmp_path):
    task = load_tasks(ROOT)["dcgan_performance_cpu_smoke"]
    candidate = {"id": "software-test", "recipe_preset": "bcap_adam",
                 "recipe_overrides": {"lr": .0001, "input_noise_std": 0., "output_noise_std": 0.}}
    request = {"candidate": candidate, "candidate_revision": "software-test",
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    old = torch.get_rng_state().clone()
    raw = adapters.run_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert torch.equal(old, torch.get_rng_state())
    assert raw["status"] == "PASS"
    assert raw["candidate_id"] == candidate["id"]
    assert raw["evidence"]["artifacts"]["updates"]["sha256"] == file_hash(tmp_path / "updates.jsonl")
    assert all(raw["evidence"][key] == value for key, value in expected_policy(task).items())
    assert (tmp_path / "training.gif").is_file()
    checkpoint = torch.load(tmp_path / "state.pt", weights_only=False)
    assert checkpoint["data_cursor"]["completed_updates"] == task["execution"]["steps"]
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS"
    assert grade["qualification_input"] is False
    assert grade["metrics"]["measured_updates"] == 2
    # A status stamp cannot fill absent raw timing evidence.
    broken = deepcopy(raw)
    del broken["evidence"]["artifacts"]["updates"]
    assert grade_result(task, broken)["status"] == "INCOMPLETE"
    # Frozen trace hashes bind a success to its actual finite measured updates.
    with (tmp_path / "updates.jsonl").open("a") as stream:
        stream.write("{}\n")
    assert grade_result(task, raw)["status"] == "INVALID"


def test_registration_is_finite_recipe_pinned_and_task_pinned(tmp_path, monkeypatch):
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "dcgan_performance")
    candidate_path = tmp_path / "configs/forge/ideas/example.json"
    atomic_json(candidate_path, {"id": "example"})
    monkeypatch.setattr(scaling, "load_tasks", lambda root: deepcopy(tasks))
    monkeypatch.setattr(scaling, "load_view", lambda root, name: deepcopy(view))
    monkeypatch.setattr(scaling, "load_idea", lambda root, name: {"id": name})
    monkeypatch.setattr(scaling, "current_family_candidates", lambda root, family_ids: [
        {"family": "example", "label": "Example", "candidate_id": "example"}])
    registered = scaling.register(tmp_path, "bounded-test", execution_backend="cpu")
    assert registered["campaign"]["budget_seconds"] == 1200
    assert registered["campaign"]["candidate_budget_seconds"] == 1200
    assert registered["qualification_input"] is False
    manifest = scaling._load(tmp_path, "bounded-test")
    assert len(manifest["task_contract_sha256"]) == 6
    assert scaling.register(tmp_path, "bounded-test", execution_backend="cpu") == registered
    atomic_json(candidate_path, {"id": "example", "changed": True})
    with pytest.raises(ValueError, match="recipe declaration changed"):
        scaling._load(tmp_path, "bounded-test")


def test_performance_view_never_qualifies_a_recipe():
    tasks = load_tasks(ROOT)
    view = load_view(ROOT, "dcgan_performance")
    result = qualify(view, tasks, [], candidate={})
    assert result["status"] == "DIAGNOSTIC"
    assert result["qualified_tier"] == 0
    assert result["eligible"] is False
    assert result["qualification_input"] is False
    view.pop("evidence_scope")
    for assignment in view["assignments"]:
        assignment["importance"] = "required"
    with pytest.raises(ValueError, match="cannot grant qualification"):
        validate_view(view, tasks)

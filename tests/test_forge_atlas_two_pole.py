"""Focused ordinary-binding regression tests; no qualification training."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from experiments.forge import atlas_two_pole
from experiments.forge.api import (
    CapabilityError, FormulationContext, task_formulation_context,
    task_policy_blockers, task_recipe_overrides,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def task():
    return json.loads((ROOT / atlas_two_pole.TASK_PATH).read_text())


@pytest.fixture
def candidate():
    return dict(recipe_preset="atlas", recipe_overrides={},
                requires_capabilities=["particle_cloud", "a2", "named_rng", "policy_controls", "policy_serving"])


@pytest.fixture
def protocol():
    return json.loads((ROOT / atlas_two_pole.PROTOCOL_PATH).read_text())


def test_ordinary_context_is_model_free_and_preserves_full_atlas_none(task, candidate, protocol, monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata binding constructed a model")
    monkeypatch.setattr(torch.nn.Linear, "__init__", forbidden)
    context = task_formulation_context(candidate, task, protocol, root=ROOT)
    full = asdict(context.recipe)
    atlas_two_pole.validate_effective_recipe(full)
    assert full["total_steps"] is None and full["continuous_policy"] == "dv12"
    assert (full["num_particles"], full["z_dim"], full["batch_size"]) == (12, 1, 12)
    assert context.capabilities()["live_sampling"] is True
    assert context.capabilities()["served_sampling"] is False
    assert context._policy is None and context._trainer is None
    assert context.receipt()["ordinary_component_contract"]["observation"]["owner"] == "live_table_and_live_critic"
    assert not task_policy_blockers(task, candidate)
    from experiments.forge.adapters import adapter_preflight
    assert adapter_preflight(task, candidate, root=ROOT) == []


def test_current_preset_has_identical_source_binding_and_task_owned_resources(task, candidate, protocol):
    binding = atlas_two_pole.resolve_binding(ROOT, candidate, task, protocol)
    overrides = task_recipe_overrides(candidate, task)
    assert overrides == atlas_two_pole.TASK_BINDINGS
    assert atlas_two_pole.digest(binding["recipe"]) == atlas_two_pole.EFFECTIVE_RECIPE_SHA256
    assert binding["source_contract"]["observation"]["observations"] == atlas_two_pole.CLOCKS
    assert binding["source_contract"]["observation"]["final_five"] == [67, 70, 74, 77, 80]
    assert binding["source_contract"]["objective"]["particle_l2"] == .02
    assert binding["source_contract"]["scientific_credit"] is False


@pytest.mark.parametrize("change", [
    {"recipe_overrides": {"lr": .01}},
    {"recipe_overrides": {"total_steps": 80}},
    {"extensions": {"unreviewed": True}},
    {"initializer": "supplied"},
    {"recipe_preset": "e22"},
])
def test_narrow_owner_does_not_unblock_other_configs(task, candidate, change):
    candidate.update(change)
    assert not atlas_two_pole.supports(task, candidate)
    assert task_policy_blockers(task, candidate)


def test_narrow_owner_does_not_unblock_other_tasks_or_gate_edits(task, candidate):
    altered = deepcopy(task)
    altered["evaluation"]["thresholds"][0][2] = .2
    assert not atlas_two_pole.supports(altered, candidate)
    assert task_policy_blockers(altered, candidate)
    ring = json.loads((ROOT / "configs/forge/tasks/ring16_acquisition.json").read_text())
    assert not atlas_two_pole.supports(ring, candidate)
    assert task_policy_blockers(ring, candidate)


def test_bcap_finite_original_behavior_binding_remains_unchanged(task):
    candidate = dict(recipe_preset="bcap", recipe_overrides={})
    assert not atlas_two_pole.supports(task, candidate)
    overrides = task_recipe_overrides(candidate, task)
    assert overrides == {"total_steps": 80}
    context = task_formulation_context(candidate, task, root=ROOT)
    assert context.recipe.total_steps == 80
    assert context.recipe.continuous_policy is None
    assert context.recipe.optimizer_family == "adam"
    assert context._ordinary_two_pole is False


def test_direct_context_cannot_use_task_as_a_blanket_policy_waiver(task):
    with pytest.raises(ValueError, match="complete unchanged current Atlas"):
        FormulationContext(recipe_preset="bcap", policy_task=task,
                           prior=task["execution"]["prior"], execution_path="public_components")


def test_ordinary_declarations_are_in_frozen_catalog_support(task):
    assert atlas_two_pole.supporting_source_paths(task) == (
        "configs/forge/tasks/two_pole.json", "configs/forge/protocols/screening.json")
    other = deepcopy(task)
    other["id"] = "different"
    assert atlas_two_pole.supporting_source_paths(other) == ()


def test_dispatch_reuses_owner_only_for_exact_ordinary_atlas(task, candidate, monkeypatch, tmp_path):
    from experiments.forge.adapters import _dispatch_task
    from experiments.forge import behavior_adapters
    monkeypatch.setattr(atlas_two_pole, "run_behavior", lambda *args: {"owner": "ordered_atlas"})
    monkeypatch.setattr(behavior_adapters, "run_behavior", lambda *args: {"owner": "legacy"})
    request = dict(candidate=candidate, tasks={"two_pole": task})
    assert _dispatch_task(request, {"task_id": "two_pole"}, tmp_path, "cpu") == {"owner": "ordered_atlas"}
    request["candidate"] = dict(recipe_preset="bcap", recipe_overrides={})
    assert _dispatch_task(request, {"task_id": "two_pole"}, tmp_path, "cpu") == {"owner": "legacy"}


def test_source_refusal_precedes_fresh_factory(tmp_path):
    calls = []
    def guard():
        raise ValueError("source refused")
    fresh = atlas_two_pole._OrdinaryFreshOwner(guard, {}, tmp_path)
    with pytest.raises(ValueError, match="source refused"):
        fresh.construct(lambda: calls.append("factory"), atlas_two_pole.owner_initial_receipt)
    assert calls == []


def test_ordinary_shared_preflight_refuses_source_pin_loss(task, candidate, protocol, monkeypatch):
    from experiments.forge.adapters import adapter_preflight
    original = atlas_two_pole._source
    def missing(root, relative):
        if relative == "particlegan/policy.py":
            raise ValueError("source drift: particlegan/policy.py")
        return original(root, relative)
    monkeypatch.setattr(atlas_two_pole, "_source", missing)
    with pytest.raises(ValueError, match="source drift"):
        task_formulation_context(candidate, task, protocol, root=ROOT)
    assert any("source drift" in message for message in adapter_preflight(task, candidate, root=ROOT))


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "trajectory"])
def test_other_original_hosts_remain_blocked(task_id, candidate, protocol):
    other = json.loads((ROOT / "configs/forge/tasks" / (task_id + ".json")).read_text())
    from experiments.forge.adapters import adapter_preflight
    with pytest.raises(CapabilityError):
        task_formulation_context(candidate, other, protocol, root=ROOT)
    assert adapter_preflight(other, candidate, root=ROOT)

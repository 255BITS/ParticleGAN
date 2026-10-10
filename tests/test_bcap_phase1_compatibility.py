"""Bounded software replay, not full-budget task qualification."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest
from experiments.forge.planning import load_idea
from benchmarks.toy_audit.api_images import WordFixture
from test_bcap_integration_core import _batch, _equal, _trainer
from test_host_integration_bcap import winner

ROOT = Path(__file__).resolve().parents[1]
MODES = [("none", 0., 0.), ("direction_blend", 0., 0.), ("none", 1., 0.),
         ("none", 0., 1.), ("direction_blend", 1., 0.), ("direction_blend", 1., 1.)]


def candidate_overrides():
    name = os.environ.get("BCAP_COMPATIBILITY_CANDIDATE")
    return deepcopy(load_idea(ROOT, name).get("recipe_overrides", {})) if name else {}


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("mode,global_weight,local_weight", MODES)
def test_each_enabled_subset_replays_public_trainer_checkpoint_and_streams(mode, global_weight, local_weight):
    # Reconstruct through the public trainer, rather than substituting optimizer
    # tensors. Data and evaluation streams are explicitly included in replay.
    from dataclasses import replace
    from particlegan import GANTrainer
    initial = _trainer(active=False)
    overrides = candidate_overrides()
    overrides.update(constraint_geometry_mode=mode, kinetic_transport_weight=global_weight,
                     kinetic_transport_local_weight=local_weight)
    recipe = replace(initial.recipe, **overrides)
    # GANTrainer owns these exact streams in its checkpoint; use the existing
    # publicly named bindings supplied by the independent scalar fixture.
    from experiments.forge.api import TRAINER_STREAM_BINDINGS
    from experiments.forge.rng import NamedStreams
    streams = NamedStreams(0, device="cpu")
    stream_args = {name: streams.generator(family, component=component, purpose=purpose)
                   for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()}
    trainer = GANTrainer(recipe, initial.G, initial.D, prior=initial.prior, seed=0, **stream_args)
    trainer.step(_batch())
    checkpoint = deepcopy(trainer.state_dict())
    global_before = torch.get_rng_state().clone()
    expected_result = trainer.step(_batch())
    expected = trainer.state_dict()
    restore_initial = _trainer(active=False)
    restored_streams = NamedStreams(0, device="cpu")
    restored_args = {name: restored_streams.generator(family, component=component, purpose=purpose)
                     for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()}
    restored = GANTrainer(recipe, restore_initial.G, restore_initial.D,
                          prior=restore_initial.prior, seed=0, **restored_args)
    restored.load_state_dict(checkpoint)
    result = restored.step(_batch())
    _equal(expected_result, result)
    _equal(expected, restored.state_dict())
    assert torch.equal(global_before, torch.get_rng_state())
    assert ("kinetic_transport" in result) == bool(global_weight)
    assert ("kinetic_transport_local" in result) == bool(local_weight)
    assert hasattr(trainer.opt_g, "bind_protected_losses") == (mode != "none")


def word_context(mode, global_weight, local_weight):
    overrides = dict(winner()["recipe_overrides"], **candidate_overrides())
    overrides.update(num_particles=5, z_dim=2,
        batch_size=256, total_steps=20000, constraint_geometry_mode=mode,
        kinetic_transport_weight=global_weight, kinetic_transport_local_weight=local_weight)
    context = FormulationContext(recipe_preset="bcap", recipe_overrides=overrides,
        prior=dict(kind="particle_cloud", sigma=0., standardize=False, learnable=True,
                   exception_reason="Bounded software replay of the original finite-vocabulary host"),
        seed=0, device="cpu", initializer="deterministic_orthogonal", execution_path="public_components",
        component_transport="output_marginal_v1" if global_weight or local_weight else None)
    fixture = WordFixture(device="cpu", seed=0, recipe_name=None, max_steps=4, components=context)
    return fixture, context


@pytest.mark.parametrize("mode,global_weight,local_weight", MODES)
def test_each_word_subset_replays_joint_ownership_counters_data_and_named_streams(mode, global_weight, local_weight):
    fixture, context = word_context(mode, global_weight, local_weight)
    fixture.step()
    saved, named = deepcopy(fixture.state_dict()), deepcopy(context.streams.state_dict())
    before = torch.get_rng_state().clone()
    expected_results = [fixture.step(), fixture.step()]
    restored, restored_context = word_context(mode, global_weight, local_weight)
    restored_context.streams.load_state_dict(named)
    restored.policy.load_state_dict(saved["api_state"])
    restored.data_generator.set_state(saved["data_generator"])
    restored.restore_component_transport(saved.get("component_transport"))
    assert state_digest([restored.step(), restored.step()]) == state_digest(expected_results)
    assert state_digest(restored.state_dict()) == state_digest(fixture.state_dict())
    assert state_digest(restored_context.streams.state_dict()) == state_digest(context.streams.state_dict())
    assert torch.equal(before, torch.get_rng_state())
    assert hasattr(fixture.opt_g, "bind_protected_losses") == (mode != "none")
    if global_weight or local_weight:
        assert fixture.transport.active_calls == 3
    else:
        assert fixture.transport is None and "component_transport" not in saved


def phase2_module():
    spec = importlib.util.spec_from_file_location("bcap_phase2_test", ROOT / "reports/forge/bcap-three-phase/phase2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase2_reserves_original_six_gates_and_optional_audit():
    module = phase2_module()
    _, tasks = module.ordinary_contract(ROOT)
    assert len(tasks) == 7
    assert sum(task["resources"]["timeout_seconds"] for task in tasks.values()) == 2520
    assert sum(tasks[name]["resources"]["timeout_seconds"] for name in module.TIMEOUTS) == 2220
    for name in module.TIMEOUTS:
        changed = deepcopy(tasks[name])
        changed["resources"]["timeout_seconds"] += 1
        assert module.original_question(changed) != module.original_question(tasks[name])
        changed = deepcopy(tasks[name])
        changed["evaluation"]["thresholds"][0][2] += 1
        assert module.original_question(changed) != module.original_question(tasks[name])


def test_new_candidate_cannot_embed_task_or_study_settings():
    module = phase2_module()
    arm = dict(parent="bcap-develop-integration-combined-v1", candidate_id="software-v3-test",
        changed_factors=["Declared global transport coefficient"], mechanism_rationale="Test reusable recipe boundary",
        recipe_overrides=dict(batch_size=1))
    with pytest.raises(ValueError, match="task-owned"):
        module.successor(ROOT, arm)
    arm["recipe_overrides"] = dict(kinetic_transport_weight=.5)
    candidate = module.successor(ROOT, arm)
    assert candidate["schema_version"] == 3
    assert not {"prior", "decision_contract", "study", "goal", "hypothesis", "seed"} & candidate.keys()

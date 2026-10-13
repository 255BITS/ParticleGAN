"""Bounded software routing checks; original tasks and grades remain unchanged."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.api_images import WordFixture
from experiments.forge import energy_adapters
from experiments.forge.adapters import _dispatch_task
from experiments.forge.api import resolve_public_recipe, task_formulation_context, task_policy_blockers
from experiments.forge.contracts import file_hash
from experiments.forge.energy_adapters import EnergyWordFixture
from experiments.forge.views import load_tasks
from particlegan import GANTrainer, Recipe
from particlegan.energy_distance import energy_distance_u
from particlegan.init import deterministic_orthogonal_


ROOT = Path(__file__).resolve().parents[1]
HOSTS = ["gaussian1d_smoke", "ring16_acquisition", "two_pole", "unused_token_hold",
         "ae_gan_hold", "five_word_joint_smoke", "clockfree_audit_measurement_v1"]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def candidate(weight=1.):
    declaration = json.loads((ROOT / "configs/forge/ideas/r1r2-non-saturating-ams-m110-v1.json").read_text())
    declaration["recipe_overrides"]["energy_distance_weight"] = weight
    return declaration


def word_fixture(weight=1.):
    task = load_tasks(ROOT)["five_word_joint_smoke"]
    context = task_formulation_context(candidate(weight), task, {"seed": 0}, device="cpu")
    cls = EnergyWordFixture if weight else WordFixture
    return cls(device="cpu", seed=0, recipe_name=None, max_steps=3, components=context), context


def trainer(weight=1.):
    with torch.random.fork_rng(devices=[]):
        recipe = resolve_public_recipe(candidate(weight)).replace(
            batch_size=4, z_dim=2, num_particles=8, total_steps=4,
            prior_kind="mog", sigma_rel=.025, standardize=False)
        g, d = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        p = recipe.make_prior(sigma=.025)
        for model in (g, d, p):
            deterministic_orthogonal_(model, seed=0)
        return GANTrainer(recipe, g, d, prior=p, seed=0)


def test_pinned_evaluator_files_and_all_seven_task_contexts_stay_supported():
    from experiments.forge.preflight import task_preflight
    tasks = load_tasks(ROOT)
    for name in HOSTS:
        task = deepcopy(tasks[name])
        for path, expected in task["evaluation"].get("sources", {}).items():
            assert file_hash(ROOT / path) == expected
        assert task_preflight(task, candidate(), {"seed": 0}, root=ROOT, tasks=tasks) == []


@pytest.mark.parametrize("corruption", [None, "field", "reference", "owner", "representation", "applicability"])
def test_inactive_metadata_projection_checks_task_ownership_before_omission(corruption):
    from conftest import _check_inactive_ownership
    binding = dict(latent_table_controls=False, representation="direct_sample_coordinates")
    ownership = dict(value=None, owner="hyperparameter", source="task prior-control applicability",
                     status="not_applicable", reference_recipe_value=0.0,
                     representation="direct_sample_coordinates")
    field = "prior_proximal_strength"
    if corruption == "field": field = "energy_distance_weight"
    elif corruption == "reference": ownership["reference_recipe_value"] = .1
    elif corruption == "owner": ownership["owner"] = "task"
    elif corruption == "representation": ownership["representation"] = "latent_prior_locations"
    elif corruption == "applicability": binding["latent_table_controls"] = True
    if corruption:
        with pytest.raises(AssertionError):
            _check_inactive_ownership(field, 0., ownership, binding)
    else:
        _check_inactive_ownership(field, 0., ownership, binding)
        _check_inactive_ownership("energy_distance_weight", 0.,
                                 dict(value=0., source="public Recipe preset k3p", status="effective"), binding)


@pytest.mark.parametrize("mutation", [
    lambda t: t["execution"].pop("transport_consumer"),
    lambda t: t["execution"]["transport_contract"].update(new_forward_passes=True),
    lambda t: t["execution"].update(host="trajectory"),
])
def test_energy_fails_closed_without_actual_declared_output_support(mutation):
    task = deepcopy(load_tasks(ROOT)["two_pole"])
    mutation(task)
    assert task_policy_blockers(task, candidate())


def test_public_trainer_uses_existing_g_panel_once_without_extra_forwards_or_draws(monkeypatch):
    control, active = trainer(0.), trainer(1.)
    counts = {}
    for label, instance in (("control", control), ("active", active)):
        counts[label] = {"G": 0, "D": 0, "sample": 0}
        for key, model in (("G", instance.G), ("D", instance.D)):
            def observe(_module, _inputs, _output, *, name=key, bucket=counts[label]):
                bucket[name] += 1
            model.register_forward_hook(observe)
        sample = instance._sample_training_prior
        def sampled(n, *, original=sample, bucket=counts[label]):
            bucket["sample"] += 1
            return original(n)
        monkeypatch.setattr(instance, "_sample_training_prior", sampled)
    calls = []
    original = Recipe.energy_distance_loss
    def capture(self, fake, real):
        calls.append((fake, real))
        return original(self, fake, real)
    monkeypatch.setattr(Recipe, "energy_distance_loss", capture)
    target = torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])
    target_calls = []
    def generator_target():
        target_calls.append(1)
        return target + .2
    plain = control.step(target, generator_real=generator_target)
    mixed = active.step(target, generator_real=generator_target)
    assert counts["control"] == counts["active"] and counts["active"]["sample"] == 2
    assert len(calls) == 1 and len(target_calls) == 2
    assert torch.equal(calls[0][1], target + .2)
    assert calls[0][0].requires_grad
    assert mixed["energy_distance"] == original(active.recipe, *calls[0]).detach()
    assert "energy_distance" not in plain
    assert torch.equal(control.latent_generator.get_state(), active.latent_generator.get_state())
    assert torch.equal(control.prior_noise_generator.get_state(), active.prior_noise_generator.get_state())


def test_enabled_trainer_checkpoint_resumes_and_cannot_load_disabled_or_changed_weight():
    active = trainer()
    panel = torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])
    active.step(panel)
    saved = active.state_dict()
    expected = active.step(panel)
    restored = trainer()
    restored.load_state_dict(saved)
    got = restored.step(panel)
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            assert torch.equal(expected[key], got[key])
    for weight in (0., .5):
        with pytest.raises(ValueError, match="recipe"):
            trainer(weight).load_state_dict(saved)


@pytest.mark.parametrize("host", ["two_pole", "unused_token_hold", "ae_gan_hold"])
def test_component_dispatch_exercises_energy_once_per_real_update_and_separate_state(tmp_path, host):
    task = deepcopy(load_tasks(ROOT)[host])
    task["execution"]["steps"] = 2
    request = {"candidate": candidate(), "protocol": {"seed": 0}, "tasks": {host: task}}
    before = torch.get_rng_state().clone()
    result = _dispatch_task(request, {"task_id": host}, tmp_path, "cpu")
    assert torch.equal(before, torch.get_rng_state())
    guards = result["evidence"]["guards"]
    assert guards["hooks_exercised"] and guards["component_energy_active_calls"] == 2
    state = result["evidence"]["component_energy"]
    assert state["calls"] == state["active_calls"] == 2 and state["weight"] == 1.
    assert result["applied"]["component_transport"] is None
    assert result["applied"]["energy_binding"]["native_loop"].endswith("run_behavior")
    from experiments.forge.artifacts import verify_artifacts
    checkpoint = result["evidence"]["provenance_checkpoint"]
    verify_artifacts(checkpoint["artifact_root"], checkpoint["artifact_manifest"])
    saved = torch.load(Path(checkpoint["artifact_root"]) / checkpoint["path"], weights_only=True)
    assert saved["component_energy"] == state
    from experiments.forge.views import grade_result
    assert grade_result(load_tasks(ROOT)[host], result)["status"] == "INCOMPLETE"


def test_native_word_step_inherited_preserves_all_model_calls_streams_and_encoder_gradient():
    control, ccontext = word_fixture(0.)
    active, acontext = word_fixture(1.)
    counts = {}
    for label, fixture in (("control", control), ("active", active)):
        counts[label] = {"G": 0, "D": 0, "E": 0}
        for name in counts[label]:
            def observe(_module, _inputs, _output, *, key=name, bucket=counts[label]):
                bucket[key] += 1
            getattr(fixture, name).register_forward_hook(observe)
    control.step()
    active.step()
    assert counts["active"] == counts["control"]
    assert active.transport is None and active.energy_consumer.active_calls == 1
    assert "component_transport" not in active.state_dict()
    assert active.state_dict()["component_energy"]["calls"] == 1
    assert "component_energy" not in control.state_dict()
    for a, b in zip(active.E.parameters(), control.E.parameters()):
        assert torch.equal(a.grad, b.grad)
    assert ccontext.streams.state_dict()["manifest"] == acontext.streams.state_dict()["manifest"]
    for key, value in ccontext.streams.state_dict()["states"].items():
        assert torch.equal(value, acontext.streams.state_dict()["states"][key])
    assert EnergyWordFixture.observe is WordFixture.observe


def test_active_word_signed_state_requires_correct_weight_and_restores_exactly():
    fixture, _ = word_fixture()
    fixture.step()
    state = fixture.state_dict()
    restored, _ = word_fixture()
    restored.policy.load_state_dict(state["api_state"])
    restored.data_generator.set_state(state["data_generator"])
    restored.restore_component_energy(state["component_energy"])
    assert restored.energy_consumer.state_dict() == fixture.energy_consumer.state_dict()
    expected, got = fixture.step(), restored.step()
    assert torch.equal(expected["loss_g"], got["loss_g"])
    for bad in (None, {**state["component_energy"], "weight": .5}, {**state["component_energy"], "loss_sum": float("nan")}):
        previous = restored.energy_consumer.state_dict()
        with pytest.raises(ValueError, match="energy checkpoint"):
            restored.restore_component_energy(bad)
        assert restored.energy_consumer.state_dict() == previous


def test_scoped_factories_restore_globals_and_leave_other_threads_original():
    originals = (energy_adapters.behavior_adapters.BehaviorComponents,
                 energy_adapters.api_images.WordFixture, energy_adapters.word_tasks.restore_parent)
    with pytest.raises(RuntimeError, match="intentional"):
        with energy_adapters._factories():
            def other_thread():
                fixture, context = word_fixture(0.)
                obj = energy_adapters.api_images.WordFixture(device="cpu", seed=0, recipe_name=None,
                                                           max_steps=3, components=context)
                return type(obj)
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(other_thread).result() is WordFixture
            raise RuntimeError("intentional")
    assert originals == (energy_adapters.behavior_adapters.BehaviorComponents,
                         energy_adapters.api_images.WordFixture, energy_adapters.word_tasks.restore_parent)


def test_zero_dispatch_keeps_original_path_and_never_consumes_energy_or_rng(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-weight energy binding was constructed or consumed")
    monkeypatch.setattr(energy_adapters, "EnergyBehaviorComponents", forbidden)
    monkeypatch.setattr(Recipe, "energy_distance_loss", forbidden)
    task = deepcopy(load_tasks(ROOT)["unused_token_hold"])
    task["execution"]["steps"] = 2
    request = {"candidate": candidate(0.), "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    before = torch.get_rng_state().clone()
    result = _dispatch_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert torch.equal(before, torch.get_rng_state())
    assert "component_energy" not in result["evidence"]
    assert "component_energy" not in result["applied"]
    assert "component_energy_requested" not in result["evidence"]["guards"]


def test_energy_clock_uses_same_stateless_objective_and_all_four_parities(tmp_path):
    from experiments.forge.clockfree import run_clockfree
    from experiments.forge.views import grade_result
    task = load_tasks(ROOT)["clockfree_audit_measurement_v1"]
    before = torch.get_rng_state().clone()
    raw = run_clockfree({"candidate": candidate(), "protocol": {"seed": 0}}, task, tmp_path, "cpu")
    assert grade_result(task, raw)["metrics"]["parity_comparisons"] == 4
    assert grade_result(task, raw)["gate_status"] == "PASS"
    assert "particlegan/energy_distance.py" in raw["evidence"]["source_audit"]["source_sha256"]
    assert torch.equal(before, torch.get_rng_state())


def test_missing_signed_hold_state_rejected_before_native_restoration(monkeypatch, tmp_path):
    fixture, context = word_fixture()
    saved = {"fixture": fixture.state_dict()}
    saved["fixture"].pop("component_energy")
    torch.save(saved, tmp_path / "saved.pt")
    parent = {"result": {"evidence": {"artifact_root": str(tmp_path),
              "checkpoint": {"path": "saved.pt", "completed_steps": 0}}}}
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid signed state reached native restoration")
    monkeypatch.setattr(energy_adapters, "_Restore", forbidden)
    before = deepcopy(fixture.policy.state_dict())
    with energy_adapters._factories(), pytest.raises(ValueError, match="missing energy checkpoint"):
        energy_adapters._restore_parent(request={}, task={"execution": {"continuation_of": "parent"}},
                                       prerequisites={"parent": parent}, fixture=fixture, context=context)
    for key, value in before["models"]["generator"].items():
        assert torch.equal(value, fixture.policy.state_dict()["models"]["generator"][key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native word ladder requires CUDA")
@pytest.mark.parametrize("word_kind", ["ordinary", "acquisition"])
def test_actual_native_word_receipt_and_certified_state_include_separate_energy(tmp_path, word_kind):
    from benchmarks.toy_audit.reproducibility import reproducible_execution
    from experiments.forge.artifacts import verify_artifacts
    tasks = load_tasks(ROOT)
    task = deepcopy(tasks["five_word_joint_smoke"])
    if word_kind == "ordinary":
        task["evaluation"]["kind"] = "word_joint"
        task["execution"]["produces_state"] = False
    request = {"candidate": candidate(), "candidate_revision": "software-integration-only",
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    @reproducible_execution
    def execute(*, device):
        result = energy_adapters.run_word(request, task, tmp_path, device, execution_limit=2,
                                          retain_scored_outputs=False)
        state = result["evidence"]["component_energy"]
        guards = result["evidence"]["guards"]
        assert state["calls"] == state["active_calls"] == guards["component_energy_active_calls"] == 2
        assert guards["hooks_exercised"] and guards["unintended_rng_deviations"] == 0
        assert "component_transport" not in result["evidence"]
        receipt = json.loads((tmp_path / "adapter-receipt.json").read_text())
        assert receipt["evidence"]["component_energy"] == state
        checkpoint = result["evidence"]["provenance_checkpoint"]
        verify_artifacts(checkpoint["artifact_root"], checkpoint["artifact_manifest"])
        saved = torch.load(Path(checkpoint["artifact_root"]) / checkpoint["path"], weights_only=True, map_location=device)
        assert saved["fixture"]["component_energy"] == state
        assert "component_transport" not in saved["fixture"]
        assert result["scope"] != "ordinary_full_task"
    execute(device="cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native word hold requires CUDA")
def test_native_hold_restores_separate_signed_energy_from_explicit_synthetic_fixture(tmp_path):
    """One real prefix step + two real hold steps; synthetic counters/metrics.

    This is a checkpoint software fixture, never an acquisition or quality
    experiment. The immutable native restore and grading checks still execute.
    """
    from benchmarks.toy_audit.reproducibility import reproducible_execution
    from experiments.forge.artifacts import manifest_artifacts, verify_artifacts
    from experiments.forge.priors import prior_policy_receipt
    from experiments.forge.sampling import JOINT_WORDS_CLEAN, executed_receipt
    from experiments.forge.state import state_digest
    from experiments.forge.views import grade_result, task_fingerprint
    from experiments.forge.word_adapter import word_context
    from tests.test_forge_word_smoke_hold import evidence as synthetic_evidence
    tasks = load_tasks(ROOT)
    smoke, hold = tasks["five_word_joint_smoke"], tasks["five_word_joint_hold"]
    revision, compatibility = "synthetic-energy-restore-only", "software-parent-only"
    request = {"candidate": candidate(), "candidate_revision": revision,
               "protocol": {"seed": 0}, "tasks": {smoke["id"]: smoke, hold["id"]: hold},
               "jobs": [{"task_id": smoke["id"], "compatibility_key": compatibility}]}
    @reproducible_execution
    def execute(*, device):
        context = word_context(request, smoke, device)
        fixture = EnergyWordFixture(device=device, seed=0, recipe_name=None,
                                    max_steps=20001, components=context)
        for purpose in ("generated_words", "paired_reconstruction", "word_confirmation",
                        "word_reconstruction_confirmation"):
            context.streams.generator("eval", component="live", purpose=purpose)
        fixture.step()  # The only actual prefix update in this software cohort.
        prefix = 834   # Explicit synthetic acquisition counter fixture.
        fixture.policy.completed_steps = prefix
        fixture.opt_d.record.observed_steps = prefix
        for optimizer in (fixture.opt_g, fixture.opt_d):
            for values in optimizer.state.values():
                if "step" in values:
                    values["step"].fill_(prefix)
        energy = fixture.energy_consumer.state_dict()
        energy.update(calls=prefix, active_calls=prefix, loss_sum=energy["loss_sum"] * prefix)
        fixture.restore_component_energy(energy)
        saved = dict(fixture=fixture.state_dict(), streams=context.streams.state_dict(),
                     parent_task_fingerprint=task_fingerprint(smoke), candidate_revision=revision)
        root = tmp_path / "synthetic-parent"
        root.mkdir()
        path = root / "acquired-state.pt"
        torch.save(saved, path)
        synthetic = synthetic_evidence()
        identity = state_digest(saved["fixture"])
        synthetic["confirmations"][0].update(primary_state_sha256=identity, confirmed_state_sha256=identity)
        synthetic.update(prior_policy=prior_policy_receipt(fixture.recipe, fixture.prior),
                         **executed_receipt(JOINT_WORDS_CLEAN, eval_output_noise="clean"),
                         artifact_root=str(root), artifact_manifest=manifest_artifacts(root),
                         checkpoint=dict(path=path.name, sha256=file_hash(path), state_sha256=state_digest(saved),
                             training_state_sha256=identity, completed_steps=prefix,
                             selection="earliest_confirmed_passing_state"))
        parent_result = {**context.receipt(), "evidence": synthetic,
                         "scope": "synthetic_restore_prerequisite_only", "actual_fixture_updates": 1}
        assert grade_result(smoke, parent_result)["gate_status"] == "PASS"
        parent = {"candidate_revision": revision, "compatibility_key": compatibility, "result": parent_result}
        raw = energy_adapters.run_word(request, hold, tmp_path / "hold", device,
                    execution_limit=2, retain_scored_outputs=False, prerequisites={smoke["id"]: parent})
        assert raw["evidence"]["continuity"]["restored_exactly"]
        assert raw["evidence"]["continuity"]["prefix_steps"] == prefix
        assert raw["evidence"]["completed_steps"] == prefix + 2
        assert raw["evidence"]["component_energy"]["active_calls"] == prefix + 2
        assert raw["evidence"]["guards"]["hooks_exercised"]
        assert grade_result(hold, raw)["gate_status"] == "INCOMPLETE"
        checkpoint = raw["evidence"]["provenance_checkpoint"]
        verify_artifacts(checkpoint["artifact_root"], checkpoint["artifact_manifest"])
        restored = torch.load(Path(checkpoint["artifact_root"]) / checkpoint["path"],
                              weights_only=True, map_location=device)
        assert restored["fixture"]["component_energy"] == raw["evidence"]["component_energy"]
    execute(device="cuda:0")

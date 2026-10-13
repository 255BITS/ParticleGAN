"""Bounded MMD software fixtures; no paid queue work or quality credit.

Original task files remain unchanged. Short component copies run two updates;
the word restore fixture performs one actual prefix and two continuation updates
plus two reference updates. Its 834/836 labels and passing producer metrics are
explicitly synthetic. The original word hold is still INCOMPLETE (2/4000).
The native hook fixture adds .125 to its already drawn G-phase target to prove
that MMD consumes that panel; this is not an ordinary task comparison.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_images import WordFixture
from benchmarks.toy_audit.gaussian1d_quality import sample_target
from experiments.forge import energy_adapters, mmd_adapters
from experiments.forge.adapters import _dispatch_task
from experiments.forge.api import (CapabilityError, FormulationContext,
    resolve_public_recipe, task_formulation_context, task_policy_blockers)
from experiments.forge.contracts import file_hash
from experiments.forge.mmd_adapters import MMDWordFixture
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from experiments.forge.views import load_tasks
from particlegan import Recipe
from particlegan.execution import serial_autograd


ROOT = Path(__file__).resolve().parents[1]
HOSTS = ("gaussian1d_smoke", "ring16_acquisition", "two_pole", "unused_token_hold",
         "ae_gan_hold", "five_word_joint_smoke", "clockfree_audit_measurement_v1")


@pytest.fixture(autouse=True)
def isolated_cpu_software_cohort():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]), serial_autograd():
        yield
    torch.set_num_threads(old)


def candidate(weight=1., *, energy=0., bandwidths=None):
    card = json.loads((ROOT / "configs/forge/ideas/r1r2-non-saturating-ams-m110-v1.json").read_text())
    card["recipe_overrides"].update(output_mmd_weight=weight, energy_distance_weight=energy)
    if bandwidths is not None:
        card["recipe_overrides"]["output_mmd_bandwidths"] = bandwidths
    card["requires_capabilities"].append("output_marginal_v1")
    return card


def word(weight=1.):
    task = load_tasks(ROOT)["five_word_joint_smoke"]
    context = task_formulation_context(candidate(weight), task, {"seed": 0}, device="cpu", root=ROOT)
    cls = MMDWordFixture if weight else WordFixture
    return cls(device="cpu", seed=0, recipe_name=None, max_steps=20001, components=context), context


def same(a, b):
    assert state_digest(a) == state_digest(b)


def test_all_seven_public_preflights_original_pins_and_owned_metadata():
    from experiments.forge.boundaries import ownership_receipt
    from experiments.forge.preflight import task_preflight
    tasks = load_tasks(ROOT)
    pins = {}
    for name in HOSTS:
        task = tasks[name]
        pins.update(task["evaluation"].get("sources", {}))
        assert task_preflight(task, candidate(), {"seed": 0}, root=ROOT, tasks=tasks) == []
        context = task_formulation_context(candidate(), task, {"seed": 0}, root=ROOT)
        assert context.capabilities()["output_marginal_v1"]
        receipt = ownership_receipt(candidate(), task, asdict(context.recipe), {"seed": 0})
        assert receipt["recipe_fields"]["output_mmd_weight"]["owner"] == "technique"
        assert receipt["recipe_fields"]["output_mmd_bandwidths"]["owner"] == "hyperparameter"
    assert len(pins) == 19
    assert all(file_hash(ROOT / path) == digest for path, digest in pins.items())


@pytest.mark.parametrize("mutation", [
    lambda t: t["execution"].pop("transport_consumer"),
    lambda t: t["execution"]["transport_contract"].update(conditioning_in_distance=True),
    lambda t: t["execution"]["transport_contract"].update(new_paired_supervision=True),
    lambda t: t["execution"]["transport_contract"].update(new_forward_passes=True),
    lambda t: t["execution"]["transport_contract"].update(new_random_draws=True),
    lambda t: t["execution"].update(host="trajectory"),
])
def test_unsupported_component_contract_fails_before_construction(mutation):
    task = deepcopy(load_tasks(ROOT)["two_pole"])
    mutation(task)
    assert task_policy_blockers(task, candidate())
    with pytest.raises(CapabilityError):
        task_formulation_context(candidate(), task, {"seed": 0})


def test_component_composition_rejected_but_native_composition_supported(tmp_path, monkeypatch):
    tasks = load_tasks(ROOT)
    declaration = candidate(energy=1.)
    for name in ("two_pole", "unused_token_hold", "ae_gan_hold", "five_word_joint_smoke"):
        task = tasks[name]
        assert "simultaneous" in " ".join(task_policy_blockers(task, declaration))
        with pytest.raises(CapabilityError):
            task_formulation_context(declaration, task, {"seed": 0})
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported composition reached native construction")
    monkeypatch.setattr(mmd_adapters.behavior_adapters, "run_behavior", forbidden)
    task = tasks["two_pole"]
    with pytest.raises(CapabilityError, match="simultaneous"):
        _dispatch_task({"candidate": declaration, "tasks": {task["id"]: task}},
                       {"task_id": task["id"]}, tmp_path, "cpu")
    assert task_policy_blockers(tasks["ring16_acquisition"], declaration) == []
    with pytest.raises(CapabilityError, match="cannot compose"):
        FormulationContext(recipe_preset="k3p", recipe_overrides={"output_mmd_weight": 1.,
                           "energy_distance_weight": 1.}, execution_path="public_components",
                           component_transport="output_marginal_v1")


@pytest.mark.parametrize("host", ["gaussian1d_smoke", "ring16_acquisition"])
def test_native_same_panel_no_extra_forwards_draws_and_exact_resume(host, monkeypatch, record_property):
    task = load_tasks(ROOT)[host]
    spec = resolve_vector_spec(task, root=ROOT)
    if host == "ring16_acquisition":
        from benchmarks.transfer_suite.vector_tasks import sample_target as draw_target
    else:
        draw_target = sample_target
    def construct(weight):
        context = task_formulation_context(candidate(weight), task, {"seed": 0}, root=ROOT)
        trainer = context.build_trainer(*build_vector_models(context, spec), max_steps=task["execution"]["steps"])
        return context, trainer
    contexts, trainers = zip(construct(0.), construct(1.))
    for role in ("G", "D", "prior"):
        same(getattr(trainers[0], role).state_dict(), getattr(trainers[1], role).state_dict())
    same(contexts[0].streams.state_dict(), contexts[1].streams.state_dict())
    counts = [{"G": 0, "D": 0}, {"G": 0, "D": 0}]
    for bucket, trainer in zip(counts, trainers):
        for role in bucket:
            def capture(_module, _inputs, _output, *, role=role, bucket=bucket):
                bucket[role] += 1
            getattr(trainer, role).register_forward_hook(capture)
    panels = []
    original = Recipe.output_mmd_loss
    def objective(self, fake, real):
        panels.append((fake, real))
        return original(self, fake, real)
    monkeypatch.setattr(Recipe, "output_mmd_loss", objective)
    saved = None
    for step in range(2):
        for context, trainer in zip(contexts, trainers):
            data = context.streams.generator("data", component="target", purpose="training", device="cpu")
            real = draw_target(spec, context.recipe.batch_size, data, step)
            target_g = real + .125
            output = trainer.step(real, generator_real=lambda: target_g, collect_stats=True)
            if context.recipe.output_mmd_weight:
                assert torch.equal(panels[-1][1], target_g)
                assert panels[-1][0].requires_grad and torch.isfinite(output["output_mmd"])
            else:
                assert "output_mmd" not in output
        same(contexts[0].streams.state_dict(), contexts[1].streams.state_dict())
        if step == 0:
            saved = deepcopy(contexts[1].state_dict())
    assert counts[0] == counts[1] and len(panels) == 2
    final = contexts[1].state_dict()
    resumed_context, resumed = construct(1.)
    resumed_context.load_state_dict(saved)
    data = resumed_context.streams.generator("data", component="target", purpose="training", device="cpu")
    real = draw_target(spec, resumed.recipe.batch_size, data, 1)
    resumed.step(real, generator_real=lambda: real + .125, collect_stats=True)
    same(final, resumed_context.state_dict())
    before = deepcopy(contexts[0].state_dict())
    with pytest.raises(ValueError, match="recipe"):
        contexts[0].load_state_dict(saved)
    same(before, contexts[0].state_dict())
    record_property("actual_training_updates", 5)
    record_property("qualification_input", False)


@pytest.mark.parametrize("host", ["two_pole", "unused_token_hold", "ae_gan_hold"])
def test_two_actual_component_updates_separate_state_and_original_incomplete_gate(tmp_path, host, record_property):
    task = deepcopy(load_tasks(ROOT)[host])
    task["execution"]["steps"] = 2  # Explicit software-only task copy.
    request = {"candidate": candidate(), "protocol": {"seed": 0}, "tasks": {host: task}}
    before = torch.get_rng_state().clone()
    result = _dispatch_task(request, {"task_id": host}, tmp_path, "cpu")
    assert torch.equal(before, torch.get_rng_state())
    guards = result["evidence"]["guards"]
    assert guards["hooks_exercised"] and guards["component_mmd_active_calls"] == 2
    state = result["evidence"]["component_mmd"]
    assert state["calls"] == state["active_calls"] == 2 and state["weight"] == 1.
    assert tuple(state["bandwidths"]) == resolve_public_recipe(candidate()).output_mmd_bandwidths
    assert result["applied"]["component_transport"] is None
    assert result["applied"]["mmd_binding"]["native_loop"].endswith("run_behavior")
    assert "component_energy" not in result["applied"]
    from experiments.forge.artifacts import verify_artifacts
    checkpoint = result["evidence"]["provenance_checkpoint"]
    verify_artifacts(checkpoint["artifact_root"], checkpoint["artifact_manifest"])
    saved = torch.load(Path(checkpoint["artifact_root"]) / checkpoint["path"], weights_only=True)
    assert saved["component_mmd"] == state
    from experiments.forge.views import grade_result
    assert grade_result(load_tasks(ROOT)[host], result)["status"] == "INCOMPLETE"
    record_property("actual_training_updates", 2)
    record_property("qualification_input", False)


def test_word_same_native_step_streams_and_original_encoder_gradients(record_property):
    control, ccontext = word(0.)
    active, acontext = word(1.)
    for role in ("G", "E", "D", "prior"):
        same(getattr(control, role).state_dict(), getattr(active, role).state_dict())
    counts = [{"G": 0, "D": 0, "E": 0}, {"G": 0, "D": 0, "E": 0}]
    for bucket, fixture in zip(counts, (control, active)):
        for name in bucket:
            def observe(_module, _inputs, _output, *, key=name, bucket=bucket):
                bucket[key] += 1
            getattr(fixture, name).register_forward_hook(observe)
    control.step()
    active.step()
    assert counts[0] == counts[1]
    assert active.transport is None and active.mmd_consumer.active_calls == 1
    assert "component_transport" not in active.state_dict()
    assert "component_mmd" not in control.state_dict()
    for a, b in zip(active.E.parameters(), control.E.parameters()):
        assert torch.equal(a.grad, b.grad)
    same(ccontext.streams.state_dict(), acontext.streams.state_dict())
    assert MMDWordFixture.observe is WordFixture.observe
    record_property("actual_training_updates", 2)
    record_property("qualification_input", False)


@pytest.mark.parametrize("energy", [0., 1.])
@pytest.mark.parametrize("host", ["two_pole", "five_word_joint_smoke"])
def test_zero_mmd_dispatch_bypasses_mmd_and_keeps_existing_energy_or_native_path(energy, host, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled MMD binding was entered or consumed")
    monkeypatch.setattr(mmd_adapters, "_factories", forbidden)
    monkeypatch.setattr(Recipe, "output_mmd_loss", forbidden)
    module = energy_adapters if energy else (mmd_adapters.word_adapter if host == "five_word_joint_smoke"
                                           else mmd_adapters.behavior_adapters)
    name = "run_word" if host == "five_word_joint_smoke" else "run_behavior"
    sentinel = {"existing_path": name, "energy": energy}
    monkeypatch.setattr(module, name, lambda *args, **kwargs: sentinel)
    task = load_tasks(ROOT)[host]
    assert _dispatch_task({"candidate": candidate(0., energy=energy), "tasks": {host: task}},
                          {"task_id": host}, tmp_path, "cpu") is sentinel
    assert "output_mmd_weight" not in resolve_public_recipe(candidate(0.)).to_dict()
    assert "output_mmd_bandwidths" not in resolve_public_recipe(candidate(0.)).to_dict()


def test_factory_thread_isolation_shared_lock_nested_scopes_and_exception_cleanup():
    originals = (mmd_adapters.behavior_adapters.BehaviorComponents,
                 mmd_adapters.api_images.WordFixture, mmd_adapters.word_tasks.restore_parent)
    assert mmd_adapters._LOCK is energy_adapters._LOCK
    with pytest.raises(RuntimeError, match="intentional"):
        with energy_adapters._factories():
            energy_globals = (mmd_adapters.behavior_adapters.BehaviorComponents,
                             mmd_adapters.api_images.WordFixture, mmd_adapters.word_tasks.restore_parent)
            with mmd_adapters._factories() as instances:
                _, context = word()
                assert isinstance(mmd_adapters.api_images.WordFixture(device="cpu", seed=0, recipe_name=None,
                                  max_steps=20001, components=context), MMDWordFixture)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    other = executor.submit(mmd_adapters.api_images.WordFixture, device="cpu", seed=0,
                        recipe_name=None, max_steps=20001, components=context).result()
                assert type(other) is WordFixture and len(instances) == 1
            assert energy_globals == (mmd_adapters.behavior_adapters.BehaviorComponents,
                mmd_adapters.api_images.WordFixture, mmd_adapters.word_tasks.restore_parent)
            raise RuntimeError("intentional")
    assert originals == (mmd_adapters.behavior_adapters.BehaviorComponents,
                        mmd_adapters.api_images.WordFixture, mmd_adapters.word_tasks.restore_parent)
    assert mmd_adapters._ACTIVE.get() is None and energy_adapters._ACTIVE.get() is None


def test_word_output_hook_restored_on_step_exception(monkeypatch):
    fixture, _ = word()
    original = fixture.transport
    def fail(*args, **kwargs):
        raise RuntimeError("intentional native step error")
    monkeypatch.setattr(mmd_adapters._Word, "step", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        fixture.step()
    assert fixture.transport is original and fixture.mmd_consumer.calls == 0


def test_mmd_parent_restore_scope_and_pending_state_cleanup_on_native_exception(monkeypatch, tmp_path):
    fixture, context = word()
    torch.save({"fixture": fixture.state_dict()}, tmp_path / "state.pt")
    parent = {"result": {"evidence": {"artifact_root": str(tmp_path),
              "checkpoint": {"path": "state.pt", "completed_steps": 0}}}}
    task = {"execution": {"continuation_of": "parent"}}
    before = deepcopy(fixture.state_dict()), deepcopy(context.streams.state_dict())
    def fail(*args, **kwargs):
        assert hasattr(fixture, "_pending_mmd_state")
        raise RuntimeError("intentional original parent validation error")
    monkeypatch.setattr(mmd_adapters, "_Restore", fail)
    with pytest.raises(ValueError, match="scoped binding"):
        mmd_adapters._restore_parent({}, task, {"parent": parent}, fixture, context)
    with mmd_adapters._factories(), pytest.raises(RuntimeError, match="original parent validation"):
        mmd_adapters._restore_parent({}, task, {"parent": parent}, fixture, context)
    assert not hasattr(fixture, "_pending_mmd_state")
    same(before[0], fixture.state_dict())
    same(before[1], context.streams.state_dict())


@pytest.mark.parametrize("corruption", ["missing", "weight", "weight_representation", "bandwidths", "representation", "nonfinite", "counter", "extra"])
def test_bad_word_mmd_state_rejected_before_parent_mutation(corruption, monkeypatch, tmp_path):
    fixture, context = word()
    saved = {"fixture": fixture.state_dict()}
    state = saved["fixture"]["component_mmd"]
    if corruption == "missing":
        saved["fixture"].pop("component_mmd")
    elif corruption == "weight":
        state["weight"] = .5
    elif corruption == "weight_representation":
        state["weight"] = int(state["weight"])
    elif corruption == "bandwidths":
        state["bandwidths"] = (1.,)
    elif corruption == "representation":
        state["bandwidths"] = list(state["bandwidths"])
    elif corruption == "nonfinite":
        state["loss_sum"] = float("nan")
    elif corruption == "counter":
        state.update(calls=1, active_calls=1, minimum_loss=0., maximum_loss=0.)
    else:
        state["unknown"] = 0
    torch.save(saved, tmp_path / "state.pt")
    parent = {"result": {"evidence": {"artifact_root": str(tmp_path),
              "checkpoint": {"path": "state.pt", "completed_steps": 0}}}}
    def forbidden(*args, **kwargs):
        raise AssertionError("bad MMD state reached native parent restoration")
    monkeypatch.setattr(mmd_adapters, "_Restore", forbidden)
    before = deepcopy(fixture.state_dict()), deepcopy(context.streams.state_dict())
    with mmd_adapters._factories(), pytest.raises(ValueError):
        mmd_adapters._restore_parent({}, {"execution": {"continuation_of": "parent"}},
                                    {"parent": parent}, fixture, context)
    same(before[0], fixture.state_dict())
    same(before[1], context.streams.state_dict())
    assert not hasattr(fixture, "_pending_mmd_state")


def test_word_one_real_prefix_two_continuation_steps_strict_original_parent_restore(tmp_path, record_property):
    from experiments.forge.artifacts import manifest_artifacts
    from experiments.forge.priors import prior_policy_receipt
    from experiments.forge.sampling import JOINT_WORDS_CLEAN, executed_receipt
    from experiments.forge.views import grade_result, task_fingerprint
    from experiments.forge.word_adapter import word_context
    from tests.test_forge_word_smoke_hold import evidence as synthetic_evidence
    tasks = load_tasks(ROOT)
    smoke, hold = tasks["five_word_joint_smoke"], tasks["five_word_joint_hold"]
    revision, compatibility = "synthetic-MMD-restore-only", "software-parent-only"
    request = {"candidate": candidate(), "candidate_revision": revision, "protocol": {"seed": 0},
        "tasks": {smoke["id"]: smoke, hold["id"]: hold},
        "jobs": [{"task_id": smoke["id"], "compatibility_key": compatibility}]}
    fixture, context = word()
    fixture.step()  # Exactly one actual update, not measured acquisition.
    prefix = 834    # Explicit synthetic reducer/restore counter fixture.
    fixture.policy.completed_steps = prefix
    fixture.opt_d.record.observed_steps = prefix
    for optimizer in (fixture.opt_g, fixture.opt_d):
        for values in optimizer.state.values():
            if "step" in values:
                values["step"].fill_(prefix)
    mmd = fixture.mmd_consumer.state_dict()
    mmd.update(calls=prefix, active_calls=prefix, loss_sum=mmd["loss_sum"] * prefix)
    fixture.restore_component_mmd(mmd)
    saved = dict(fixture=fixture.state_dict(), streams=context.streams.state_dict(),
                 parent_task_fingerprint=task_fingerprint(smoke), candidate_revision=revision)
    parent_root = tmp_path / "synthetic-parent"
    parent_root.mkdir()
    path = parent_root / "acquired-state.pt"
    torch.save(saved, path)
    synthetic = synthetic_evidence()
    identity = state_digest(saved["fixture"])
    synthetic["confirmations"][0].update(primary_state_sha256=identity, confirmed_state_sha256=identity)
    synthetic.update(prior_policy=prior_policy_receipt(fixture.recipe, fixture.prior),
        **executed_receipt(JOINT_WORDS_CLEAN, eval_output_noise="clean"),
        artifact_root=str(parent_root), artifact_manifest=manifest_artifacts(parent_root),
        checkpoint=dict(path=path.name, sha256=file_hash(path), state_sha256=state_digest(saved),
            training_state_sha256=identity, completed_steps=prefix, selection="earliest_confirmed_passing_state"))
    parent_result = {**context.receipt(), "evidence": synthetic,
                     "scope": "synthetic_restore_prerequisite_only", "actual_fixture_updates": 1}
    assert grade_result(smoke, parent_result)["gate_status"] == "PASS"  # Synthetic reducer fixture only.
    parent = {"candidate_revision": revision, "compatibility_key": compatibility, "result": parent_result}
    restored_context = word_context(request, hold, "cpu", root=ROOT)
    restored = MMDWordFixture(device="cpu", seed=0, recipe_name=None, max_steps=20001, components=restored_context)
    with mmd_adapters._factories():
        continuity = mmd_adapters._restore_parent(request, hold, {smoke["id"]: parent}, restored, restored_context)
    assert continuity["restored_exactly"] and continuity["prefix_steps"] == prefix
    same(saved["fixture"], restored.state_dict())
    same(saved["streams"], restored_context.streams.state_dict())
    assert not hasattr(restored, "_pending_mmd_state")
    for _ in range(2):
        restored.step()
        fixture.step()
    same(fixture.state_dict(), restored.state_dict())
    same(context.streams.state_dict(), restored_context.streams.state_dict())
    assert restored.mmd_consumer.active_calls == restored.completed_steps == 836
    raw = {**restored_context.receipt(), "evidence": dict(prior_policy=prior_policy_receipt(restored.recipe, restored.prior),
        executed_updates=2, completed_steps=836, continuity=continuity,
        guards=dict(optimizer_updates={role: 836 for role in ("generator", "encoder", "prior", "discriminator")}))}
    assert grade_result(hold, raw)["gate_status"] == "INCOMPLETE"
    record_property("actual_training_updates", 5)
    record_property("synthetic_prefix_steps", 834)
    record_property("synthetic_final_steps", 836)
    record_property("qualification_input", False)


def test_native_clock_same_stateless_mmd_and_all_four_full_restore_parities(tmp_path, record_property):
    from experiments.forge.clockfree import run_clockfree
    from experiments.forge.views import grade_result
    task = load_tasks(ROOT)["clockfree_audit_measurement_v1"]
    before = torch.get_rng_state().clone()
    raw = run_clockfree({"candidate": candidate(), "protocol": {"seed": 0}}, task, tmp_path, "cpu")
    assert grade_result(task, raw)["metrics"]["parity_comparisons"] == 4
    assert grade_result(task, raw)["gate_status"] == "PASS"  # Separate software cohort, no grade reuse.
    assert "particlegan/kernel_mmd.py" in raw["evidence"]["source_audit"]["source_sha256"]
    assert torch.equal(before, torch.get_rng_state())
    proof = torch.load(Path(raw["evidence"]["artifact_root"]) / "comparisons.pt", weights_only=True)
    assert proof["recipe"]["output_mmd_weight"] == 1.
    assert tuple(proof["recipe"]["output_mmd_bandwidths"]) == resolve_public_recipe(candidate()).output_mmd_bandwidths
    assert raw["cost"]["completed_updates"] == 24
    record_property("actual_training_updates", 24)
    record_property("qualification_input", False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native word ladder requires CUDA")
@pytest.mark.parametrize("word_kind", ["ordinary", "acquisition"])
def test_two_cuda_word_updates_actual_dispatch_receipts_and_saved_signed_state(tmp_path, word_kind, record_property):
    """Two real steps, explicit ordinary/ladder software copies, no acquisition."""
    from benchmarks.toy_audit.reproducibility import reproducible_execution
    from experiments.forge.artifacts import verify_artifacts
    task = deepcopy(load_tasks(ROOT)["five_word_joint_smoke"])
    if word_kind == "ordinary":
        task["evaluation"]["kind"] = "word_joint"
        task["execution"]["produces_state"] = False
    request = {"candidate": candidate(), "candidate_revision": "software-MMD-integration-only",
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    @reproducible_execution
    def execute(*, device):
        result = mmd_adapters.run_word(request, task, tmp_path, device, execution_limit=2,
                                      retain_scored_outputs=False)
        state = result["evidence"]["component_mmd"]
        guards = result["evidence"]["guards"]
        assert state["calls"] == state["active_calls"] == guards["component_mmd_active_calls"] == 2
        assert guards["hooks_exercised"] and guards["unintended_rng_deviations"] == 0
        assert "component_energy" not in result["evidence"]
        receipt = json.loads((tmp_path / "adapter-receipt.json").read_text())
        assert tuple(receipt["evidence"]["component_mmd"]["bandwidths"]) == tuple(state["bandwidths"])
        checkpoint = result["evidence"]["provenance_checkpoint"]
        verify_artifacts(checkpoint["artifact_root"], checkpoint["artifact_manifest"])
        saved = torch.load(Path(checkpoint["artifact_root"]) / checkpoint["path"], weights_only=True, map_location=device)
        assert saved["fixture"]["component_mmd"] == state
        assert "component_transport" not in saved["fixture"]
        assert result["scope"] != "ordinary_full_task"
    execute(device="cuda:0")
    record_property("actual_training_updates", 2)
    record_property("qualification_input", False)

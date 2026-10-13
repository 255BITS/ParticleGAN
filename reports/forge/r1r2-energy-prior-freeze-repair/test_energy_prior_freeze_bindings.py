"""Bounded CPU software cohorts, never qualification or a paid experiment.

Native scalar: three actual updates per matched learned/frozen fixture, plus
one resumed update. Word: one real prefix and two continuation updates; the
834-step acquisition label and passing producer metrics are explicitly synthetic.
The original CUDA-only ladder guard remains untouched and the hold is INCOMPLETE.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_images import WordFixture
from benchmarks.toy_audit.gaussian1d_quality import sample_target
from experiments.forge import energy_adapters
from experiments.forge.api import task_formulation_context
from experiments.forge.artifacts import manifest_artifacts, verify_artifacts
from experiments.forge.contracts import file_hash
from experiments.forge.energy_adapters import EnergyWordFixture
from experiments.forge.priors import prior_policy_receipt
from experiments.forge.sampling import JOINT_WORDS_CLEAN, executed_receipt
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from experiments.forge.views import grade_result, load_tasks, task_fingerprint
from experiments.forge.word_adapter import word_context
from particlegan.execution import serial_autograd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = "2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12"


@pytest.fixture(autouse=True)
def cpu_software_cohort():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]), serial_autograd():
        yield
    torch.set_num_threads(previous)


def candidate(policy="frozen"):
    card = json.loads((ROOT / "configs/forge/ideas/r1r2-non-saturating-energy-u-v1.json").read_text())
    card["recipe_overrides"]["prior_update"] = policy
    if policy == "frozen":
        card["requires_capabilities"].remove("learned_locations")
    return card


def same(a, b):
    assert state_digest(a) == state_digest(b)


def counts(optimizer, model):
    values = [int(optimizer.state[p]["step"]) for p in model.parameters()
              if "step" in optimizer.state.get(p, {})]
    return min(values) if values else 0


def scalar(policy):
    task = load_tasks(ROOT)["gaussian1d_smoke"]
    context = task_formulation_context(candidate(policy), task, {"seed": 0}, device="cpu", root=ROOT)
    spec = resolve_vector_spec(task, root=ROOT)
    trainer = context.build_trainer(*build_vector_models(context, spec), max_steps=task["execution"]["steps"])
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    return context, trainer, spec, data


def word(policy="frozen"):
    task = load_tasks(ROOT)["five_word_joint_smoke"]
    request = {"candidate": candidate(policy), "protocol": {"seed": 0}}
    context = word_context(request, task, "cpu", root=ROOT)
    fixture = EnergyWordFixture(device="cpu", seed=0, recipe_name=None, max_steps=20001, components=context)
    return context, fixture


def test_source_and_evaluator_bytes_match_actual_frozen_execution():
    publication = json.loads((ROOT / "reports/forge/r1r2-energy-distance-repair/publication/results.json").read_text())
    row = next(r for r in publication["certified_results"] if r["role"] == "energy_u"
               and r["task_id"] == "ring16_acquisition")
    archived = Path(row["checkpoint"]["artifact_root"]).parent / "request.json"
    source = json.loads(archived.read_text())["request"]["source"]
    assert source["digest"] == SOURCE and len(source["files"]) == 1216
    for path, digest in source["files"].items():
        assert file_hash(ROOT / path) == file_hash(Path(source["snapshot_path"]) / path) == digest
    for task in load_tasks(ROOT).values():
        if task["id"] in ("gaussian1d_smoke", "ring16_acquisition", "two_pole", "unused_token_hold",
                          "ae_gan_hold", "five_word_joint_smoke", "clockfree_audit_measurement_v1"):
            for path, digest in task["evaluation"].get("sources", {}).items():
                assert file_hash(ROOT / path) == digest


def test_positive_energy_native_freeze_preserves_initialization_consumption_and_resume():
    learned_context, learned, spec, learned_data = scalar("learned")
    frozen_context, frozen, _, frozen_data = scalar("frozen")
    for role in ("G", "D", "prior"):
        same(getattr(learned, role).state_dict(), getattr(frozen, role).state_dict())
    same(learned_context.initialization, frozen_context.initialization)
    same(learned_context.streams.state_dict(), frozen_context.streams.state_dict())
    initial = deepcopy(frozen_context.state_dict())
    locations = frozen.prior.z.clone()
    assert not frozen.prior.z.requires_grad and "z" in dict(frozen.prior.named_buffers())
    assert not list(frozen.prior.parameters())
    assert len(frozen.opt_g.param_groups) == 1 < len(learned.opt_g.param_groups)
    forward_counts = [{"G": 0, "D": 0}, {"G": 0, "D": 0}]
    for bucket, trainer in zip(forward_counts, (learned, frozen)):
        for role in ("G", "D"):
            def capture(module, inputs, result, *, bucket=bucket, role=role):
                bucket[role] += 1
            getattr(trainer, role).register_forward_hook(capture)
    checkpoint = None
    for step in range(3):
        a = sample_target(spec, learned.recipe.batch_size, learned_data, step)
        b = sample_target(spec, frozen.recipe.batch_size, frozen_data, step)
        assert torch.equal(a, b)
        for trainer in (learned, frozen):
            output = trainer.step(a, collect_stats=True)
            assert torch.isfinite(output["energy_distance"])
        same(learned_context.streams.state_dict(), frozen_context.streams.state_dict())
        assert torch.equal(frozen.prior.z, locations)
        assert counts(frozen.opt_g, frozen.prior) == 0
        assert counts(frozen.opt_g, frozen.G) == counts(frozen.opt_d, frozen.D) == step + 1
        if step == 1:
            checkpoint = deepcopy(frozen_context.state_dict())
    assert forward_counts[0] == forward_counts[1]
    for role in ("G", "D"):
        assert state_digest(getattr(frozen, role).state_dict()) != state_digest(initial["trainer"]["models"][role])
    final = frozen_context.state_dict()
    resumed_context, resumed, _, resumed_data = scalar("frozen")
    resumed_context.load_state_dict(checkpoint)
    resumed.step(sample_target(spec, resumed.recipe.batch_size, resumed_data, 2), collect_stats=True)
    same(final, resumed_context.state_dict())
    assert torch.equal(resumed.prior.z, locations)
    with pytest.raises(ValueError, match="recipe"):
        learned_context.load_state_dict(checkpoint)


def test_word_positive_energy_frozen_matches_initial_tensors_and_consumed_streams():
    a_context, a = word("learned")
    b_context, b = word("frozen")
    assert EnergyWordFixture.observe is WordFixture.observe
    for role in ("G", "E", "D", "prior"):
        same(getattr(a, role).state_dict(), getattr(b, role).state_dict())
    same(a_context.initialization, b_context.initialization)
    same(a_context.streams.state_dict(), b_context.streams.state_dict())
    locations = b.prior.z.clone()
    assert not list(b.prior.parameters()) and "z" in dict(b.prior.named_buffers())
    assert len(b.opt_g.param_groups) == 2 < len(a.opt_g.param_groups)
    a.step()
    b.step()
    same(a_context.streams.state_dict(), b_context.streams.state_dict())
    assert torch.equal(locations, b.prior.z)
    for role, optimizer in (("G", b.opt_g), ("E", b.opt_g), ("D", b.opt_d)):
        assert counts(optimizer, getattr(b, role)) == 1
    assert counts(b.opt_g, b.prior) == 0
    assert b.energy_consumer.active_calls == 1


def test_cpu_word_one_prefix_two_continuation_updates_strict_signed_restore(tmp_path):
    from tests.test_forge_word_smoke_hold import evidence as synthetic_evidence
    tasks = load_tasks(ROOT)
    smoke, hold = tasks["five_word_joint_smoke"], tasks["five_word_joint_hold"]
    revision, compatibility = "synthetic-energy-frozen-restore-only", "software-parent-only"
    request = {"candidate": candidate(), "candidate_revision": revision, "protocol": {"seed": 0},
               "tasks": {smoke["id"]: smoke, hold["id"]: hold},
               "jobs": [{"task_id": smoke["id"], "compatibility_key": compatibility}]}
    context, fixture = word()
    fixture.step()  # Exactly one real prefix update, never a passing acquisition.
    locations = fixture.prior.z.clone()
    prefix = 834  # Explicit synthetic counter, not 834 actual updates.
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
    parent_root = tmp_path / "synthetic-parent"
    parent_root.mkdir()
    path = parent_root / "acquired-state.pt"
    torch.save(saved, path)
    synthetic = synthetic_evidence()
    synthetic["guards"]["optimizer_updates"]["prior"] = 0
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
    restored = EnergyWordFixture(device="cpu", seed=0, recipe_name=None, max_steps=20001, components=restored_context)
    with energy_adapters._factories():
        continuity = energy_adapters._restore_parent(request=request, task=hold,
            prerequisites={smoke["id"]: parent}, fixture=restored, context=restored_context)
    assert continuity["restored_exactly"] and continuity["prefix_steps"] == prefix
    same(saved["fixture"], restored.state_dict())
    same(saved["streams"], restored_context.streams.state_dict())
    restored.step()
    restored.step()
    fixture.step()
    fixture.step()
    same(fixture.state_dict(), restored.state_dict())
    same(context.streams.state_dict(), restored_context.streams.state_dict())
    assert torch.equal(locations, restored.prior.z)
    assert not list(restored.prior.parameters())
    actual_counts = {role: counts(opt, module) for role, opt, module in (
        ("generator", restored.opt_g, restored.G), ("encoder", restored.opt_g, restored.E),
        ("prior", restored.opt_g, restored.prior), ("discriminator", restored.opt_d, restored.D))}
    assert actual_counts == {"generator": 836, "encoder": 836, "prior": 0, "discriminator": 836}
    assert restored.energy_consumer.active_calls == 836
    raw = {**restored_context.receipt(), "evidence": dict(prior_policy=prior_policy_receipt(restored.recipe, restored.prior),
        executed_updates=2, completed_steps=836, continuity=continuity, guards=dict(optimizer_updates=actual_counts))}
    assert grade_result(hold, raw)["gate_status"] == "INCOMPLETE"
    verify_artifacts(parent_root, synthetic["artifact_manifest"])


@pytest.mark.parametrize("corruption", ["missing", "weight", "nonfinite", "counter"])
def test_frozen_word_rejects_corrupted_signed_state_before_native_restore(corruption, tmp_path, monkeypatch):
    context, fixture = word()
    state = {"fixture": fixture.state_dict()}
    component = state["fixture"]["component_energy"]
    if corruption == "missing":
        state["fixture"].pop("component_energy")
    elif corruption == "weight":
        component["weight"] = .5
    elif corruption == "nonfinite":
        component["loss_sum"] = float("nan")
    else:
        component.update(calls=1, active_calls=1, minimum_loss=0., maximum_loss=0.)
    torch.save(state, tmp_path / "state.pt")
    parent = {"result": {"evidence": {"artifact_root": str(tmp_path),
              "checkpoint": {"path": "state.pt", "completed_steps": 0}}}}
    def forbidden(*args, **kwargs):
        raise AssertionError("corrupt signed state reached original native restore")
    monkeypatch.setattr(energy_adapters, "_Restore", forbidden)
    before = deepcopy(fixture.state_dict()), deepcopy(context.streams.state_dict())
    with energy_adapters._factories(), pytest.raises(ValueError):
        energy_adapters._restore_parent(request={}, task={"execution": {"continuation_of": "parent"}},
            prerequisites={"parent": parent}, fixture=fixture, context=context)
    same(before[0], fixture.state_dict())
    same(before[1], context.streams.state_dict())

"""Actual constructor/JSON transport controls, never a numerical acquisition.

Keep every original host, full budget and strict source validator. Only the
full-acquisition entry point is replaced with a labelled two-update CPU probe;
it calls the real producer constructor and public UpdatePolicy unchanged.
"""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest
import torch

from experiments.forge import adapters, policy_cohorts as registry
from experiments.forge.api import CapabilityError, task_formulation_context
from experiments.forge.policy_adapters import typed_state_digest


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("trajectory", "residual_student", "unipolar", "mid_scale_identity",
         "unused", "cover", "ae", "word")


@pytest.fixture(autouse=True)
def structural_cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    try:
        yield
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def wire_request(name):
    """Current source makers and real ownership receipt, then JSON transport."""
    if name in NAMES[:4]:
        from experiments.forge import conditional_policy_contracts as module
        task = module.make_conditional_variant(ROOT, name)
        producer_name, constructor, entry = "conditional_policy_adapters", "ConditionalPolicyFixture", "run_behavior"
    elif name == "unused":
        from experiments.forge import routed_policy_contracts as module
        task = module.make_unused_variant(ROOT)
        producer_name, constructor, entry = "routed_policy_adapters", "UnusedTokenRoutedFixture", "run_behavior"
    elif name == "cover":
        from experiments.forge import multibank_policy_contracts as module
        task = module.make_variant(ROOT)
        producer_name, constructor, entry = "multibank_policy_adapters", "CoverMultibankFixture", "run_behavior"
    elif name == "ae":
        from experiments.forge import ae_routed_policy_contracts as module
        task = module.make_ae_variant(ROOT)
        producer_name, constructor, entry = "ae_routed_policy_adapters", "AERoutedFixture", "run_behavior"
    else:
        assert name == "word"
        from experiments.forge import word_joint_policy_contracts as module
        task = module.make_variant(ROOT)
        producer_name, constructor, entry = "word_joint_policy_adapters", "WordJointPolicyFixture", "run_word"
    # These are software-only candidates, with the exact named family/pair and
    # original seed. No submitted idea, queue, historical state or GPU is used.
    request = {"candidate": {"id": "software-only-json-runtime-" + name,
        "trainer_family": module.FAMILY, "recipe_preset": "atlas", "task_cohort": module.COHORT,
        "recipe_overrides": {"lr": .0053125, "prior_lr_mult": 1.5}}, "protocol": {"seed": 0}}
    context = task_formulation_context(request["candidate"], task, request["protocol"], device="cpu", root=ROOT)
    compiled = deepcopy(task)
    compiled.update(preflight_blockers=[], field_ownership=context.receipt()["field_ownership"])
    request["tasks"] = {task["id"]: compiled}
    wire = json.loads(json.dumps(request, allow_nan=False))
    canonical = json.loads(json.dumps(task, allow_nan=False))
    producer = importlib.import_module("experiments.forge." + producer_name)
    return wire, {"task_id": task["id"]}, canonical, producer, getattr(producer, constructor), entry


def request_bytes(request):
    return json.dumps(request, sort_keys=True, allow_nan=False).encode()


@pytest.mark.parametrize("name", NAMES)
def test_json_compiled_task_reaches_actual_full_constructor_and_preserves_two_update_prefix(name, monkeypatch, tmp_path):
    request, job, canonical, producer, constructor, entry = wire_request(name)
    wire_before = request_bytes(request)
    task_before = deepcopy(request["tasks"][job["task_id"]])
    rng = torch.get_rng_state().clone()
    clean_context = task_formulation_context(request["candidate"], canonical, request["protocol"], device="cpu", root=ROOT)
    clean = constructor(request, canonical, device="cpu", context=clean_context)
    initial = typed_state_digest(clean.state_dict())
    clean.step()
    first = deepcopy(clean.state_dict())
    clean.step()
    terminal = typed_state_digest(clean.state_dict())
    terminal_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    seen = []

    def structural_probe(actual_request, actual_task, output, device, *, context):
        assert actual_request is request
        assert actual_task == canonical
        assert actual_task is not actual_request["tasks"][job["task_id"]]
        assert not registry.COMPILER_ANNOTATIONS & actual_task.keys()
        assert actual_request["tasks"][job["task_id"]] == task_before
        value = constructor(actual_request, actual_task, device=device, context=context)
        assert context.policy is value.policy
        assert value.max_steps == canonical["execution"]["steps"]
        assert value.max_steps > 2
        assert value.task == canonical
        assert context.policy_task == canonical
        assert typed_state_digest(value.state_dict()) == initial
        value.step()
        assert typed_state_digest(value.state_dict()) == typed_state_digest(first)
        # Exercise the complete real checkpoint loader before the next update.
        value.load_state_dict(deepcopy(first))
        value.step()
        assert typed_state_digest(value.state_dict()) == terminal
        assert torch.equal(torch.get_rng_state(), terminal_rng)
        assert context.receipt()["policy_lifecycle"]["quality_qualification"] is False
        seen.append(value.completed_steps)
        return {"software_only": True, "scientific_qualification": False,
                "completed_steps": value.completed_steps, "external_max_steps": value.max_steps, "cost": {}}

    # Replace only the numerical full-loop entry. All context/Recipe, declaration,
    # source, construction, owners, optimizer, policy and checkpoint code is real.
    monkeypatch.setattr(producer, entry, structural_probe)
    result = adapters.run_task(request, job, tmp_path / name, "cpu")
    assert seen == [2]
    assert result["completed_steps"] == 2 and result["scientific_qualification"] is False
    assert request_bytes(request) == wire_before
    assert not (tmp_path / name).exists()  # No scientific receipt/GIF is forged.


@pytest.mark.parametrize("name, message", [
    ("trajectory", "conditional task changed its original objective, host, initializer, gates or resources"),
    ("unused", "routed task changed its original objective, host, initialization or gates"),
])
def test_original_constructor_boundary_error_is_reproduced_before_models_or_updates(name, message):
    request, job, _, _, constructor, _ = wire_request(name)
    task = request["tasks"][job["task_id"]]
    rng, before = torch.get_rng_state().clone(), request_bytes(request)
    with pytest.raises(ValueError) as error:
        constructor(request, task, device="cpu")
    assert str(error.value) == message
    assert torch.equal(rng, torch.get_rng_state())
    assert request_bytes(request) == before


@pytest.mark.parametrize("name", NAMES)
def test_nonempty_compiled_preflight_refuses_direct_runtime_dispatch(name, monkeypatch, tmp_path):
    request, job, _, producer, _, entry = wire_request(name)
    blockers = ["software-only declared ownership blocker", "software-only source prerequisite missing"]
    request["tasks"][job["task_id"]]["preflight_blockers"] = blockers[:]
    before, rng = request_bytes(request), torch.get_rng_state().clone()
    monkeypatch.setattr(producer, entry, lambda *args, **kwargs: pytest.fail("blocked producer was entered"))
    with pytest.raises(CapabilityError) as error:
        adapters.run_task(request, job, tmp_path / name, "cpu")
    assert error.value.blockers == blockers
    assert request_bytes(request) == before
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("change", ["unknown_top_level", "steps", "threshold", "sampler", "owner", "resources", "source", "cohort"])
def test_runtime_normalization_never_hides_scientific_or_source_mutation(name, change, monkeypatch, tmp_path):
    request, job, _, producer, constructor, entry = wire_request(name)
    task = request["tasks"][job["task_id"]]
    if change == "unknown_top_level": task["runtime_override"] = {"steps": 2}
    elif change == "steps": task["execution"]["steps"] -= 1
    elif change == "threshold": task["evaluation"]["thresholds"][0][2] += 1
    elif change == "sampler": task["evaluation"]["policy_observation"]["sampler"] = "forced_ema"
    elif change == "owner": task["execution"]["policy_contract"]["table_owner"] = "foreign_owner"
    elif change == "resources": task["execution"]["resources"]["batch_size"] += 1
    elif change == "source": task["execution"]["policy_contract"]["sources"]["experiments/forge/adapters.py"] = "0" * 64
    else: task["task_cohort"] = task["execution"]["policy_contract"]["cohort"] = "unknown_policy_v1"
    before, rng = request_bytes(request), torch.get_rng_state().clone()
    constructed = []

    def constructor_probe(actual_request, actual_task, output, device, *, context):
        # Source identities are checked by the untouched strict producer's
        # root-bound validator, even if metadata-only context resolution passed.
        value = constructor(actual_request, actual_task, device=device, context=context)
        constructed.append(value)
        pytest.fail("forged scientific task reached a constructed public owner")

    monkeypatch.setattr(producer, entry, constructor_probe)
    with pytest.raises((ValueError, CapabilityError)):
        adapters.run_task(request, job, tmp_path / name, "cpu")
    assert not constructed
    assert request_bytes(request) == before
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("change", ["blockers_dict", "blockers_non_string", "ownership_list", "ownership_empty",
                                   "wrong_task", "boolean_version", "foreign_key", "nonfinite"])
def test_runtime_rejects_malformed_compiler_annotations_before_producer(change, monkeypatch, tmp_path):
    request, job, _, producer, _, entry = wire_request("word")
    task = request["tasks"][job["task_id"]]
    if change == "blockers_dict": task["preflight_blockers"] = {"runtime_override": "execute"}
    elif change == "blockers_non_string": task["preflight_blockers"] = [True]
    elif change == "ownership_list": task["field_ownership"] = []
    elif change == "ownership_empty": task["field_ownership"] = {}
    elif change == "wrong_task": task["field_ownership"]["task_id"] = "two_pole"
    elif change == "boolean_version": task["field_ownership"]["schema_version"] = True
    elif change == "foreign_key": task["field_ownership"]["runtime_override"] = {"lr": 9.}
    else: task["field_ownership"]["recipe_fields"]["lr"]["value"] = float("nan")
    before = deepcopy(task)
    monkeypatch.setattr(producer, entry, lambda *args, **kwargs: pytest.fail("malformed producer was entered"))
    with pytest.raises(ValueError, match="compiled"):
        adapters.run_task(request, job, tmp_path, "cpu")
    assert typed_state_digest(task) == typed_state_digest(before)


@pytest.mark.parametrize("name, handler", [("vector_two_broad", "_vector"), ("img_intensity2", "_image")])
def test_original_ordinary_dispatch_keeps_wire_task_and_annotations_untouched(name, handler, monkeypatch, tmp_path):
    task = json.loads((ROOT / f"configs/forge/tasks/{name}.json").read_text())
    task.update(preflight_blockers=["software-only ordinary annotation"], field_ownership={"ordinary": True})
    request, job = {"tasks": {task["id"]: task}}, {"task_id": task["id"]}
    before = request_bytes(request)
    sentinel = {"software_only": True, "scientific_qualification": False, "cost": {}}

    def ordinary(actual_request, actual_task, output, device):
        assert actual_request is request and actual_task is task
        assert actual_task["preflight_blockers"] == ["software-only ordinary annotation"]
        assert actual_task["field_ownership"] == {"ordinary": True}
        return sentinel

    monkeypatch.setattr(adapters, handler, ordinary)
    assert adapters.run_task(request, job, tmp_path, "cpu") is sentinel
    assert request_bytes(request) == before

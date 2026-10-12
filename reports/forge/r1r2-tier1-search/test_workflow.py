"""Admission ceilings and observed comparison evidence; no training."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("r1r2_bounded_workflow", Path(__file__).with_name("workflow.py"))
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def manifest():
    return dict(schema_version=1, id="test-round1", round=1, baseline_role="baseline",
        max_candidates=8, paid_cap_seconds=3600,
        campaign=dict(id="test", candidate_budget_seconds=2520, budget_seconds=17640),
        candidates=[dict(role="baseline", candidate="baseline", study="test-study")])


@pytest.mark.parametrize("delta", [dict(round=3), dict(max_candidates=9), dict(paid_cap_seconds=3601)])
def test_finite_round_rejects_overruns(delta):
    value = manifest()
    value.update(delta)
    with pytest.raises(ValueError):
        workflow.validate_manifest(value)


def test_second_round_requires_prior_readout_and_lower_recipe_ceiling():
    value = manifest()
    value.update(round=2, max_candidates=4, paid_cap_seconds=5400)
    value["campaign"]["budget_seconds"] = 10080
    with pytest.raises(ValueError, match="concluded first-round"):
        workflow.validate_manifest(value)
    value.update(previous_registration="round1-registration.json", previous_results="round1-results.json")
    assert workflow.validate_manifest(value) == value


def test_wrong_python_is_rejected_before_preparation_or_admission(monkeypatch):
    value = manifest()
    value["scientific_python_executable"] = "/different/scientific/python"
    monkeypatch.setattr(workflow, "read_json", lambda _path: value)
    with pytest.raises(ValueError, match="scientific Python interpreter"):
        workflow.prepare(SimpleNamespace(manifest="unused"))


def row(role):
    return dict(role=role, task_id="ring16_acquisition", comparison_protocol=dict(schema_version=1,
        initial_tensors_sha256=dict(generator="g", discriminator="d", prior="p"),
        actual_target_batches=dict(critic=dict(batches=1600, samples=409600, sha256="b"),
                                  generator=dict(batches=1600, samples=409600, sha256="b")),
        rng_checkpoints=dict(all_named_streams_checkpointed=True)))


@pytest.mark.parametrize("field", ["initial_tensors_sha256", "actual_target_batches"])
def test_actual_comparison_mismatch_cannot_be_reported_as_matched(field):
    a, b = row("control"), row("candidate")
    b["comparison_protocol"][field] = {}
    with pytest.raises(ValueError, match="Actual comparison condition changed"):
        workflow.comparison_audit([a, b])


def test_missing_actual_batch_evidence_is_explicitly_unavailable():
    a, b = row("control"), row("candidate")
    b.pop("comparison_protocol")
    result = workflow.comparison_audit([a, b])[0]
    assert result["status"] == "UNAVAILABLE"
    assert result["missing_roles"] == ["candidate"]


def test_every_consumed_stream_must_be_checkpointed():
    value = row("candidate")
    value["comparison_protocol"]["rng_checkpoints"]["all_named_streams_checkpointed"] = False
    with pytest.raises(ValueError, match="consumed stream"):
        workflow.comparison_audit([row("control"), value])


def test_target_data_agreement_does_not_require_matching_trained_states():
    a, b = row("control"), row("candidate")
    a["checkpoint"], b["checkpoint"] = {"state_sha256": "a"}, {"state_sha256": "b"}
    assert workflow.comparison_audit([a, b])[0]["status"] == "PASS"


def test_paid_cap_counts_all_outstanding_full_reservations(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.Queue, "_available_budget", lambda *_a, **_kw: (True, None))
    queue = workflow.BoundedQueue(tmp_path, paid_cap_seconds=3600)
    state = dict(charges=[dict(seconds=2200)], jobs={"running": dict(reserved_seconds=500)})
    assert queue._available_budget(state, {}, 900) == (True, None)
    assert queue._available_budget(state, {}, 901)[0] is False
    state["charges"].append(dict(seconds=1))
    assert queue._available_budget(state, {}, 900)[0] is False


def test_paid_cap_retains_original_task_budget(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(workflow.Queue, "_available_budget", lambda _self, _state, _request, seconds, **_kw:
                        (observed.append(seconds) or True, None))
    queue = workflow.BoundedQueue(tmp_path, paid_cap_seconds=3600)
    request = dict(original_timeout=900)
    assert queue._available_budget(dict(charges=[], jobs={}), request, 900)[0]
    assert observed == [900] and request == dict(original_timeout=900)


def test_clock_proof_keeps_its_distinct_scope():
    audit = workflow.comparison_audit([dict(role="candidate", task_id="clockfree_audit_measurement_v1")])
    assert audit[0]["status"] == "SEPARATE_CLOCK_PROOF"


def test_candidate_ownership_annotations_do_not_change_the_frozen_task():
    a = dict(id="test", execution=dict(steps=1600), field_ownership=dict(lr=.0085), preflight_blockers=[])
    b = dict(id="test", execution=dict(steps=1600), field_ownership=dict(lr=.00085), preflight_blockers=[])
    assert workflow.task_contract_hash(a) == workflow.task_contract_hash(b)
    b["execution"]["steps"] = 6400
    assert workflow.task_contract_hash(a) != workflow.task_contract_hash(b)


def test_source_binding_refresh_preserves_every_scientific_task_field():
    receipt = workflow.read_json(Path(__file__).with_name("protocol-source-bindings.json"))
    for task in receipt["tasks"]:
        old = subprocess.check_output(["git", "show", f'{receipt["source_commit_before_refresh"]}:{task["path"]}'],
                                      cwd=workflow.ROOT)
        import hashlib
        assert hashlib.sha256(old).hexdigest() == task["before_file_sha256"]
        current = workflow.read_json(workflow.ROOT / task["path"])
        previous = json.loads(old)
        previous["evaluation"].pop("sources")
        current["evaluation"].pop("sources")
        assert previous == current
        assert workflow.stable_hash(current) == task["before_scientific_law_sha256"] == task["after_scientific_law_sha256"]


def charged_attempt(aid, request, seconds):
    return dict(attempt_id=aid, seconds=seconds, owner=dict(request=request, campaign=request, revision="recipe"))


def test_cached_prior_round_attempt_is_validated_without_a_second_charge():
    state = dict(charges=[charged_attempt("baseline", "round1", 12), charged_attempt("new", "round2", 18)])
    charges = workflow.paid_charge_index(state)
    old = workflow.attempt_cost(charges, {"round2"}, "baseline", 300,
                               dict(task_results=[dict(cost=dict(wall_seconds=12))]))
    new = workflow.attempt_cost(charges, {"round2"}, "new", 300,
                               dict(task_results=[dict(cost=dict(wall_seconds=18))]))
    assert old["reused"] is True and old["seconds"] == 0
    assert old["original_paid_seconds"] == 12 and old["original_charge_owner"]["request"] == "round1"
    assert new["reused"] is False and new["seconds"] == 18
    assert sum(charge["seconds"] for charge in charges.values()) == 30
    assert sum(cost["seconds"] for cost in (old, new)) == 18


def test_cached_attempt_still_requires_exact_original_paid_receipt():
    charges = workflow.paid_charge_index(dict(charges=[charged_attempt("baseline", "round1", 12)]))
    with pytest.raises(ValueError, match="Paid accounting mismatch"):
        workflow.attempt_cost(charges, {"round2"}, "baseline", 300,
                              dict(task_results=[dict(cost=dict(wall_seconds=13))]))


def test_aggregate_accounting_rejects_duplicate_charges_outside_current_round():
    charge = charged_attempt("baseline", "round1", 12)
    with pytest.raises(ValueError, match="Duplicate paid charge"):
        workflow.paid_charge_index(dict(charges=[charge, charge]))


def test_external_certified_reuse_has_no_local_charge():
    cost = workflow.attempt_cost({}, {"round2"}, "external", 300,
                                 dict(task_results=[dict(cost=dict(wall_seconds=12))]))
    assert cost["reused"] is True and cost["seconds"] == 0
    assert cost["original_paid_seconds"] is None and cost["original_charge_owner"] is None


def test_round2_children_change_only_global_learning_rate():
    value = workflow.read_json(Path(__file__).with_name("phase2.json"))
    assert value["max_candidates"] == 4 and len(value["searches"]) == 3
    for search in value["searches"]:
        sid = search["id"]
        spec = workflow.read_json(workflow.ROOT / f"configs/forge/searches/{sid}.json")
        assert spec["schema_version"] == 2 and spec["grid"] == {"lr": [.00425]}
        parent = workflow.read_json(workflow.ROOT / f'configs/forge/configurations/{spec["base_candidate"]}.json')
        child = next(c for c in workflow.read_json(Path(__file__).with_name("round2-registration.json"))["candidates"].values()
                     if c["card"].get("search") == sid)
        card = workflow.read_json(workflow.ROOT / f'configs/forge/configurations/{child["candidate"]}.json')
        before, after = parent["resolved_configuration_recipe"], card["resolved_configuration_recipe"]
        assert {key for key in before if before[key] != after[key] and key != "name"} == {"lr"}
        assert before["lr"] == .0085 and after["lr"] == .00425


def test_round2_cached_baseline_keeps_exact_scientific_and_job_identity():
    prior = workflow.read_json(Path(__file__).with_name("registration.json"))
    followup = workflow.read_json(Path(__file__).with_name("round2-registration.json"))
    before, after = prior["candidates"]["baseline"], followup["candidates"]["baseline"]
    assert {key: val for key, val in before.items() if key != "card"} == {
        key: val for key, val in after.items() if key != "card"}
    assert followup["source_digest"] == prior["source_digest"]

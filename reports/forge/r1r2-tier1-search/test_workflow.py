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

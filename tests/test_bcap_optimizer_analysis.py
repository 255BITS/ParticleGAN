"""Reporting safeguards for the frozen optimizer screen; no training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "reports/forge/dualnorm-tier1/analyze.py"
SPEC = importlib.util.spec_from_file_location("bcap_optimizer_analysis", MODULE)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def trial(config, passed=(), incomplete=()):
    return {"candidate_id": "candidate-" + config, "configuration_id": config,
            "submission_status": "completed", "source_digest": analysis.SOURCE_DIGEST,
            "tasks": [{"task": name, "qualification_tier": 1,
                       "importance": "required" if name in analysis.REQUIRED else "diagnostic",
                       "gate_status": "PASS" if name in passed else "INCOMPLETE" if name in incomplete else "FAIL",
                       "compatibility_key": name + config}
                      for name in (*analysis.REQUIRED, analysis.DIAGNOSTIC)]}


def test_selection_uses_whole_config_count_then_hash_and_keeps_errors():
    first = trial("a", passed=analysis.REQUIRED[:2])
    second = trial("b", passed=analysis.REQUIRED[2:4], incomplete=[analysis.REQUIRED[5]])
    selected = analysis.whole_selection([second, first])
    assert selected["selected_configuration_id"] == "a"
    assert selected["required_pass_count"] == 2
    assert selected["required_total"] == 6
    assert not selected["qualified"]
    assert selected["selection_kind"] == "best_observed"
    # The error stays an unknown scientific outcome in the same denominator.
    assert second["tasks"][5]["gate_status"] == "INCOMPLETE"


@pytest.mark.parametrize("pending_name", [analysis.REQUIRED[-1], analysis.DIAGNOSTIC])
def test_no_selection_before_every_independent_current_tier_peer_finishes(pending_name):
    complete = trial("a", passed=analysis.REQUIRED[:2])
    pending = trial("b", passed=analysis.REQUIRED[:3])
    next(task for task in pending["tasks"] if task["task"] == pending_name)["gate_status"] = "UNKNOWN"
    assert not analysis.execution_complete(pending)
    selected = analysis.whole_selection([complete, pending])
    assert selected["selected_candidate_id"] is None
    assert selected["selection_kind"] == "pending"
    assert not selected["qualified"]


def test_rng_device_indices_do_not_create_a_false_stream_difference():
    binding = {"family": "data", "component": "target", "purpose": "training",
               "seed": 7, "initial_state_sha256": "cuda-state", "device": "cuda:0"}
    first = {"seed": 0, "version": "forge-rng-v1", "bindings": {"unused-key": binding}}
    second = deepcopy(first)
    second["bindings"]["unused-key"]["device"] = "cuda:1"
    assert analysis.canonical_rng(first) == analysis.canonical_rng(second)
    second["bindings"]["unused-key"]["device"] = "cpu"
    assert analysis.canonical_rng(first) != analysis.canonical_rng(second)
    second = deepcopy(first)
    second["bindings"]["unused-key"]["initial_state_sha256"] = "another-state"
    assert analysis.canonical_rng(first) != analysis.canonical_rng(second)


def test_trace_excludes_unreliable_input_gradient_fields_even_if_nonfinite():
    row = {"step": 1, "optimizer": "D", "layers": [], "players": {},
           "critic_log_spectral_product": 2., "critic_input_gradient_mean_real": float("nan"),
           "critic_input_gradient_mean_fake": 100., "critic_spectral_product": 7.4}
    safe = analysis.valid_trace([row])
    assert safe == [{"step": 1, "optimizer": "D", "layers": [], "players": {},
                    "critic_log_spectral_product": 2.}]
    row["critic_log_spectral_product"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        analysis.valid_trace([row])


def test_missing_error_audits_are_not_counted_as_initialization_or_stream_matches():
    request = {"tasks": {"task": {"execution": {"steps": 80}, "evaluation": {"weights": "live"}}}}
    observed = [("failed", {}, (request, {}, {}, {"error": {"type": "ValueError"}}))]
    audit = analysis.audit_task("task", observed, 41)
    assert audit["certified_receipts"] == 1
    assert audit["expected_configurations"] == 41
    assert audit["initialization"] == {}
    assert audit["stream_starts"] == {}
    assert audit["guards"] == {}
    assert audit["missing_training_audit_candidates"] == ["failed"]


def test_explicit_zero_fixture_is_retained_as_its_own_initialization_contract():
    fixture = {"critic": "stored_host_weights", "particles": "zeros"}
    request = {"tasks": {"task": {"execution": {"steps": 80, "fixed_initialization": fixture},
                                      "evaluation": {"weights": "live"}}}}
    row = {"applied": {"initialization": {}, "rng": {"seed": 0, "version": "forge-rng-v1", "bindings": {}}},
           "evidence": {"guards": {"all_finite": True, "optimizer_updates": {"prior": 80}}}}
    audit = analysis.audit_task("task", [("fixed", {}, (request, {}, {}, row))], 1)
    assert audit["declared_fixed_initialization"] == fixture
    assert audit["missing_training_audit_candidates"] == []
    assert audit["initialization"] == {}
    assert audit["optimizer_update_counts"] == {"prior": {"receipts": 1, "distinct_counts": [80]}}


def test_metadata_only_initializer_never_establishes_an_aggregate_state_match():
    request = {"tasks": {"task": {"execution": {"steps": 80}, "evaluation": {"weights": "live"}}}}
    row = {"initialization": {"generator": {"initializer": "named_parameters", "parameter_seeds": {"w": 7},
                                                "parameters": {"w": {"sha256": "a" * 64}}}},
           "rng": {"seed": 0, "version": "forge-rng-v1", "bindings": {}},
           "evidence": {"guards": {"all_finite": True}}}
    observed = [("metadata", {}, (request, {}, {}, row))]
    audit = analysis.audit_task("task", observed, 1)
    generator = audit["initialization"]["generator"]
    assert generator["component_metadata_receipts"] == 1
    assert generator["receipts"] == 0
    assert generator["state_hashes"] == []
    assert not generator["matched_present"]
    assert not generator["matched_all_component_receipts"]
    assert not generator["complete_for_expected_configurations"]
    assert generator["missing_aggregate_hash_candidates"] == ["metadata"]
    assert audit["missing_training_audit_candidates"] == ["metadata"]


def test_missing_hash_is_explicit_beside_a_present_component_state_hash():
    request = {"tasks": {"task": {"execution": {"steps": 80}, "evaluation": {"weights": "live"}}}}
    row = {"initialization": {"generator": {"initial_state_sha256": "a" * 64}},
           "rng": {"seed": 0, "version": "forge-rng-v1", "bindings": {}},
           "evidence": {"guards": {"all_finite": True}}}
    missing = deepcopy(row)
    missing["initialization"]["generator"] = {"initializer": "named_parameters"}
    observed = [("present", {}, (request, {}, {}, row)), ("missing", {}, (request, {}, {}, missing))]
    generator = analysis.audit_task("task", observed, 2)["initialization"]["generator"]
    assert generator["matched_present"]
    assert not generator["matched_all_component_receipts"]
    assert generator["state_hashes"] == ["a" * 64]
    assert generator["missing_aggregate_hash_candidates"] == ["missing"]


def test_missing_metrics_do_not_get_imputed_from_adam():
    assert analysis.numeric_delta({}, {"hq": .9}) == {}
    assert analysis.numeric_delta({"modes": 9, "finite": True}, {"modes": 10, "finite": False}) == {"modes": -1}


def test_damaged_receipt_retains_invalid_status_without_attributing_metrics(tmp_path):
    attempt = tmp_path / "reports/forge/attempts/a"
    attempt.mkdir(parents=True)
    for name, value in (("request", {}), ("result", {}), ("evidence", {"result_hash": "tampered"})):
        (attempt / (name + ".json")).write_text(json.dumps(value))
    reader = analysis.Artifacts(tmp_path, tmp_path / "queue")
    candidate = {"candidate_id": "candidate", "candidate_revision": "revision"}
    task = {"task": "task", "attempt_id": "a", "gate_status": "INVALID", "metrics": {"hq": .9}}
    assert reader.receipt(candidate, task) is None
    compact = analysis.compact_task(task, None)
    assert compact["gate_status"] == "INVALID"
    assert compact["metrics"] == {}
    task["gate_status"] = "PASS"
    with pytest.raises(ValueError, match="binding mismatch"):
        reader.receipt(candidate, task)


def repair_fixture():
    candidate = trial("a")
    first, second = candidate["tasks"][:2]
    first.update(attempt_id="replacement", gate_status="PASS")
    second.update(attempt_id="numerical", gate_status="INCOMPLETE")
    request = {"candidate_revision": "revision", "source": {"digest": "source"},
               "protocol": {"seed": 0}, "runtime": {}, "execution_backend": "cuda"}
    def result(identity, task, gate, status, retry=None):
        row = {"task_id": task["task"], "compatibility_key": task["compatibility_key"], "gate_status": gate}
        value = {"attempt_id": identity, "raw": {"attempt_status": status}, "task_results": [row]}
        if retry:
            value["retry_of"] = retry
        return value
    original = result("original", first, "INCOMPLETE", "error")
    link = {"attempt_id": "original", "result_hash": analysis.stable_hash(original),
            "reason": "Restore interrupted execution environment", "authorized_at": "2026-10-05T19:35:12Z"}
    replacement = result("replacement", first, "PASS", "completed", link)
    numerical = result("numerical", second, "INCOMPLETE", "error")
    results = {"original": original, "replacement": replacement, "numerical": numerical}
    receipts = {name: (deepcopy(request), value, {"result_hash": analysis.stable_hash(value)}, value["task_results"][0])
                for name, value in results.items()}
    candidate["attempt_history"] = [{"attempt_id": name, "result_hash": analysis.stable_hash(value),
        "valid_receipt": True, "superseded_by": "replacement" if name == "original" else None,
        "gate_statuses": {row["task_id"]: row["gate_status"] for row in value["task_results"]},
        "wall_seconds": cost} for (name, value), cost in zip(results.items(), (1., 2., 3.))]
    return candidate, receipts


def test_repaired_attempt_does_not_add_a_current_cell_or_hide_original_cost():
    candidate, receipts = repair_fixture()
    history = analysis.history_summary([candidate], None, receipts)
    assert history["current_cells"] == 7
    assert history["current_selected_attempts"] == 2
    assert history["recorded_attempts"] == 3
    assert history["superseded_original_attempts"] == 1
    assert history["all_attempt_wall_seconds"] == 6
    assert history["superseded_attempt_wall_seconds"] == 1
    repair = history["execution_repairs"][0]
    assert repair["original_attempt"] == "original"
    assert repair["replacement_attempt"] == "replacement"
    assert repair["scientific_identity_preserved"]
    numerical = next(row for row in history["attempts"] if row["attempt_id"] == "numerical")
    assert numerical["selected_current"]
    assert numerical["gate_statuses"] == {analysis.REQUIRED[1]: "INCOMPLETE"}
    assert numerical["superseded_by"] is None


@pytest.mark.parametrize("changed_contract", ["scientific_failure", "seed"])
def test_retry_history_rejects_scientific_failure_replacement_or_seed_change(changed_contract):
    candidate, receipts = repair_fixture()
    if changed_contract == "scientific_failure":
        original = receipts["original"][1]
        original["task_results"][0]["gate_status"] = "FAIL"
        original_hash = analysis.stable_hash(original)
        receipts["original"][2]["result_hash"] = original_hash
        receipts["replacement"][1]["retry_of"]["result_hash"] = original_hash
        candidate["attempt_history"][0]["result_hash"] = original_hash
        replacement_hash = analysis.stable_hash(receipts["replacement"][1])
        receipts["replacement"][2]["result_hash"] = replacement_hash
        candidate["attempt_history"][1]["result_hash"] = replacement_hash
    else:
        receipts["replacement"][0]["protocol"]["seed"] = 1
    with pytest.raises(ValueError, match="invalid execution-repair lineage"):
        analysis.history_summary([candidate], None, receipts)

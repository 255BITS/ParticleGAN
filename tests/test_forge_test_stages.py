"""Stage measurements stay source-bound and never manufacture suite qualification."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from experiments.forge.contracts import read_json, stable_hash
from experiments.forge import test_stages as stages
from experiments.forge.views import task_execution_fingerprint, task_evaluation_fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = "dacf96749a4effc0648a7a0945266d508804110e"


def fixture(stage_id, task_id=None):
    registry, tasks = stages.load_registry(ROOT)
    stage = next(row for row in registry["stages"] if row["id"] == stage_id)
    task = tasks[task_id or stage["source_tasks"][0]]
    source = {"id": "source-test", "task_id": task["id"], "lane": "Forge", "definition_match": "matches",
              "conditions": {key: "a" * 64 for key in
                             ("task_contract_sha256", "protocol_sha256", "rng_sha256", "source_digest")},
              "contract": {"execution_sha256": task_execution_fingerprint(task),
                           "evaluation_sha256": task_evaluation_fingerprint(task)}, "cells": {"recipe": ["record"]}}
    source["conditions"].update(protocol={"seed": 0}, runtime={"execution_backend": "cpu"})
    source["conditions"]["protocol_sha256"] = stable_hash(source["conditions"]["protocol"])
    source["conditions"]["task_contract_sha256"] = stable_hash(source["contract"])
    record = {"status": "PASS", "evidence_available": True, "receipt": "reports/forge/technique-receipts/old.json",
              "receipt_sha256": "b" * 64,
              "result": {"compatibility_key": "c" * 64,
                         "cost": {"completed_steps": task["execution"]["steps"]},
                         "sampling": {key: task["evaluation"][key] for key in
                                      ("sampling_contract_version", "sampling_law", "eval_output_noise")},
                         "metrics": {}, "evaluator_summary": {}}}
    return stage, task, source, record


def derive(values):
    return stages.derive_stage(*values)


def native_metrics(n):
    return {"n": n, "valid_n": n, "all_finite": True, "frozen_pass": True,
            "cov_trace_bias": 0., "abs_cov_trace_bias": 0., "mass_tv": 0.,
            "center_rms_sigma": 0., "radial_ks": 0.}


def native_fixture(stage_id):
    values = fixture(stage_id)
    summary = values[-1]["result"]["evaluator_summary"]
    summary.update(coverage_status="PASS", holdout_metrics=native_metrics(100000),
                   oracle_metrics=native_metrics(100000), terminal_checks=[
                       {"step": step, "metrics": native_metrics(20000), "passed": True}
                       for step in (6000, 6250, 6500, 6750, 7000)])
    return values


def test_original_combined_contracts_and_budgets_remain_exact():
    registry, tasks = stages.load_registry(ROOT)
    for name, pin in registry["sources"].items():
        original = subprocess.check_output(["git", "show", BASE + ":" + pin["path"]], cwd=ROOT)
        assert (ROOT / pin["path"]).read_bytes() == original
    assert tasks["gaussian1d_stability"]["execution"]["steps"] == 6000
    assert tasks["grid100"]["execution"]["steps"] == 7000
    assert tasks["five_word_joint_hold"]["execution"]["steps"] == 4000
    view = read_json(ROOT / "configs/forge/views/discriminator_stability.json")
    assert next(a for a in view["assignments"] if a["task"] == "five_word_joint_hold")["qualification_tier"] == 3


@pytest.mark.parametrize("stage_id", ["gaussian_quality", "gaussian_stationary_retention", "gaussian_shift_recovery",
                                     "native100_quality", "native100_terminal_retention", "word_quality", "word_retention"])
def test_aggregate_pass_does_not_prove_any_split_stage(stage_id):
    values = fixture(stage_id)
    assert values[-1]["status"] == "PASS"
    assert derive(values)["status"] == "UNMEASURED"


def test_gaussian_quality_requires_confirmed_endpoint_not_earlier_acquisition():
    values = fixture("gaussian_quality")
    result = values[-1]["result"]
    result["metrics"] = {"step": 1000, "sample_count": 4096, "finite_fraction": 1.,
                         "mean_error_sigma": 0., "std_ratio": 1., "cdf_ks": .01}
    result["evaluator_summary"]["confirmed_steps"] = [42]
    assert derive(values)["status"] == "FAIL"
    result["evaluator_summary"]["confirmed_steps"].append(1000)
    assert derive(values)["status"] == "PASS"
    result["metrics"]["cdf_ks"] = .050001
    assert derive(values)["status"] == "FAIL"


def test_gaussian_retention_and_recovery_are_independent_components_of_a_combined_failure():
    stationary = fixture("gaussian_stationary_retention")
    shifted = fixture("gaussian_shift_recovery")
    for values in (stationary, shifted):
        values[-1]["status"] = "FAIL"
        values[-1]["result"]["evaluator_summary"] = {
            "stationary_checks": 72, "stationary_passes": 71, "reacquisition_status": "PASS",
            "shift_hold_checks": 24, "shift_hold_passes": 24, "frozen_checks": 48, "frozen_passes": 48}
    assert derive(stationary)["status"] == "FAIL"
    assert derive(shifted)["status"] == "PASS"  # Frozen-control pass count was diagnostic in the original gate.
    shifted[-1]["result"]["evaluator_summary"]["shift_hold_passes"] = 23
    assert derive(shifted)["status"] == "FAIL"
    stationary[-1]["result"]["evaluator_summary"]["stationary_passes"] = 73
    assert derive(stationary)["status"] == "INVALID"


def test_native_quality_does_not_promote_failed_retention_and_ignores_pass_stamps():
    quality, retention = native_fixture("native100_quality"), native_fixture("native100_terminal_retention")
    for values in (quality, retention):
        values[-1]["status"] = "FAIL"
        values[-1]["result"]["evaluator_summary"]["terminal_checks"][1]["metrics"]["radial_ks"] = .040001
    assert derive(quality)["status"] == "PASS"
    assert derive(retention)["status"] == "FAIL"
    quality[-1]["result"]["evaluator_summary"]["holdout_metrics"]["center_rms_sigma"] = .200001
    assert derive(quality)["status"] == "FAIL"


def test_native_retention_needs_all_original_steps_live_bounds_and_complete_coverage():
    values = native_fixture("native100_terminal_retention")
    summary = values[-1]["result"]["evaluator_summary"]
    assert derive(values)["status"] == "PASS"
    summary["terminal_checks"][0]["step"] = 5999
    assert derive(values)["status"] == "UNMEASURED"
    summary["terminal_checks"][0]["step"] = 6000
    summary["coverage_status"] = "FAIL"
    assert derive(values)["status"] == "FAIL"
    summary["coverage_status"] = "PASS"
    summary["terminal_checks"][0]["metrics"]["frozen_pass"] = False
    assert derive(values)["status"] == "FAIL"


def test_native_quality_needs_independent_original_holdout_and_valid_oracle():
    values = native_fixture("native100_quality")
    summary = values[-1]["result"]["evaluator_summary"]
    assert derive(values)["status"] == "PASS"
    summary["holdout_metrics"]["n"] = 20000
    assert derive(values)["status"] == "INVALID"
    summary["holdout_metrics"] = native_metrics(100000)
    summary["oracle_metrics"]["all_finite"] = False
    assert derive(values)["status"] == "INVALID"


def test_word_quality_and_all_check_retention_have_different_evidence_requirements():
    quality, retention = fixture("word_quality"), fixture("word_retention")
    quality[-1]["result"]["evaluator_summary"] = {
        "confirmed_steps": [834], "first_confirmed_step": 834, "confirmed_checks": 1, "observations": 24}
    assert derive(quality)["status"] == "PASS"
    retention[-1]["result"]["evaluator_summary"] = {"confirmed_checks": 24, "observations": 25}
    assert derive(retention)["status"] == "FAIL"
    retention[-1]["result"]["evaluator_summary"]["confirmed_checks"] = 25
    assert derive(retention)["status"] == "PASS"
    quality[-1]["result"]["evaluator_summary"]["confirmed_steps"] = [835]
    assert derive(quality)["status"] == "INVALID"


@pytest.mark.parametrize("mutation", [
    lambda source, record: source.update(lane="API"),
    lambda source, record: source.update(definition_match="CHANGED"),
    lambda source, record: source["conditions"].pop("rng_sha256"),
    lambda source, record: source["conditions"]["protocol"].update(seed=1),
    lambda source, record: source["conditions"].update(runtime={}),
    lambda source, record: source["contract"].update(execution_sha256="d" * 64),
    lambda source, record: record.update(receipt_sha256=None),
    lambda source, record: record.update(status="INVALID"),
    lambda source, record: record["result"]["sampling"].update(eval_output_noise="noisy"),
    lambda source, record: record["result"]["cost"].update(completed_steps=6999),
])
def test_stage_cells_cannot_cross_source_protocol_runtime_or_incomplete_execution(mutation):
    values = native_fixture("native100_quality")
    mutation(values[2], values[3])
    assert derive(values)["status"] == "UNMEASURED"


def test_fixed_rotated_target_cannot_fill_unfrozen_rotation_recovery():
    registry, _ = stages.load_registry(ROOT)
    rotation = next(row for row in registry["stages"] if row["id"] == "native100_rotation_recovery")
    assert rotation["source_tasks"] == [] and rotation["protocol_status"] == "needs_definition"
    assert stages.derive_stage(rotation, None, {}, {"status": "PASS"})["status"] == "UNMEASURED"


def test_stage_report_preserves_cohort_keys_and_never_changes_combined_records():
    values = native_fixture("native100_quality")
    _, _, source, record = values
    performance = {"qualification_input": False, "recipes": [{"id": "recipe", "selected": True}],
                   "tests": [source], "records": {"record": record}}
    performance["input_digest"] = stable_hash(performance)
    before = deepcopy(performance)
    report = stages.build_report(ROOT, performance=performance)
    assert performance == before
    assert report["qualification_input"] is report["qualification_reuse"] is report["training_launched"] is False
    assert report["additional_training_updates"] == 0
    stage = next(row for row in report["stages"] if row["id"] == "native100_quality")
    cohort = next(row for row in stage["cohorts"] if row["source_task"] == "grid100")
    assert cohort["conditions"] == source["conditions"]
    assert cohort["cells"][0]["record"] == "record"
    assert cohort["cells"][0]["receipt_sha256"] == record["receipt_sha256"]
    assert cohort["cells"][0]["status"] == "PASS"
    performance["records"]["record"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="verified"):
        stages.build_report(ROOT, performance=performance)


def test_committed_stage_reports_match_the_source_bound_generator():
    outputs = stages.generated_reports(ROOT)
    for path, text in outputs.items():
        assert path.read_text() == text
    report = json.loads(outputs[ROOT / "reports/forge/test-stages.json"])
    assert report["additional_training_updates"] == 0
    assert "fixed rotated target" in outputs[ROOT / "reports/forge/test-stages.md"]

"""Separate source-bound quality and retention measurements without regrading a suite.

The original runner, combined evaluator, receipts and qualification rows are
unchanged. Only independently sufficient, already audited submetrics can fill
a stage. Aggregate PASS/FAIL strings cannot fill a missing stage measurement.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from .contracts import file_hash, read_json, stable_hash
from .views import task_execution_fingerprint, task_evaluation_fingerprint

REGISTRY = Path("configs/forge/test-stages.json")
PREFIX = Path("reports/forge/test-stages")
KINDS = {"gaussian_endpoint", "gaussian_stationary", "gaussian_shift",
         "native_holdout", "native_terminal", "word_confirmed", "word_hold", "needs_protocol"}


def load_registry(root):
    root = Path(root)
    registry = read_json(root / REGISTRY)
    if (registry.get("schema_version") != 1 or registry.get("revision") != 1
            or registry.get("qualification_input") is not False
            or registry.get("execution_policy") != "shared_original_runs_no_additional_updates"):
        raise ValueError("test stages require the nonqualifying shared-run contract")
    tasks = {}
    for name, pin in registry["sources"].items():
        if pin["path"] != f"configs/forge/tasks/{name}.json":
            raise ValueError("test stage source path differs from task identity")
        task = read_json(root / pin["path"])
        if (task["id"] != name or file_hash(root / pin["path"]) != pin["sha256"]
                or task_execution_fingerprint(task) != pin["execution_sha256"]
                or task_evaluation_fingerprint(task) != pin["evaluation_sha256"]):
            raise ValueError("test stage source contract changed: " + name)
        tasks[name] = task
    ids = set()
    kinds = {"gaussian_endpoint": "gaussian_smoke", "gaussian_stationary": "gaussian_stability",
             "gaussian_shift": "gaussian_stability", "native_holdout": "native_accuracy",
             "native_terminal": "native_accuracy", "word_confirmed": "word_smoke", "word_hold": "word_hold"}
    for stage in registry["stages"]:
        if (stage["id"] in ids or stage["kind"] not in KINDS
                or type(stage["qualification_tier"]) is not int
                or stage["qualification_tier"] not in {2, 3}
                or len(set(stage["source_tasks"])) != len(stage["source_tasks"])
                or any(name not in tasks for name in stage["source_tasks"])):
            raise ValueError("invalid or duplicate test stage")
        if stage["kind"] == "needs_protocol":
            if stage["source_tasks"] or stage["protocol_status"] != "needs_definition":
                raise ValueError("undefined rotation recovery cannot inherit a fixed-target result")
        elif not stage["source_tasks"] or stage["protocol_status"] != "defined":
            raise ValueError("defined stages require exact original task sources")
        elif (any(tasks[name]["evaluation"]["kind"] != kinds[stage["kind"]] for name in stage["source_tasks"])
              or stage["qualification_tier"] != (2 if stage["kind"] in
                  {"gaussian_endpoint", "native_holdout", "word_confirmed"} else 3)):
            raise ValueError("stage capability or source evaluator differs from its contract")
        ids.add(stage["id"])
    return registry, tasks


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _grade(status, reason, **details):
    return {"status": status, "reason": reason, "qualification_input": False,
            "qualification_reuse": False, **details}


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def _bounds(metrics, thresholds):
    if any(name not in metrics for name, _, _ in thresholds):
        return None
    if any(not _finite(metrics[name]) for name, _, _ in thresholds):
        return False
    return all(metrics[name] >= bound if op == ">=" else metrics[name] <= bound if op == "<="
               else metrics[name] == bound for name, op, bound in thresholds)


def _count(summary, passes, checks, expected):
    if passes not in summary or checks not in summary:
        return None
    if (type(summary[passes]) is not int or type(summary[checks]) is not int
            or summary[checks] != expected or not 0 <= summary[passes] <= expected):
        raise ValueError("invalid stage observation counts")
    return summary[passes] == expected


def _native_quality(metrics, task, sample_count):
    """Use audited coverage submetrics and recompute all original fidelity bounds."""
    required = {"n", "valid_n", "all_finite", "frozen_pass", "cov_trace_bias"}
    required.update(task["evaluation"]["accuracy_limits"])
    if required - metrics.keys():
        return None
    if (type(metrics["n"]) is not int or type(metrics["valid_n"]) is not int
            or metrics["n"] != sample_count or type(metrics["all_finite"]) is not bool
            or type(metrics["frozen_pass"]) is not bool):
        raise ValueError("native stage sample or coverage audit differs from the original protocol")
    if not _finite(metrics["cov_trace_bias"]):
        return False
    bounds = [[name, "<=", limit] for name, limit in task["evaluation"]["accuracy_limits"].items()]
    bounded = _bounds(metrics, bounds)
    return bool(bounded and all(_finite(metrics[name]) and metrics[name] >= 0
                               for name in task["evaluation"]["accuracy_limits"])
                and metrics["valid_n"] == sample_count and metrics["all_finite"]
                and metrics["frozen_pass"] and math.isclose(metrics["abs_cov_trace_bias"],
                    abs(metrics["cov_trace_bias"]), abs_tol=1e-12))


def derive_stage(stage, task, source_test, record):
    """Consume a verified test_performance record, keeping its original cohort key."""
    if stage["kind"] == "needs_protocol":
        return _grade("UNMEASURED", "Rotation event, deadline and continuation protocol are not frozen.")
    conditions, contract = source_test.get("conditions", {}), source_test.get("contract") or {}
    result = record.get("result", {})
    sampling = result.get("sampling", {})
    if (source_test.get("lane") != "Forge" or source_test.get("definition_match") != "matches"
            or source_test.get("task_id") != task["id"] or task["id"] not in stage["source_tasks"]
            or contract.get("execution_sha256") != task_execution_fingerprint(task)
            or contract.get("evaluation_sha256") != task_evaluation_fingerprint(task)
            or not record.get("evidence_available") or record.get("status") not in {"PASS", "FAIL"}
            or not record.get("receipt") or not _digest(record.get("receipt_sha256"))
            or not _digest(result.get("compatibility_key"))
            or any(not _digest(conditions.get(key)) for key in
                   ("task_contract_sha256", "protocol_sha256", "rng_sha256", "source_digest"))
            or conditions.get("task_contract_sha256") != stable_hash(contract)
            or not isinstance(conditions.get("protocol"), dict)
            or conditions.get("protocol_sha256") != stable_hash(conditions.get("protocol"))
            or not isinstance(conditions.get("runtime"), dict) or not conditions["runtime"]
            or any(sampling.get(key) != task["evaluation"][key] for key in
                   ("sampling_contract_version", "sampling_law", "eval_output_noise"))):
        return _grade("UNMEASURED", "No complete compatible source-bound submetric receipt.")
    completed = result.get("cost", {}).get("completed_steps")
    if type(completed) is not int or completed < task["execution"]["steps"]:
        return _grade("UNMEASURED", "Original full-budget execution is not evidenced.")
    summary = result.get("evaluator_summary", {})
    metrics = result.get("metrics", {})
    kind = stage["kind"]
    try:
        if kind == "gaussian_endpoint":
            endpoint = task["execution"]["steps"]
            confirmations = summary.get("confirmed_steps")
            if (not isinstance(confirmations, list) or metrics.get("step") != endpoint
                    or any(type(step) is not int for step in confirmations)):
                passed = None
            else:
                bounded = _bounds(metrics, task["evaluation"]["thresholds"])
                passed = None if bounded is None else bounded and endpoint in confirmations
        elif kind == "gaussian_stationary":
            passed = _count(summary, "stationary_passes", "stationary_checks",
                            task["evaluation"]["stationary_checks"])
        elif kind == "gaussian_shift":
            held = _count(summary, "shift_hold_passes", "shift_hold_checks",
                          task["evaluation"]["shift_hold_checks"])
            recovery = summary.get("reacquisition_status")
            passed = None if (held is None or recovery not in {"PASS", "FAIL"}
                              or summary.get("frozen_checks") != task["evaluation"]["shift_checks"]) else (
                                  held and recovery == "PASS")
        elif kind == "word_confirmed":
            hits, count = summary.get("confirmed_steps"), summary.get("confirmed_checks")
            observations = task["evaluation"]["observations"]
            schedule = {math.ceil(i * task["execution"]["steps"] / observations)
                        for i in range(1, observations + 1)}
            if hits is None or count is None or summary.get("observations") != observations:
                passed = None
            elif (not isinstance(hits, list) or type(count) is not int or count != len(hits)
                  or any(type(step) is not int or step not in schedule for step in hits)
                  or hits != sorted(set(hits)) or count > observations
                  or summary.get("first_confirmed_step") != (hits[0] if hits else None)):
                raise ValueError("invalid independently confirmed word-state evidence")
            else:
                passed = bool(hits)
        elif kind == "word_hold":
            passed = _count(summary, "confirmed_checks", "observations",
                            task["evaluation"]["observations"] + 1)
        elif kind == "native_holdout":
            passed = _native_quality(summary.get("holdout_metrics", {}), task,
                                     task["evaluation"]["holdout_samples"])
            oracle = _native_quality(summary.get("oracle_metrics", {}), task,
                                     task["evaluation"]["holdout_samples"])
            if oracle is None or summary.get("coverage_status") not in {"PASS", "FAIL"}:
                passed = None
            elif not oracle:
                raise ValueError("original target oracle is not valid")
        elif kind == "native_terminal":
            checks = summary.get("terminal_checks")
            count = task["evaluation"]["minimum_stable_checks"]
            stop, interval = task["execution"]["steps"], task["evaluation"]["eval_interval"]
            expected = [stop - interval * offset for offset in reversed(range(count))]
            if not isinstance(checks, list) or [row.get("step") for row in checks] != expected:
                passed = None
            else:
                grades = [_native_quality(row.get("metrics", {}), task,
                                          task["evaluation"]["eval_samples"]) for row in checks]
                passed = None if any(grade is None for grade in grades) else all(grades)
                if summary.get("coverage_status") not in {"PASS", "FAIL"}:
                    passed = None
                elif passed:
                    passed = summary["coverage_status"] == "PASS"
    except (ValueError, TypeError, KeyError) as exc:
        return _grade("INVALID", str(exc))
    if passed is None:
        return _grade("UNMEASURED", "Aggregate verdict lacks sufficient audited stage submetrics.")
    return _grade("PASS" if passed else "FAIL", "Derived from original audited stage submetrics.",
                  source_record_status=record["status"])


def build_report(root, *, performance=None):
    root = Path(root)
    registry, tasks = load_registry(root)
    if performance is None:
        from .test_performance import build_report as test_report
        performance = test_report(root)
    copy = dict(performance)
    if copy.pop("input_digest", None) != stable_hash(copy) or copy.get("qualification_input") is not False:
        raise ValueError("test stages require a verified nonqualifying performance projection")
    selected = {row["id"] for row in performance["recipes"] if row["selected"]}
    stages = []
    for definition in registry["stages"]:
        cohorts = []
        for name in definition["source_tasks"]:
            matched = [test for test in performance["tests"] if test["task_id"] == name
                       and test["lane"] == "Forge" and test["definition_match"] == "matches"]
            for test in matched:
                cells = []
                for recipe, records in sorted(test["cells"].items()):
                    for key in records:
                        record = performance["records"][key]
                        cells.append({"recipe": recipe, "record": key, "receipt": record.get("receipt"),
                                      "receipt_sha256": record.get("receipt_sha256"),
                                      **derive_stage(definition, tasks[name], test, record)})
                cohorts.append({"source_task": name, "source_test": test["id"],
                                "conditions": test["conditions"], "cells": cells})
            if not matched:
                cohorts.append({"source_task": name, "source_test": None, "conditions": None,
                                "cells": [{"recipe": recipe, "record": None,
                                           **_grade("UNMEASURED", "No compatible complete source-bound stage evidence.")}
                                          for recipe in sorted(selected)]})
        counts = Counter(cell["status"] for cohort in cohorts for cell in cohort["cells"])
        stages.append({**definition, "cohorts": cohorts, "counts": dict(sorted(counts.items()))})
    report = {"schema": "forge_quality_retention_stages_v1", "revision": registry["revision"],
              "qualification_input": False, "qualification_reuse": False, "training_launched": False,
              "additional_training_updates": 0, "stages": stages, "sources": registry["sources"],
              "source_performance_digest": performance["input_digest"],
              "inputs": {REGISTRY.as_posix(): file_hash(root / REGISTRY),
                         **{pin["path"]: pin["sha256"] for pin in registry["sources"].values()}},
              "projector_sha256": file_hash(Path(__file__))}
    report["input_digest"] = stable_hash(report)
    return report


def render_markdown(report):
    lines = ["# Quality and retention stages", "",
             "Tier 2 measures fixed-target quality. Tier 3 measures continued retention or target-change recovery. "
             "These independent measurement contracts share the original runs, budgets and numerical bounds.", "",
             "Original combined tasks and recorded qualification grades remain intact. Stage cells confer no new "
             "suite qualification. An aggregate PASS or FAIL cannot fill missing stage evidence; insufficient or "
             "incompatible source-bound submetrics are UNMEASURED. Counts below are receipt cells, not independent runs.", "",
             "| Stage | Tier | Original source tasks | Exact criterion | Stage cells |",
             "| --- | ---: | --- | --- | --- |"]
    for stage in report["stages"]:
        links = ", ".join(f"[{name}](../../configs/forge/tasks/{name}.json)" for name in stage["source_tasks"]) or "Protocol not frozen"
        counts = ", ".join(f"{count} {status}" for status, count in stage["counts"].items()) or "UNMEASURED"
        lines.append(f"| {stage['title']} | {stage['qualification_tier']} | {links} | {stage['criteria']} | {counts} |")
    lines += ["", "The JSON drilldown keeps every recipe, source, protocol/RNG/runtime cohort and receipt hash separate. "
              "A stage result never selects a different recipe or pools evidence across cohorts. Costs belong to the "
              "original run once; stages add no updates and must not be summed as new run costs.", "",
              "100G terminal retention covers only the original five-check window within 7,000 updates. "
              "It does not establish long continuation or recovery from a rotation. `rotated100` is a fixed rotated "
              "target; its quality does not prove rotation recovery. That separate stage remains unmeasured until "
              "the event, own-checkpoint pairing, deadline and continuation budget are frozen.", "",
              "The conditional `unipolar`, `cover_leftover` and `mid_scale_identity` tests preserve neutral/content/identity "
              "constraints across control inputs during a fresh fit. They have no later target event or acquired-state "
              "continuation and are outside this temporal split. `mode_hold` is also a fresh fit, despite its name.", "",
              "[Stage contracts](../../configs/forge/test-stages.json) · [Source-bound stage drilldown](test-stages.json) · "
              "[Original combined performance](test-performance.md)", "",
              f"Stage input digest `{report['input_digest']}`.", ""]
    return "\n".join(lines)


def generated_reports(root, *, performance=None):
    report = build_report(root, performance=performance)
    return {Path(root) / PREFIX.with_suffix(".json"): json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            Path(root) / PREFIX.with_suffix(".md"): render_markdown(report)}

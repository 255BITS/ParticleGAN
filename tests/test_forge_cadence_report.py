from copy import deepcopy
import json
import math

import pytest

from experiments.forge.cadence_report import build_report, render_markdown


def task(name, status="PASS", *, role="required", tier=1, score=.8):
    return {"task": name, "qualification_tier": tier, "importance": role,
            "gate_status": status, "metrics": {"score": score, "nested": {"error": .1},
                                               "finite": True, "samples": [1, 2]},
            "comparison_binding": "task-binding-" + name, "attempt_id": "attempt-" + name,
            "cost": {"wall_seconds": 2}}


@pytest.fixture
def summary():
    tasks = [task("smoke"), task("held", "FAIL", role="held", tier=2),
             task("clock", role="diagnostic"), task("endurance", tier=3)]
    trials = [{"candidate_id": f"candidate-{k}", "configuration_id": f"config-{k}",
               "settings": {"reg_every": k}, "tasks": deepcopy(tasks),
               "source_digest": "source", "protocol_hash": "protocol",
               "runtime_cohort": {"device": "cuda", "torch": "frozen"},
               "cost": {"wall_seconds": 10 * k}, "attempt_ids": [f"attempt-{k}"]}
              for k in (1, 4, 8, 16)]
    return {"study_id": "cadence-study", "source_digest": "source", "protocol_hash": "protocol",
            "runtime_cohort": {"device": "cuda", "torch": "frozen"},
            "tuning_through_tier": 3, "trials": trials, "logs": "/archive/progress.log",
            "cost": {"wall_seconds": 290}}


def test_complete_gate_preservation_metrics_and_roles_are_not_pooled(summary):
    summary["trials"][1]["tasks"][1].update(gate_status="PASS", metrics={"score": .9, "nested": {"error": .1}})
    summary["trials"][2]["tasks"][0]["metrics"]["score"] = .7
    original = deepcopy(summary)
    report = build_report(summary)
    assert summary == original
    assert report["complete"] and report["qualification_credit"] is False
    assert report["recommendation"]["status"] == "pass preserving candidates"
    assert report["recommendation"]["candidate_ids"] == ["candidate-4", "candidate-8", "candidate-16"]
    trial = report["trials"][1]
    assert trial["regressions"] == [] and trial["repairs"] == ["held"]
    assert trial["tiers"]["1"]["required"]["denominator"] == 1
    assert trial["tiers"]["1"]["diagnostic"]["pass"] == 1
    assert trial["tiers"]["2"]["held"]["pass"] == 1
    delta = next(p for p in report["trials"][2]["pairs"] if p["task"] == "smoke")["metric_deltas"]
    assert {m["metric"] for m in delta} == {"score", "nested.error"}
    assert next(m["delta"] for m in delta if m["metric"] == "score") == pytest.approx(-.1)
    assert report["cost"] == {"wall_seconds": 290}
    assert report["trials"][3]["cost"] == {"wall_seconds": 160}


def test_held_pass_loss_counts_as_regression_and_unknown_as_unavailable(summary):
    for trial in summary["trials"]:
        trial["tasks"][1]["gate_status"] = "PASS"
    for trial in summary["trials"][1:]:
        trial["tasks"][1]["gate_status"] = "FAIL"
    report = build_report(summary)
    assert report["recommendation"]["status"] == "no pass preserving variant"
    assert all(t["regressions"] == ["held"] for t in report["trials"][1:])
    summary["trials"][1]["tasks"][1]["gate_status"] = "INVALID"
    report = build_report(summary)
    assert report["recommendation"]["status"] == "incomplete"
    assert report["trials"][1]["regressions"] == []
    assert report["trials"][1]["unavailable"] == ["held"]


@pytest.mark.parametrize("field,value", [("source_digest", "other"), ("protocol_hash", "other"),
                                         ("runtime_cohort", {"device": "cpu"})])
def test_mixed_cohort_never_produces_deltas_or_preservation(summary, field, value):
    summary["trials"][1][field] = value
    report = build_report(summary)
    assert not report["complete"]
    trial = report["trials"][1]
    assert not trial["pass_preserving"]
    assert all(pair["status"] == "UNAVAILABLE" and not pair["metric_deltas"] for pair in trial["pairs"])
    assert any(field in reason for reason in trial["pairs"][0]["unavailable_reasons"])


def test_missing_cell_binding_and_mismatched_counts_remain_visible(summary):
    summary["trials"][1]["tasks"].pop()
    summary["trials"][2]["tasks"][0].pop("comparison_binding")
    summary["trials"][3]["counts"] = {"PASS": 500}
    report = build_report(summary)
    assert report["recommendation"]["status"] == "incomplete"
    assert report["trials"][1]["missing_cells"] == ["endurance"]
    assert report["trials"][1]["tiers"]["3"]["required"] == {
        "denominator": 1, "pass": 0, "fail": 0, "unavailable": 1, "statuses": {"UNAVAILABLE": 1}}
    pair = next(p for p in report["trials"][2]["pairs"] if p["task"] == "smoke")
    assert "missing task comparison_binding" in pair["unavailable_reasons"]
    assert not report["trials"][3]["counts_consistent"]
    assert report["trials"][3]["recomputed_counts"] == {"FAIL": 1, "PASS": 3}


def test_nonfinite_and_missing_metric_are_not_deltas_or_json_nan(summary):
    summary["trials"][1]["tasks"][0]["metrics"]["score"] = math.nan
    summary["trials"][1]["tasks"][0]["cadence"] = {"untrusted": math.inf}
    summary["cost"]["wall_seconds"] = math.inf
    report = build_report(summary)
    pair = next(p for p in report["trials"][1]["pairs"] if p["task"] == "smoke")
    assert pair["unavailable_metrics"] == ["score"]
    assert {m["metric"] for m in pair["metric_deltas"]} == {"nested.error"}
    assert report["recommendation"]["status"] == "incomplete"
    assert pair["cadence"] == {"untrusted": None}
    assert report["cost"]["wall_seconds"] is None
    json.dumps(report, allow_nan=False)


def test_diagnostic_coverage_restores_all_original_tiers_and_roles(summary):
    original = [{key: row[key] for key in ("task", "qualification_tier", "importance")}
                for row in summary["trials"][0]["tasks"]]
    summary["tuning_through_tier"] = 1
    summary["evidence_scope"] = "research_diagnostic"
    for trial in summary["trials"]:
        for row in trial["tasks"]:
            row.update(importance="diagnostic", qualification_tier=1)
    report = build_report(summary, source_assignments=original)
    assert report["diagnostic_scope"] and not report["qualification_credit"]
    assert report["complete"]
    assert report["trials"][1]["tiers"]["2"]["held"]["denominator"] == 1
    assert report["trials"][1]["tiers"]["3"]["required"]["pass"] == 1
    mapped = {a["task"]: a for a in original}
    assert build_report(summary, source_assignments=mapped) == report


def test_duplicate_control_task_and_assignment_conflicts_are_not_hidden(summary):
    summary["trials"][0]["tasks"].append(deepcopy(summary["trials"][0]["tasks"][0]))
    summary["trials"][1]["tasks"][1]["importance"] = "required"
    report = build_report(summary)
    assert not report["complete"]
    assert set(report["trials"][1]["unavailable"]) == {"smoke", "held"}


def test_future_tiers_and_empty_or_changed_recipe_evidence(summary):
    summary["tuning_through_tier"] = 4
    for trial in summary["trials"]:
        trial["tasks"][-1]["qualification_tier"] = 4
    assert build_report(summary)["trials"][0]["tiers"]["4"]["required"]["pass"] == 1
    summary["trials"][1]["tasks"][0]["metrics"] = {}
    summary["trials"][2]["settings"]["lr"] = .5
    summary["trials"][0]["resolved_recipe"] = {"reg_every": 1, "reg_coeff": 1.0}
    for trial in summary["trials"][1:]:
        trial["resolved_recipe"] = {"reg_every": trial["settings"]["reg_every"], "reg_coeff": 1.0}
    summary["trials"][3]["resolved_recipe"]["reg_coeff"] = 16.0
    report = build_report(summary)
    assert not report["complete"]
    reasons = " ".join(reason for t in report["trials"][1:] for p in t["pairs"] for reason in p["unavailable_reasons"])
    assert "no paired finite scalar metrics" in reasons
    assert "unmatched non-cadence settings" in reasons
    assert "unmatched non-cadence resolved_recipe" in reasons


def test_unmeasured_qualification_and_diagnostic_scope_never_grant_credit(summary):
    for trial in summary["trials"]:
        trial["qualification"] = {"full_view_qualified": True}
    report = build_report(summary)
    assert all(t["full_view_qualified"] for t in report["trials"])
    summary["trials"][1]["tasks"][0]["gate_status"] = "FAIL"
    assert not build_report(summary)["trials"][1]["full_view_qualified"]
    summary["evidence_scope"] = "research_diagnostic"
    report = build_report(summary)
    assert all(not t["full_view_qualified"] and not t["qualification_credit"] for t in report["trials"])
    assert not report["qualification_credit"]


def test_markdown_cadence_is_observed_not_inferred_from_wall_time(summary):
    summary["trials"][1]["tasks"][0]["cadence"] = {
        "optimizer": {"observed_steps_in_window": 32, "applied_record_calls_in_window": 8},
        "observations": {"applied": 8, "skipped": 24},
        "first_application_update": 4, "last_application_update": 32,
        "ranges": {"s_thermal": {"min": .01, "max": 1., "last": .1}}}
    text = render_markdown(build_report(summary))
    assert "| 4 | smoke | 32 | 8 | 8 / 24 | 4 / 32 | 0.01 / 1 / 0.1 |" in text
    assert "do not sum" in text
    assert "| 4 | 40 | UNAVAILABLE |" in text


@pytest.mark.parametrize("controls", [0, 2])
def test_exactly_one_control_required(summary, controls):
    summary["trials"] = summary["trials"][1:] if controls == 0 else [summary["trials"][0]] * 2
    with pytest.raises(ValueError, match="exactly one"):
        build_report(summary)


def test_markdown_escapes_cells_and_links_existing_board_only(summary):
    summary["study_id"] = "study|<script>\nnext"
    summary["trials"][1]["tasks"][0]["metrics"] = {"metric|injection\nnext": .4}
    text = render_markdown(build_report(summary))
    assert "study&#124;&lt;script&gt; next" in text
    assert "metric&#124;injection next" in text
    assert text.count("[The current technique inventory](../technique-inventory.md)") == 1
    assert "statistical noninferiority" in text
    assert "speed ranking" in text
    assert "/archive/progress.log" in text

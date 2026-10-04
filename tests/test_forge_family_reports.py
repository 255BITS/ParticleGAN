"""Reporting controls: counts, evidence identity, criteria and navigable pages."""
from copy import deepcopy
from pathlib import Path
import re

import pytest

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.family_reports import build_progress, generated_pages, render_leaderboard, score
from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint


@pytest.fixture
def report(tmp_path):
    contracts = {}
    bindings = {}
    for name in ("shared", "failed", "missing", "held", "diagnostic"):
        task = {"schema_version": 1, "id": name, "adapter": "fixture",
                "execution": {"steps": 24}, "evaluation": {"kind": "transfer_sustained",
                "thresholds": [["score", ">=", .5]], "observations": 24, "minimum_stable_checks": 5},
                "resources": {"timeout_seconds": 60}, "requires_capabilities": [], "dependencies": []}
        atomic_json(tmp_path / f"configs/forge/tasks/{name}.json", task)
        contract = {"execution_sha256": task_execution_fingerprint(task),
                    "evaluation_sha256": task_evaluation_fingerprint(task), "timeout_seconds": 60}
        bindings[name] = stable_hash(contract)
        contracts[bindings[name]] = contract

    def view(name, assignments, **extra):
        definition = {"id": name, "revision": 1, "assignments": [
            {"task": task, "qualification_tier": tier, "importance": role, "order": order}
            for order, (task, tier, role) in enumerate(assignments)],
            "calibration": {"status": "provisional"}, **extra}
        atomic_json(tmp_path / f"configs/forge/views/{name}.json", definition)
    view("alpha", [("shared", 1, "required"), ("failed", 2, "required"),
                   ("missing", 3, "required"), ("diagnostic", 2, "diagnostic")])
    view("beta", [("shared", 1, "required"), ("held", 2, "required")])
    view("diagnostics", [("diagnostic", 1, "diagnostic")], evidence_scope="research_diagnostic")
    selected = {"candidate_id": "k3p-config", "candidate_revision": "revision-one", "trainer_family": "k3p",
                "technique": "K3P", "qualified_tier": 0, "attempt_ids": ["attempt-one"],
                "bindings": {"source_digest": "source-one", "task_contracts": bindings},
                "runtime_cohort": {"execution_backend": "cuda"},
                "tasks": [{"task_id": name, "status": status} for name, status in
                          (("shared", "PASS"), ("failed", "FAIL"), ("missing", "UNKNOWN"), ("held", "PASS"))]}
    summary = {"candidate_id": selected["candidate_id"], "candidate_revision": selected["candidate_revision"],
               "provenance": {"source_digest": "source-one"}, "task_results": [
                   {"task_id": "failed", "reason": "score below .5", "metrics": {"score": .25},
                    "evaluator_summary": {"metric_checks": {"score": {"value": .25, "op": ">=",
                        "threshold": .5, "status": "FAIL"}}}}]}
    atomic_json(tmp_path / "reports/forge/technique-receipts/attempt-one.json", summary)
    publication = {"view": "alpha", "view_revision": 1, "rows": [selected], "task_contracts": contracts,
                   "provenance": {"input_digest": "fixture"}}
    return tmp_path, publication


def generate(report):
    root, publication = report
    before = deepcopy(publication["rows"])
    publication["family_progress"] = build_progress(root, publication)
    assert publication["rows"] == before
    return publication["family_progress"]["families"][0]["cohorts"][0]


def test_shared_tasks_count_per_view_and_diagnostics_never_fill_required_totals(report):
    cohort = generate(report)
    assert {tier: score(count) for tier, count in cohort["tiers"].items()} == {
        "1": "2/2", "2": "1/2", "3": "0(*)/1"}
    assert score(cohort["total"]) == "3(*)/5"
    assert [score(view["total"]) for view in cohort["views"]] == ["1(*)/3", "2/2"]
    assert len(cohort["tasks"]) == 5  # Four unique requirements plus a diagnostic.
    assert cohort["tasks"]["diagnostic"]["status"] == "UNKNOWN"


def test_completed_failure_has_no_marker_and_new_result_refreshes_every_total(report):
    root, publication = report
    cohort = generate(report)
    assert score(cohort["views"][0]["tiers"]["2"]) == "0/1"
    publication["rows"][0]["tasks"][2]["status"] = "PASS"
    updated = generate(report)
    assert score(updated["total"]) == "4/5"
    assert score(updated["views"][0]["total"]) == "2/3"


def test_unrun_family_defaults_to_zero_with_full_denominators(report):
    root, publication = report
    selected = publication["rows"][0]
    selected["tasks"] = []
    selected["attempt_ids"] = []
    cohort = generate(report)
    assert score(cohort["total"]) == "0(*)/5"
    assert [score(cohort["tiers"][tier]) for tier in ("1", "2", "3")] == ["0(*)/2", "0(*)/2", "0(*)/1"]


def test_changed_gate_preserves_recorded_failure_and_marks_coverage_incomplete(report):
    root, publication = report
    path = root / "configs/forge/tasks/failed.json"
    declaration = read_json(path)
    declaration["evaluation"]["thresholds"][0][2] = .75
    atomic_json(path, declaration)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["status"] == "FAIL"
    assert cohort["tasks"]["failed"]["current_contract"] == "CHANGED"
    assert score(cohort["views"][0]["tiers"]["2"]) == "0(*)/1"
    text = next(iter(generated_pages(root, publication).values()))
    assert "| score | 0.25 | >= 0.5 | FAIL |" in text
    assert "| score | >= 0.75 |" in text
    assert "All 24 declared observations" in text and "5 consecutive passing terminal" in text


@pytest.mark.parametrize("field", ["candidate_id", "candidate_revision", "source_digest"])
def test_receipt_cannot_supply_metrics_from_another_configuration_or_source(report, field):
    root, publication = report
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    target = receipt["provenance"] if field == "source_digest" else receipt
    target[field] = "another-cohort"
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="selected configuration/source"):
        build_progress(root, publication)


def test_runtime_groups_never_sum_or_borrow_each_others_passes(report):
    root, publication = report
    cpu = deepcopy(publication["rows"][0])
    cpu.update(runtime_cohort={"execution_backend": "cpu"}, attempt_ids=[], tasks=[])
    publication["rows"].append(cpu)
    generate(report)
    cohorts = publication["family_progress"]["families"][0]["cohorts"]
    assert [score(cohort["total"]) for cohort in cohorts] == ["3(*)/5", "0(*)/5"]
    assert len(generated_pages(root, publication)) == 1
    publication["rows"].append(deepcopy(cpu))
    with pytest.raises(ValueError, match="multiple selected configurations"):
        build_progress(root, publication)


def test_leaderboard_clicks_resolve_to_family_tiers_including_empty_tiers(report):
    root, publication = report
    generate(report)
    pages = generated_pages(root, publication)
    for path, content in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    overview = root / "reports/forge/technique-inventory.md"
    text = render_leaderboard(root, publication, overview)
    assert "**[3(*)/5]" in text and "↳ [alpha]" in text
    table = text.split("| --- | ---: | ---: | ---: | ---: |", 1)[1].split("\n\n", 1)[0]
    for target in re.findall(r"\]\(([^)]+)\)", table):
        filename, fragment = target.split("#")
        assert filename.startswith("families/")
        assert f'<a name="{fragment}"></a>' in (overview.parent / filename).read_text()
    assert "**[2/2]" in text  # A complete Tier 1 has no incomplete marker.
    assert publication["family_progress"] == build_progress(root, publication)
    assert pages == generated_pages(root, publication)


def test_committed_pages_and_every_drilldown_link_match_the_generator():
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    assert publication["family_progress"] == build_progress(root, publication)
    pages = generated_pages(root, publication)
    overview = root / "reports/forge/technique-inventory.md"
    pages[overview] = render_leaderboard(root, publication, overview)
    for page, text in pages.items():
        assert page.read_text() == text
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target:
                continue
            filename, _, fragment = target.partition("#")
            linked = (page.parent / filename).resolve()
            assert linked.is_file(), (page, target)
            if fragment:
                assert f'<a name="{fragment}"></a>' in pages.get(linked, linked.read_text()), (page, target)

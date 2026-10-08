"""Reporting controls: counts, evidence identity, criteria and navigable pages."""
from copy import deepcopy
from pathlib import Path
import re

import pytest

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.family_reports import (build_progress, certified_retry_successors,
                                              generated_pages, render_leaderboard, score)
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


def test_optimizer_alias_selects_whole_best_configuration_and_preserves_evidence(report):
    root, publication = report
    variant = deepcopy(publication["rows"][0])
    variant.update(candidate_id="k3p-optimizer-config", trainer_family="k3p-optimizer", attempt_ids=[])
    variant["bindings"]["source_digest"] = "source-two"
    variant["bindings"]["recipe_sha256"] = "recipe-two"
    variant["tasks"] = [{"task_id": name, "status": status} for name, status in
                        (("shared", "PASS"), ("failed", "PASS"), ("missing", "PASS"), ("held", "FAIL"))]
    publication["rows"].append(variant)
    publication["recipe_contracts"] = {"recipe-two": {
        "optimizer_family": "dualnorm", "optimizer_momentum": 0, "lr": .012,
        "d_lr_mult": 1.5, "prior_lr_mult": 2.5, "reg_arm": "b_cap"}}
    atomic_json(root / "configs/forge/trainer-families.json", {"families": [
        {"id": "k3p"}, {"id": "k3p-optimizer", "reporting_family": "k3p"}]})
    original = deepcopy(publication["rows"])
    cohort = generate(report)
    progress = publication["family_progress"]
    assert len(progress["families"]) == 1
    assert cohort["row_index"] == 1 and score(cohort["total"]) == "4/5"
    assert cohort["tasks"]["held"]["status"] == "FAIL"  # Cannot borrow the baseline PASS.
    assert publication["rows"] == original
    assert len(progress["configuration_families"]) == 2
    page = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert "Best recorded configuration" in page
    assert "optimizer_family=dualnorm" in page and "optimizer_momentum=0" in page
    assert "critic rate: **0.018**" in page and "prior rate: **0.03**" in page
    assert "source-two" in page and "source-one" in page


def test_optimizer_alias_keeps_exact_runtime_groups_and_stable_ties(report):
    root, publication = report
    variant = deepcopy(publication["rows"][0])
    variant.update(candidate_id="variant", trainer_family="optimizer", attempt_ids=[])
    publication["rows"].append(variant)
    atomic_json(root / "configs/forge/trainer-families.json", {"families": [
        {"id": "k3p"}, {"id": "optimizer", "reporting_family": "k3p"}]})
    assert generate(report)["row_index"] == 0  # Existing solution wins a complete tie.
    variant["runtime_cohort"] = {"execution_backend": "cpu"}
    generate(report)
    cohorts = publication["family_progress"]["families"][0]["cohorts"]
    assert {item["backend"]: item["row_index"] for item in cohorts} == {"cuda": 0, "cpu": 1}


def test_explicit_current_optimizer_wins_display_without_borrowing_or_hiding_history(report):
    from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin
    root, publication = report
    variant = deepcopy(publication["rows"][0])
    variant.update(candidate_id="selected-optimizer", trainer_family="optimizer", attempt_ids=[])
    variant["bindings"]["source_digest"] = "selected-source"
    variant["tasks"] = [{"task_id": name, "status": "FAIL"} for name in ("shared", "failed", "missing", "held")]
    publication["rows"].append(variant)
    old = deepcopy(variant)
    old["runtime_cohort"]["model"] = "older-runtime"
    old["bindings"]["source_digest"] = "archived-source"
    publication["historical_family_rows"] = [old]
    atomic_json(root / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [
        {"id": "k3p", "label": "K3P", "candidates": ["k3p-config"],
         "canonical_candidate": "k3p-config", "current_configuration_family": "optimizer"},
        {"id": "optimizer", "label": "Optimizer", "candidates": ["selected-optimizer"],
         "canonical_candidate": "selected-optimizer", "reporting_family": "k3p"}]})
    pin = family_row_pin(variant, selection_kind="historical_incumbent", reason="Explicit current optimizer.")
    atomic_json(root / CURRENT_SELECTION, {"schema_version": 1, "scope": "whole_candidate_family_current",
        "default_adoption": False, "view": "alpha", "policy_fingerprint": "fixture", "selections": [pin]})
    original = deepcopy(publication["rows"] + publication["historical_family_rows"])
    current = generate(report)
    family = publication["family_progress"]["families"][0]
    assert current["row_index"] == 1 and score(current["total"]) == "0/5"
    assert len(family["cohorts"]) == 2
    assert publication["rows"] + publication["historical_family_rows"] == original
    page = root / "reports/forge/technique-inventory.md"
    text = render_leaderboard(root, publication, page)
    assert sum(line.startswith("| **[") for line in text.splitlines()) == 1
    assert "older-runtime" not in text
    detail = generated_pages(root, publication)[root / "reports/forge/families/k3p.md"]
    assert "Current benchmark configuration" in detail and "Archived runtime cohort" in detail
    assert "selected-source" in detail and "archived-source" in detail
    pin["scientific_row_sha256"] = "different-row"
    atomic_json(root / CURRENT_SELECTION, {"schema_version": 1, "scope": "whole_candidate_family_current",
        "default_adoption": False, "view": "alpha", "policy_fingerprint": "fixture", "selections": [pin]})
    with pytest.raises(ValueError, match="exactly one verified scientific row"):
        build_progress(root, publication)


def test_hidden_inventory_entry_preserves_numerical_rows_without_navigation(report):
    root, publication = report
    hidden = deepcopy(publication["rows"][0])
    hidden.update(candidate_id="legacy", trainer_family="legacy", technique="Legacy", attempt_ids=[])
    publication["rows"].append(hidden)
    atomic_json(root / "configs/forge/trainer-families.json", {"families": [
        {"id": "k3p"}, {"id": "legacy", "inventory_visible": False}]})
    original = deepcopy(publication["rows"])
    generate(report)
    assert publication["rows"] == original
    assert len(publication["family_progress"]["families"]) == 1
    assert "Legacy" not in render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert root / "reports/forge/families/legacy.md" not in generated_pages(root, publication)


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


def test_changed_gate_preserves_recorded_failure_without_unrun_marker(report):
    root, publication = report
    path = root / "configs/forge/tasks/failed.json"
    declaration = read_json(path)
    declaration["evaluation"]["thresholds"][0][2] = .75
    atomic_json(path, declaration)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["status"] == "FAIL"
    assert cohort["tasks"]["failed"]["current_contract"] == "CHANGED"
    assert cohort["tasks"]["failed"]["changed_contract_fields"] == ["evaluation (gates or sampling law)"]
    assert "Test definition changed since this run: evaluation" in cohort["tasks"]["failed"]["coverage_reason"]
    assert cohort["tasks"]["failed"]["execution_recorded"] is True
    assert score(cohort["views"][0]["tiers"]["2"]) == "0/1"
    text = next(iter(generated_pages(root, publication).values()))
    assert "| score | 0.25 | >= 0.5 | FAIL |" in text
    assert "| score | >= 0.75 |" in text
    assert "All 24 declared observations" in text and "5 consecutive passing terminal" in text


@pytest.mark.parametrize("status", ["PASS", "FAIL"])
def test_missing_recorded_definition_does_not_hide_an_executed_result(report, status):
    root, publication = report
    selected = publication["rows"][0]
    selected["tasks"][1]["status"] = status
    selected["bindings"]["task_contracts"].pop("failed")
    cohort = generate(report)
    result = cohort["tasks"]["failed"]
    assert result["current_contract"] == "unbound"
    assert result["execution_recorded"] is True
    assert score(cohort["views"][0]["tiers"]["2"]) == ("1/1" if status == "PASS" else "0/1")
    assert "Recorded test definition unavailable" in result["coverage_reason"]


@pytest.mark.parametrize("status,raw_status", [("ERROR", "error"), ("INVALID", "completed"),
                                               ("INCOMPLETE", "timeout"), ("INCOMPLETE", "cancelled")])
def test_attempted_errors_show_cause_attempt_and_cost_without_pass_credit(report, status, raw_status):
    root, publication = report
    publication["rows"][0]["tasks"][1]["status"] = status
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    receipt["task_results"][0].update(gate_status=status, raw_status=raw_status,
        reason="specific failure: missing evaluator certificate" if status == "INVALID" else "specific worker failure",
        cost={"execution_seconds": 12.5, "wall_seconds": 0, "charged_task": "shared"})
    atomic_json(path, receipt)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["execution_recorded"] is True
    assert cohort["tasks"]["failed"]["attempt_id"] == "attempt-one"
    assert score(cohort["views"][0]["tiers"]["2"]) == "0/1"
    assert cohort["views"][0]["tiers"]["2"]["counts"] == {status: 1}
    text = next(iter(generated_pages(root, publication).values()))
    expected_reason = "specific failure: missing evaluator certificate" if status == "INVALID" else "specific worker failure"
    assert expected_reason in text
    assert "Attempt: `attempt-one`; raw outcome: " + raw_status in text
    assert "Recorded execution seconds: 12.5; charged wall seconds: 0" in text
    assert "charged once to shared" in text
    assert "Compact metrics and receipt provenance" in text


@pytest.mark.parametrize("status", ["UNKNOWN", "NOT_RUN", "BLOCKED", "INVALID", "INCOMPLETE", "ERROR"])
def test_no_task_execution_evidence_retains_marker_even_when_cohort_has_an_attempt(report, status):
    root, publication = report
    publication["rows"][0]["tasks"][2]["status"] = status
    cohort = generate(report)
    assert cohort["tasks"]["missing"]["execution_recorded"] is False
    assert cohort["tasks"]["missing"]["attempt_id"] is None
    assert score(cohort["views"][0]["tiers"]["3"]) == "0(*)/1"


def test_worker_launch_failure_with_zero_execution_retains_unrun_marker(report):
    root, publication = report
    publication["rows"][0]["tasks"][1]["status"] = "INCOMPLETE"
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    receipt["task_results"][0].update(raw_status="error", reason="worker launch failed",
                                     cost={"execution_seconds": 0, "wall_seconds": 0})
    atomic_json(path, receipt)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["execution_recorded"] is False
    assert score(cohort["views"][0]["tiers"]["2"]) == "0(*)/1"
    assert "worker launch failed" in next(iter(generated_pages(root, publication).values()))


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


def _compact_repair(report):
    root, publication = report
    parent = read_json(root / "reports/forge/technique-receipts/attempt-one.json")
    parent.update(attempt_id="attempt-one", attempt_status="error", certificate_validated=True,
                  qualification_input=False, qualification_reuse=False, runtime={"python": "3.12"})
    parent["provenance"].update(canonical_result_hash=stable_hash({"interrupted": True}), source_origin_commit="commit-one")
    parent["task_results"][0].update(compatibility_key="one-complete-task-law", gate_status="INCOMPLETE",
                                     raw_status="error", reason="worker stopped", cost={"wall_seconds": 12.5})
    child = deepcopy(parent)
    child.update(attempt_id="attempt-retry", attempt_status="completed",
                 retry_of={"attempt_id": "attempt-one", "result_hash": parent["provenance"]["canonical_result_hash"],
                           "reason": "Explicit environment repair", "authorized_at": "2026-10-05T19:35:12+00:00"})
    child["provenance"]["canonical_result_hash"] = stable_hash({"repaired": True})
    result = child["task_results"][0]
    result.update(gate_status="PASS", raw_status="completed", reason="complete repaired run",
                  metrics={"score": .9}, cost={"wall_seconds": 7.25})
    result["evaluator_summary"]["metric_checks"]["score"].update(value=.9, status="PASS")
    return {"attempt-one": parent, "attempt-retry": child}


@pytest.mark.parametrize("reverse", [False, True])
def test_certified_repair_uses_successor_metrics_and_preserves_original_history_without_raw_receipts(report, reverse):
    root, publication = report
    summaries = _compact_repair(report)
    for attempt, summary in summaries.items():
        atomic_json(root / f"reports/forge/technique-receipts/{attempt}.json", summary)
    selected = publication["rows"][0]
    selected["attempt_ids"] = list(reversed(summaries)) if reverse else list(summaries)
    selected["tasks"][1]["status"] = "PASS"
    selected["cost"] = {"wall_seconds": 19.75}
    original = deepcopy(publication["rows"])
    protected = {path: path.read_bytes() for path in (root / "reports/forge/technique-receipts").glob("*.json")}
    assert not (root / "reports/forge/attempts").exists()
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["status"] == "PASS"
    assert cohort["tasks"]["failed"]["attempt_id"] == "attempt-retry"
    assert cohort["tasks"]["failed"]["metrics"]["metrics"] == {"score": .9}
    assert cohort["retry_history"][0]["attempt_id"] == "attempt-one"
    assert cohort["retry_history"][0]["superseded_by"] == "attempt-retry"
    assert cohort["retry_history"][0]["original_task_costs"] == {"failed": {"wall_seconds": 12.5}}
    assert score(cohort["total"]) == "4(*)/5"
    text = next(iter(generated_pages(root, publication).values()))
    assert "Certified execution repair" in text and "attempt-one.json" in text and "attempt-retry.json" in text
    assert "original outcome and cost remain recorded" in text
    assert publication["rows"] == original
    assert all(path.read_bytes() == content for path, content in protected.items())


@pytest.mark.parametrize("tamper", ["hash", "parent_missing", "authorization", "reason", "certificate", "qualification",
                                    "parent_completed", "parent_fail", "runtime", "source", "origin", "candidate", "revision",
                                    "task_key", "duplicate_task", "extra_task", "identity"])
def test_compact_repair_rejects_unproven_or_changed_lineage(report, tamper):
    summaries = _compact_repair(report)
    parent, child = summaries["attempt-one"], summaries["attempt-retry"]
    if tamper == "hash":
        child["retry_of"]["result_hash"] = "wrong-hash"
    elif tamper == "parent_missing":
        del summaries["attempt-one"]
    elif tamper in {"authorization", "reason"}:
        child["retry_of"]["authorized_at" if tamper == "authorization" else "reason"] = ""
    elif tamper == "certificate":
        parent["certificate_validated"] = False
    elif tamper == "qualification":
        child["qualification_input"] = True
    elif tamper == "parent_completed":
        parent["attempt_status"] = "completed"
    elif tamper == "parent_fail":
        parent["task_results"][0]["gate_status"] = "FAIL"
    elif tamper == "runtime":
        child["runtime"]["python"] = "other"
    elif tamper in {"source", "origin"}:
        child["provenance"]["source_digest" if tamper == "source" else "source_origin_commit"] = "other"
    elif tamper in {"candidate", "revision"}:
        child["candidate_id" if tamper == "candidate" else "candidate_revision"] = "other"
    elif tamper == "task_key":
        child["task_results"][0]["compatibility_key"] = "different-task-law"
    elif tamper in {"duplicate_task", "extra_task"}:
        extra = deepcopy(child["task_results"][0])
        if tamper == "extra_task":
            extra["task_id"] = "other-task"
        child["task_results"].append(extra)
    elif tamper == "identity":
        child["attempt_id"] = "other-attempt"
    with pytest.raises(ValueError, match="compact retry"):
        certified_retry_successors(summaries)


def test_compact_repair_rejects_conflicting_branches_and_cycles(report):
    summaries = _compact_repair(report)
    sibling = deepcopy(summaries["attempt-retry"])
    sibling["attempt_id"] = "attempt-sibling"
    summaries["attempt-sibling"] = sibling
    with pytest.raises(ValueError, match="conflicting compact retry branches"):
        certified_retry_successors(summaries)
    del summaries["attempt-sibling"]
    parent, child = summaries["attempt-one"], summaries["attempt-retry"]
    child["attempt_status"] = "error"
    child["task_results"][0]["gate_status"] = "INCOMPLETE"
    parent["retry_of"] = {**child["retry_of"], "attempt_id": "attempt-retry",
                          "result_hash": child["provenance"]["canonical_result_hash"]}
    with pytest.raises(ValueError, match="cyclic compact retry"):
        certified_retry_successors(summaries)


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


def test_cuda_labels_are_omitted_only_when_one_runtime_is_unambiguous(report):
    root, publication = report
    generate(report)
    page = root / "reports/forge/technique-inventory.md"
    text = render_leaderboard(root, publication, page)
    assert "**[K3P](" in text and "K3P (cuda)" not in text
    cpu = deepcopy(publication["rows"][0])
    cpu.update(runtime_cohort={"execution_backend": "cpu"}, attempt_ids=[], tasks=[])
    publication["rows"].append(cpu)
    generate(report)
    text = render_leaderboard(root, publication, page)
    assert "K3P (cuda)" in text and "K3P (cpu)" in text


def test_task_devices_and_policy_parent_remain_visible_in_provenance(report):
    root, publication = report
    path = root / "reports/forge/technique-receipts/attempt-one.json"
    receipt = read_json(path)
    receipt["task_results"][0]["device"] = "cpu"
    atomic_json(path, receipt)
    task_path = root / "configs/forge/tasks/failed.json"
    task = read_json(task_path)
    task["policy_parent"] = {"id": "shared", "task_sha256": "original-parent-hash"}
    atomic_json(task_path, task)
    cohort = generate(report)
    assert cohort["tasks"]["failed"]["device"] == "cpu"
    text = next(iter(generated_pages(root, publication).values()))
    assert "Actual task device: `cpu`" in text
    assert "original-parent-hash" in text and "no cells to the parent clean cohort" in text


def test_scoped_task_variant_discovery_and_links_use_its_actual_declaration_path(report):
    root, publication = report
    original = root / "configs/forge/tasks/held.json"
    variant = root / "configs/forge/task-variants/policy-cohort/held.json"
    variant.parent.mkdir(parents=True)
    original.rename(variant)
    cohort = generate(report)
    assert cohort["tasks"]["held"]["current_contract"] == "matches"
    assert publication["family_progress"]["task_paths"]["held"] == variant.relative_to(root).as_posix()
    text = next(iter(generated_pages(root, publication).values()))
    assert "task-variants/policy-cohort/held.json" in text


def test_separate_policy_view_retains_required_results_without_expanding_parent_family_totals(report):
    root, publication = report
    original = generate(report)
    definition = read_json(root / "configs/forge/views/beta.json")
    definition.update(id="policy-coverage", reporting={"family_totals": False})
    atomic_json(root / "configs/forge/views/policy-coverage.json", definition)
    updated = generate(report)
    assert updated["total"] == original["total"] and updated["tiers"] == original["tiers"]
    assert publication["family_progress"]["scoped_views"] == ["policy-coverage"]
    assert updated["scoped_views"][0]["tiers"]["1"] == {"passed": 1, "total": 1,
        "incomplete": False, "counts": {"PASS": 1}}
    text = next(iter(generated_pages(root, publication).values()))
    assert "Separate cohort coverage (excluded from family totals)" in text
    assert "This ordinary lane retains its own required gates" in text
    overview = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "↳ [policy-coverage]" not in overview


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
        filename, _, fragment = target.partition("#")
        assert filename.startswith("families/")
        if fragment:
            assert f'<a name="{fragment}"></a>' in (overview.parent / filename).read_text()
        else:
            assert "## Technique overview" in (overview.parent / filename).read_text()
    assert "**[2/2]" in text  # A complete Tier 1 has no incomplete marker.
    assert publication["family_progress"] == build_progress(root, publication)
    assert pages == generated_pages(root, publication)


def test_committed_pages_and_every_drilldown_link_match_the_generator():
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    assert publication["family_progress"] == build_progress(root, publication)
    pages = generated_pages(root, publication)
    overview = root / "reports/forge/technique-inventory.md"
    from reports.forge.regenerate_technique_inventory import _current_markdown
    pages[overview] = _current_markdown(publication, root, overview)
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


def test_current_clock_measurement_retains_original_evidence_alongside_contract_drift():
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    recorded = deepcopy(publication["rows"])
    progress = build_progress(root, publication)
    assert publication["rows"] == recorded
    assert progress["qualification_input"] is False
    round_definition = read_json(root / "configs/forge/rounds/tier1-completion-v1.json")
    original_families = {row["family"] for row in round_definition["candidate_roster"]}
    cohorts = {family["id"]: family["cohorts"][0] for family in
               progress["families"] + progress["configuration_families"]
               if family["id"] in original_families}
    assert set(cohorts) == original_families
    for name, cohort in cohorts.items():
        clock = cohort["tasks"]["clockfree_audit_measurement_v1"]
        if name in {"atlas", "e22"}:
            assert cohort["tiers"]["1"]["incomplete"] is True
            assert clock["status"] == "BLOCKED"
            continue
        # Replacing the sustained question with independently confirmed smoke
        # creates an unknown cell; the recorded scientific rows stay unchanged.
        assert cohort["tiers"]["1"]["incomplete"] is True
        assert cohort["tasks"]["five_word_joint_smoke"]["status"] == "UNKNOWN"
        assert sum(cohort["tiers"]["1"]["counts"].values()) == 22
        assert clock["status"] == "FAIL" and clock["current_contract"] == "matches"
        assert cohort["tasks"]["clockfree_audit"]["status"] == "UNKNOWN"
        assert cohort["tasks"]["clockfree_audit"]["current_contract"] == "unbound"
        assert recorded[cohort["row_index"]]["qualified_tier"] == 0
    bcap = cohorts["bcap"]
    assert score(bcap["tiers"]["1"]) == "19(*)/22"
    clock_view = next(view for view in bcap["views"] if view["id"] == "clockfree_continuous")
    assert score(clock_view["tiers"]["1"]) == "3/4"
    # Detailed historical clock diagnostics retain their original cohort.
    clock = next(cohort["tasks"]["clockfree_audit_measurement_v1"]
                 for family in progress["families"] + progress["configuration_families"] if family["id"] == "bcap"
                 for cohort in family["cohorts"] if "clock_audit" in cohort["tasks"]["clockfree_audit_measurement_v1"])
    assert clock["clock_audit"]["comparisons"]["step_label"]["digest_equal"] is False
    assert clock["clock_audit"]["comparisons"]["horizon"]["digest_equal"] is False
    assert clock["training_media"][0]["recorded_grade"] == "FAIL"


def test_all_current_non_policy_families_have_executed_tier1_without_new_qualification():
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    rows = deepcopy(publication["rows"])
    progress = build_progress(root, publication)
    selection = read_json(root / "configs/forge/selections/family-current-v1.json")
    original_families = {pin["trainer_family"] for pin in selection["selections"]
                         if pin["selection_kind"] != "current_measurement"} - {"atlas", "e22"}
    registry = read_json(root / "configs/forge/trainer-families.json")["families"]
    original_families -= {family["id"] for family in registry if family.get("inventory_visible") is False}
    ordinary = [family for family in progress["families"] + progress["configuration_families"]
                if family["id"] in original_families]
    assert {family["id"] for family in ordinary} == original_families
    for family in ordinary:
        for cohort in family["cohorts"]:
            assert cohort["tiers"]["1"]["incomplete"] is False
            assert "(*)" not in score(cohort["tiers"]["1"])
            assert set(cohort["tiers"]["1"]["counts"]) <= {"PASS", "FAIL"}
            assert (rows + publication.get("historical_family_rows", []))[cohort["row_index"]]["qualified_tier"] == 0
    publication["family_progress"] = progress
    text = render_leaderboard(root, publication, root / "reports/forge/technique-inventory.md")
    assert "Atlas/E22 retain (*) for blocked, unrun tests" in text
    assert publication["rows"] == rows


def test_current_measurement_families_complete_only_their_declared_view_scope():
    from experiments.forge.trainer_families import scientific_row_hash
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    rows = deepcopy(publication["rows"])
    selection = read_json(root / "configs/forge/selections/family-current-v1.json")
    pins = {pin["trainer_family"]: pin for pin in selection["selections"]}
    progress = build_progress(root, publication)
    families = {family["id"]: family for family in progress["families"] + progress["configuration_families"]}
    unmeasured = {row["trainer_family"] for row in rows
                  if row["selection"]["selection_kind"] == "unmeasured_declaration"}
    registry = read_json(root / "configs/forge/trainer-families.json")["families"]
    hidden = {family["id"] for family in registry if family.get("inventory_visible") is False}
    assert set(families) == (set(pins) | unmeasured) - hidden
    assert unmeasured == {family["id"] for family in registry
                          if family.get("unmeasured_display_backend")} - hidden
    for row in rows:
        if row["trainer_family"] in unmeasured:
            assert not row["attempt_ids"] and row["qualified_tier"] == 0
            assert row["selection"]["qualified"] is False and row["selection"]["default_adoption"] is False
            assert all(task["status"] in {"UNKNOWN", "BLOCKED", "NOT_RUN"}
                       for task in row["tasks"] + row.get("nonrequired_tasks", []))
    for name, pin in pins.items():
        if pin["selection_kind"] != "current_measurement":
            continue
        required = set(pin.get("measurement_tasks", []))
        for view_name in pin["measurement_views"]:
            # A pinned measurement covered its original view, not a new task
            # introduced by a later publication-only policy refresh.
            view = read_json(root / f"configs/forge/view-history/{view_name}-v7.json")
            required.update(assignment["task"] for assignment in view["assignments"]
                            if assignment["importance"] == "required" and assignment["qualification_tier"] == 1)
        assert required
        current = [cohort for cohort in families[name]["cohorts"]
                   if cohort["row_index"] < len(rows)
                   and scientific_row_hash(rows[cohort["row_index"]]) == pin["scientific_row_sha256"]]
        assert len(current) == 1
        for cohort in current:
            original = rows[cohort["row_index"]]
            recorded_tasks = {task["task_id"]: task for task in original["tasks"] + original.get("nonrequired_tasks", [])}
            assert all(recorded_tasks[task]["status"] in {"PASS", "FAIL"} for task in required)
            views = {view["id"]: view for view in cohort["views"] + cohort.get("scoped_views", [])}
            for view_name in pin["measurement_views"]:
                tier = views[view_name]["tiers"]["1"]
                assert tier["incomplete"] is True
                assert set(tier["counts"]) <= {"PASS", "FAIL", "UNKNOWN"}
                assert cohort["tasks"]["five_word_joint_smoke"]["status"] == "UNKNOWN"
            # Other goal views retain any unknown cells; a bounded measurement
            # grants neither execution nor qualification outside its scope.
            for view in cohort["views"]:
                declaration = read_json(root / f"configs/forge/views/{view['id']}.json")
                statuses = [cohort["tasks"][assignment["task"]]["status"] for assignment in declaration["assignments"]
                            if assignment["importance"] == "required" and assignment["qualification_tier"] == 1]
                if any(status in {"UNKNOWN", "NOT_RUN", "BLOCKED"} for status in statuses):
                    assert view["tiers"]["1"]["incomplete"] is True
            assert (rows + publication.get("historical_family_rows", []))[cohort["row_index"]]["qualified_tier"] == 0
    assert publication["rows"] == rows


def test_legacy_gan_pages_preserve_original_whole_rows_and_do_not_enter_current_totals():
    from experiments.forge.trainer_families import scientific_row_hash
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    selection = read_json(root / "configs/forge/selections/family-current-v1.json")
    current = [row for row in publication["rows"] if row["trainer_family"] == "release07-gan-v3"]
    assert len(current) == 1
    old_pins = {}
    for pin in selection["historical_selections"]:
        old_pins.setdefault(pin["trainer_family"], set()).add(pin["scientific_row_sha256"])
    for row in publication["historical_family_rows"]:
        assert scientific_row_hash(row) in old_pins[row["trainer_family"]]
    aliases = publication["family_progress"]["historical_families"]
    historical_ids = {family["id"] for family in read_json(root / "configs/forge/trainer-families.json")["historical_families"]}
    assert {family["id"] for family in aliases} == set(old_pins) & historical_ids
    assert not historical_ids & {family["id"] for family in publication["family_progress"]["families"]}
    pages = generated_pages(root, publication)
    for family in aliases:
        text = pages[root / family["page"]]
        assert "Historical cohort navigation" in text
        assert '<a name="cohort-cuda-7f9c23eb0e27-tier-1"></a>' in text
        assert "not pooled into it" in text


def test_first_case_snapshot_is_archival_and_missing_later_reports_add_no_cells(report, monkeypatch):
    root, publication = report
    from experiments.forge import family_reports
    historical = {"first_case": {"task_id": "two_pole", "status": "FAIL"},
                  "campaign_status": "PENDING_ADAPTER_AND_BUDGET", "remaining_not_run": 25}
    monkeypatch.setattr(family_reports, "_full_original_atlas_first_case", lambda root, load: deepcopy(historical))
    before = deepcopy(publication["rows"])
    progress = build_progress(root, publication)
    assert publication["rows"] == before
    assert "atlas_evidence_navigation" not in progress
    archived = progress["full_original_atlas_common26"]
    assert archived["archived_snapshot"] is True
    assert archived["counts_scope"] == "AT_FIRST_ACCEPTED_CASE"
    assert archived["remaining_not_run"] == historical["remaining_not_run"]
    assert score(progress["families"][0]["cohorts"][0]["total"]) == "3(*)/5"

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
    assert cohort["tasks"]["failed"]["changed_contract_fields"] == ["evaluation (gates or sampling law)"]
    assert "Current coverage is stale: changed evaluation" in cohort["tasks"]["failed"]["coverage_reason"]
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
        filename, fragment = target.split("#")
        assert filename.startswith("families/")
        assert f'<a name="{fragment}"></a>' in (overview.parent / filename).read_text()
    assert "**[2/2]" in text  # A complete Tier 1 has no incomplete marker.
    assert publication["family_progress"] == build_progress(root, publication)
    assert pages == generated_pages(root, publication)


def test_committed_pages_and_every_drilldown_link_match_the_generator():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    assert publication["family_progress"] == build_progress(root, publication)
    pages = generated_pages(root, publication)
    overview = root / "reports/forge/technique-inventory.md"
    spec = importlib.util.spec_from_file_location(
        "committed_inventory_publisher", root / "reports/forge/regenerate_technique_inventory.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    # The native publisher also verifies and appends registered research
    # navigation; check its full output and every linked artifact.
    pages[overview] = publisher._current_markdown(publication, root, overview)
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


def test_current_schedule_audit_does_not_relabel_or_reuse_historical_clock_measurement():
    from experiments.forge.scoped_publications import load_publications
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    recorded = deepcopy(publication["rows"])
    historical = load_publications(root)
    # Remove direct audit results from the display fixture, even if the current
    # selected configuration has since measured them. Archived clock results
    # must never supply either missing operational or strict-clock-free cells.
    unmeasured = deepcopy(publication)
    for row in unmeasured["rows"]:
        for key in ("tasks", "nonrequired_tasks"):
            row[key] = [item for item in row.get(key, [])
                        if item["task_id"] not in {"schedule_contract_audit", "clockfree_audit"}]
        for name in ("schedule_contract_audit", "clockfree_audit"):
            row.get("bindings", {}).get("task_contracts", {}).pop(name, None)
    progress = build_progress(root, unmeasured)
    assert publication["rows"] == recorded
    assert progress["qualification_input"] is False
    cohorts = {family["id"]: family["cohorts"][0] for family in progress["families"]}
    for cohort in cohorts.values():
        audit = cohort["tasks"]["schedule_contract_audit"]
        assert audit["status"] == "UNKNOWN" and audit["current_contract"] == "unbound"
        assert cohort["tiers"]["1"]["incomplete"] is True
        assert sum(cohort["tiers"]["1"]["counts"].values()) == 23
        assert "clockfree_audit_measurement_v1" not in cohort["tasks"]
        assert cohort["tasks"]["clockfree_audit"]["status"] == "UNKNOWN"
        assert cohort["tasks"]["clockfree_audit"]["current_contract"] == "unbound"
        stability = next(view for view in cohort["views"] if view["id"] == "discriminator_stability")
        assert stability["tiers"]["1"]["total"] == 7
    definitions = {view["id"]: view for view in progress["views"]}
    stability = {item["task"]: item for item in definitions["discriminator_stability"]["assignments"]}
    strict = {item["task"]: item for item in definitions["clockfree_continuous"]["assignments"]}
    assert stability["schedule_contract_audit"]["importance"] == "required"
    assert strict["clockfree_audit"]["importance"] == "required"
    assert strict["clockfree_audit"]["qualification_tier"] == 3
    assert strict["schedule_contract_audit"]["qualification_tier"] == 1
    assert definitions["clockfree_continuous"]["eligibility"]["claim_contract"]["learning"] == "clockfree"
    assert read_json(root / "configs/forge/tasks/schedule_contract_audit.json")["evaluation"]["clockfree_claim"] is False
    assert "clockfree_audit_measurement_v1" not in stability
    clock = historical["clock_audits"][("bcap", "clockfree_audit_measurement_v1")]
    old_result = historical["final_measurements"][("bcap", "clockfree_audit_measurement_v1")]
    assert old_result["status"] == clock["recorded_grade"] == "FAIL"
    assert old_result["attempt_id"] == clock["attempt_id"]
    assert old_result["canonical_result_hash"] == clock["canonical_result_hash"]
    assert old_result["attempt_id"] == "5732e526ff764977af157e194f8a5749"
    assert old_result["canonical_result_hash"] == "a3fec24dab8f8761b4273f6fbb5ac641026a9371b40b87761e2617513862fbba"
    assert clock["comparisons"]["step_label"]["digest_equal"] is False
    assert clock["comparisons"]["horizon"]["digest_equal"] is False
    assert historical["media"][("bcap", "clockfree_audit_measurement_v1", clock["attempt_id"])]["recorded_grade"] == "FAIL"
    assert load_publications(root) == historical


def test_legacy_gan_pages_preserve_original_whole_rows_and_do_not_enter_current_totals():
    from experiments.forge.trainer_families import scientific_row_hash
    root = Path(__file__).resolve().parents[1]
    publication = read_json(root / "reports/forge/technique-inventory.json")
    selection = read_json(root / "configs/forge/selections/family-current-v1.json")
    current = [row for row in publication["rows"] if row["trainer_family"] == "release07-gan-v3"]
    assert len(current) == 1
    old_pins = {pin["trainer_family"]: pin for pin in selection["historical_selections"]}
    for row in publication["historical_family_rows"]:
        assert scientific_row_hash(row) == old_pins[row["trainer_family"]]["scientific_row_sha256"]
    aliases = publication["family_progress"]["historical_families"]
    assert {family["id"] for family in aliases} == set(old_pins)
    assert not set(old_pins) & {family["id"] for family in publication["family_progress"]["families"]}
    pages = generated_pages(root, publication)
    for family in aliases:
        text = pages[root / family["page"]]
        assert "Historical cohort navigation" in text
        assert '<a name="cohort-cuda-7f9c23eb0e27-tier-1"></a>' in text
        assert "not pooled into it" in text

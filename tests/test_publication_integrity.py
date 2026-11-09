"""Scientific scope corrections never manufacture or transfer experiment grades."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from experiments.forge.contracts import read_json, stable_hash
from experiments.forge.family_reports import _selected_cohorts, render_problem_list
from experiments.forge.leaderboard_contract import fixed_outcomes
from experiments.forge.publication_integrity import (
    CORRECTIONS, IDENTITY_FIELDS, annotate_publication, coverage_disposition, load_corrections,
)
from reports.forge import regenerate_technique_inventory as inventory

ROOT = Path(__file__).resolve().parents[1]


def original_coverage():
    source = read_json(ROOT / CORRECTIONS)["source_coverage"]
    data = subprocess.check_output(["git", "show", source["git_blob"]], cwd=ROOT)
    import hashlib
    assert hashlib.sha256(data).hexdigest() == source["sha256"]
    return json.loads(data)


def test_corrections_bind_original_observations_without_rewriting_their_verdicts():
    coverage = original_coverage()
    original = deepcopy(coverage)
    corrections = load_corrections(ROOT)
    records = {record["key"]: record for record in coverage["records"]}
    assert corrections and set(corrections) <= records.keys()
    for key in corrections:
        record = records[key]
        disposition = coverage_disposition(ROOT, record)
        assert disposition["eligible"] is False
        assert disposition["scope"] == "archived_contract_diagnostic"
        assert record["key"] == stable_hash([record[field] for field in IDENTITY_FIELDS])
    assert coverage == original


def test_correction_cannot_follow_a_task_label_to_corrected_source():
    corrections = load_corrections(ROOT)
    record = deepcopy(next(row for row in original_coverage()["records"] if row["key"] in corrections))
    record["source_digest"] = "f" * 64
    with pytest.raises(ValueError, match="invalid original identity"):
        coverage_disposition(ROOT, record)
    record["key"] = stable_hash([record[field] for field in IDENTITY_FIELDS])
    assert coverage_disposition(ROOT, record) is None


def test_same_scientific_key_cannot_silently_change_runtime_or_observed_verdict():
    corrections = load_corrections(ROOT)
    record = next(row for row in original_coverage()["records"] if row["key"] in corrections)
    for field, value in (("runtime", {"python": "different-runtime"}), ("status", "INVALID")):
        changed = {**record, field: value}
        with pytest.raises(ValueError, match="original observation/runtime"):
            coverage_disposition(ROOT, changed)


def test_annotation_preserves_exact_selected_cohort_and_all_receipt_fields():
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    before = deepcopy(publication)
    projected = annotate_publication(ROOT, publication)
    corrected = 0
    for (family, original), (_, actual) in zip(
            _selected_cohorts(publication["family_progress"]),
            _selected_cohorts(projected["family_progress"])):
        for task, old in original["tasks"].items():
            new = actual["tasks"][task]
            assert {key: value for key, value in new.items() if key != "evidence_correction"} == old
            corrected += "evidence_correction" in new
        assert actual["total"] == original["total"], family["id"]
    assert corrected > 0
    assert publication == before


def test_operational_outcomes_are_not_scientific_failures():
    statuses = {status: status for status in ("PASS", "FAIL", "BLOCKED", "ERROR", "INVALID", "INCOMPLETE", "NOT_RUN")}
    counts = fixed_outcomes(statuses, list(statuses), [])
    assert (counts["passed"], counts["failed"], counts["not_measured"]) == (1, 1, 1)
    assert counts["other"] == {"BLOCKED": 1, "ERROR": 1, "INCOMPLETE": 1, "INVALID": 1}
    text = inventory._format_tier1_bucket(list(statuses), len(statuses), bold_score=True)
    assert "1 fail" in text and "1 blocked" in text and "1 error" in text


def test_current_readout_never_extrapolates_diagnostic_coverage_to_higher_tiers(monkeypatch):
    payload = {"source_digest": "a" * 64, "view_revision": 16, "req_total": 2, "diag_total": 1,
               "rows": [{"family_id": "example", "family_label": "Example", "family_page": "example.md",
                         "candidate_id": "example", "card_rel": "example.json", "candidate_revision": "b" * 64,
                         "source_digest": "a" * 64, "req_outcomes": ["PASS", "BLOCKED"],
                         "diag_outcomes": ["ERROR"], "used_records": [], "all_preflight_blocked": False}]}
    monkeypatch.setattr(inventory, "_concluded_current_source_tier1_payload", lambda *args: payload)
    monkeypatch.setattr(inventory, "_selected_recipe_binary_task_statuses",
                        lambda *args: pytest.fail("ordinary table must not use diagnostic projection"))
    text = inventory._current_source_tier1_readouts_markdown({}, ROOT, ROOT / "report.md")
    assert "1/2 (1 blocked)" in text and "0/1 (1 error)" in text
    assert "Tier 2" not in text and "binary PASS or FAIL" not in text


def test_problem_summary_follows_actual_selected_table_not_stale_historical_claims():
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    task = "ring_extension"
    passers = [family["label"] for family, cohort in _selected_cohorts(publication["family_progress"])
               if cohort["tasks"][task]["status"] == "PASS"]
    text = render_problem_list(ROOT, publication, ROOT / "reports/forge/problems.md")
    line = next(line for line in text.splitlines() if line.startswith("| [Keep a learned ring"))
    expected = ("Selected recorded PASS: " + ", ".join(passers) if passers else "No selected recorded PASS")
    assert expected in line
    assert "Historical diagnosis (2026-10-15)" in text

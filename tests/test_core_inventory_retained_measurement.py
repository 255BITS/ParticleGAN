"""Saved reporting laws, Git declarations and immutable whole pins; no training."""
from copy import deepcopy
import subprocess

import pytest

from experiments.forge import trainer_families as families
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.views import load_view, load_tasks, task_execution_fingerprint, task_evaluation_fingerprint
from reports.forge import regenerate_technique_inventory as publication
from test_forge_trainer_families import study


@pytest.fixture
def retained(study):
    root, rows, catalogs, _, _, _ = study
    selected = rows[0]
    selected.update(trainer_family="r1r2", attempt_ids=["retained-attempt"])
    required = {a["task"] for a in load_view(root, "discriminator_stability")["assignments"]
                if a["importance"] == "required" and a["qualification_tier"] == 1}
    for name, task in load_tasks(root).items():
        if name not in required:
            continue
        contract = {"execution_sha256": task_execution_fingerprint(task),
                    "evaluation_sha256": task_evaluation_fingerprint(task),
                    "timeout_seconds": task["resources"]["timeout_seconds"]}
        digest = stable_hash(contract)
        catalogs["task_contracts"][digest] = contract
        selected["bindings"]["task_contracts"][name] = digest
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "configs"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Report Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Original measurement laws"], cwd=root, check=True)
    origin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    selected["bindings"].update(source_origin_commit=origin, recorded_source_origin_commits=[origin])
    pin = families.family_row_pin(selected, selection_kind="current_measurement", reason="Original selection",
                                  measurement_views=["discriminator_stability"])
    summary = {"certificate_validated": True, "qualification_reuse": False, "qualification_input": False,
               "candidate_id": selected["candidate_id"], "candidate_revision": selected["candidate_revision"],
               "provenance": {"canonical_result_hash": "c" * 64, "source_digest": pin["source_digest"],
                              "source_origin_commit": origin, "original_files": {}}}
    atomic_json(root / "reports/forge/technique-receipts/retained-attempt.json", summary)
    source = {**deepcopy(catalogs), "policy_fingerprint": stable_hash(load_view(root, "discriminator_stability")),
              "tier_requirements": {t: [x for x in required] if t == "1" else [str(i) for i in range(v["total"])]
                                    for t, v in selected["tiers"].items()},
              "frozen_source": {"commit": origin, "source_digests": [pin["source_digest"]]},
              "provenance": {"qualified_receipts": {"retained-attempt": {
                  "canonical_result_hash": "c" * 64, "source_digest": pin["source_digest"], "original_file_sha256": {}}}}}
    entry = {"source_commit": origin, "json_sha256": stable_hash(source), "snapshot": "fixture-source.json"}
    selected.update(publication_key=entry["json_sha256"], selection={"selection_kind": "current_measurement",
                    "reason": pin["reason"], "measurement_views": pin["measurement_views"]})
    previous = {"publication_scope": "current_technique_inventory", "view": "discriminator_stability",
                "rows": [deepcopy(selected)], "provenance": {}}
    previous["provenance"]["input_digest"] = stable_hash(previous)
    atomic_json(root / publication.CURRENT_PREFIX.with_suffix(".json"), previous)
    task_path = root / "configs/forge/tasks/two_pole.json"
    task = read_json(task_path)
    task["evaluation"]["evaluator_revision"] = {"id": "New source binding"}
    atomic_json(task_path, task)
    reports = [(entry, source, {selected["candidate_id"]: deepcopy(selected)})]
    return root, selected, pin, catalogs, previous, reports


def call(retained):
    root, row, pin, catalogs, previous, reports = retained
    return publication._recorded_measurement_pin(root, "r1r2", [row], pin, view_id="discriminator_stability",
        catalogs=catalogs, previous=previous, reports=reports, guard=families._current_pin)


def test_recorded_scope_preserves_numeric_row_and_live_guard(retained):
    root, row, pin, catalogs, _, reports = retained
    before = families.scientific_row_hash(row)
    guard = families._current_pin
    with pytest.raises(ValueError, match="current execution"):
        guard(root, "r1r2", [row], pin, view_id="discriminator_stability", catalogs=catalogs)
    with publication._retain_recorded_measurements(root, reports) as proofs:
        same, meta = families._current_pin(root, "r1r2", [row], pin, view_id="discriminator_stability", catalogs=catalogs)
    assert families._current_pin is guard
    assert same == row and families.scientific_row_hash(same) == before
    assert meta["selection_kind"] == "recorded_source_measurement" and meta["freshness"] == "stale"
    assert not meta["qualified"] and not meta["measurement_complete"] and meta["recorded_measurement_complete"]
    assert proofs[0]["original_pin"] == pin and proofs[0]["recorded_guard_pass"]
    assert [x["task"] for x in proofs[0]["contract_drift"]] == ["two_pole"]
    with pytest.raises(ValueError, match="current execution"):
        guard(root, "r1r2", [row], pin, view_id="discriminator_stability", catalogs=catalogs)


@pytest.mark.parametrize("tamper", ["new_pin", "origin", "catalog", "budget", "git_law", "publication"])
def test_stale_fallback_fails_closed(retained, tamper):
    root, row, pin, catalogs, previous, reports = retained
    if tamper == "new_pin":
        pin["reason"] = "New unrecorded selection"
    elif tamper == "origin":
        row["bindings"]["recorded_source_origin_commits"].append("0" * 40)
        previous["rows"][0]["bindings"] = deepcopy(row["bindings"])
    elif tamper == "catalog":
        catalogs["task_contracts"][row["bindings"]["task_contracts"]["two_pole"]]["evaluation_sha256"] = "x" * 64
    elif tamper == "budget":
        path = root / "configs/forge/tasks/two_pole.json"
        task = read_json(path); task["resources"]["timeout_seconds"] += 1; atomic_json(path, task)
    elif tamper == "git_law":
        reports[0][1]["policy_fingerprint"] = "0" * 64
    else:
        previous["rows"][0]["tasks"][0]["status"] = "UNKNOWN"
    if tamper == "origin":
        previous["provenance"].pop("input_digest")
        previous["provenance"]["input_digest"] = stable_hash(previous)
    with pytest.raises(ValueError):
        call(retained)


def test_stale_publication_can_republish_same_original_pin(retained):
    root, row, pin, _, previous, reports = retained
    _, metadata, _ = call(retained)
    previous["rows"][0]["selection"] = metadata
    previous["provenance"].pop("input_digest")
    previous["provenance"]["input_digest"] = stable_hash(previous)
    assert call(retained)[1]["original_pin"] == pin
    previous["rows"][0]["selection"]["original_pin"]["reason"] = "changed"
    previous["provenance"].pop("input_digest")
    previous["provenance"]["input_digest"] = stable_hash(previous)
    with pytest.raises(ValueError, match="original pin changed"):
        call(retained)


def test_unrelated_errors_and_configured_standard_never_fallback(retained):
    root, row, pin, catalogs, _, reports = retained
    row["tasks"][0]["status"] = "UNKNOWN"
    changed = families.family_row_pin(row, selection_kind="current_measurement", reason=pin["reason"],
                                     measurement_views=pin["measurement_views"])
    with publication._retain_recorded_measurements(root, reports):
        with pytest.raises(ValueError, match="every required Tier 1"):
            families._current_pin(root, "r1r2", [row], changed, view_id="discriminator_stability", catalogs=catalogs)
        standard = families.family_row_pin(row, selection_kind="configured_standard", reason="New standard")
        with pytest.raises(ValueError, match="every required Tier 1"):
            families._current_pin(root, "r1r2", [row], standard, view_id="discriminator_stability", catalogs=catalogs)


def test_stale_main_table_label_preserves_recorded_score_cells(retained):
    from experiments.forge.family_reports import render_leaderboard
    root, row, _, _, _, _ = retained
    _, metadata, _ = call(retained)
    row["selection"] = metadata
    count = {"passed": 0, "total": 6, "executed": 6, "incomplete": 0}
    cohort = {"anchor": "original", "backend": "cuda", "views": [],
              "total": count, "tiers": {str(i): count for i in range(1, 4)}}
    family = {"id": "r1r2", "label": "R1/R2", "page": "reports/forge/families/r1r2.md",
              "cohorts": [cohort], "tags": []}
    result = {"rows": [row], "family_progress": {"families": [family]}, "provenance": {"input_digest": "fixture"}}
    page = root / "reports/forge/technique-inventory.md"
    normal = render_leaderboard(root, result, page)
    labeled = publication._stale_measurement_labels(normal, result, root, page)
    assert "Recorded source measurement · **stale**; no current qualification" in labeled
    old_cells = next(line for line in normal.splitlines() if line.startswith("| **")).split(" | ")[1:]
    new_cells = next(line for line in labeled.splitlines() if line.startswith("| **")).split(" | ")[1:]
    assert old_cells == new_cells

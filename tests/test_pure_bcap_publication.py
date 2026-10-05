"""Pure BCAP publication infrastructure; synthetic receipts, no training."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION, scientific_row_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pure_bcap_publisher", ROOT / "reports/forge/pure-bcap/publish.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)
export_spec = importlib.util.spec_from_file_location("pure_bcap_exporter", ROOT / "reports/forge/pure-bcap/export.py")
exporter = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(exporter)
archive_spec = importlib.util.spec_from_file_location("pure_bcap_archiver", ROOT / "reports/forge/pure-bcap/archive.py")
archiver = importlib.util.module_from_spec(archive_spec)
archive_spec.loader.exec_module(archiver)
SOURCE, COMMIT = "a" * 64, "b" * 40


@pytest.fixture
def completed(tmp_path):
    required = [f"task-{i}" for i in range(6)]
    names = [*required, "audit"]
    trials, candidates, rows, paid, counts = [], [], [], [], Counter()
    plan = {"round": "unit-pure-round", "source_digest": SOURCE, "view_revision": 5,
            "configuration_count": 10, "campaign_cap_seconds": 100, "required_task_ids": required,
            "task_ids": names, "specs": [f"configs/forge/searches/loss-{i}.json" for i in range(5)], "trials": []}
    for index in range(10):
        candidate = "bcap-pure--" + stable_hash(index)
        revision, configuration = stable_hash((index, "revision")), f"{index:064x}"
        tasks, trial_tasks = [], []
        for task_index, name in enumerate(names):
            attempt = f"attempt-{index}-{task_index}"
            status = "PASS" if task_index == 6 or task_index < (2 if index in (2, 7) else 1) else "FAIL"
            importance = "required" if name in required else "diagnostic"
            if importance == "required":
                counts[status] += 1
            receipt_path = publication.REPORT / "receipts" / (attempt + ".json")
            measurement = {"task_id": name, "gate_status": status, "metrics": {},
                           "compatibility_key": stable_hash((index, name)),
                           "cost": {"execution_seconds": 1 / 7}, "reason": "unit fixture"}
            atomic_json(tmp_path / receipt_path, {
                "attempt_id": attempt, "candidate_id": candidate, "candidate_revision": revision,
                "campaign_id": plan["round"], "attempt_status": "completed", "task_results": [measurement],
                "qualification_input": False, "certificate_validated": True,
                "provenance": {"source_digest": SOURCE, "source_origin_commit": COMMIT,
                               "canonical_result_hash": stable_hash(attempt)}})
            paid.append({"attempt_id": attempt, "candidate_id": candidate, "candidate_revision": revision,
                "campaign_id": plan["round"], "round": plan["round"], "attempt_status": "completed",
                "paid_wall_seconds": 1 / 7, "source_digest": SOURCE, "source_commit": COMMIT,
                "canonical_result_hash": stable_hash(attempt), "task_statuses": {name: status},
                "receipt": receipt_path.as_posix(), "receipt_sha256": file_hash(tmp_path / receipt_path)})
            gif = tmp_path / publication.REPORT / "media" / configuration[:12] / (name + ".gif")
            gif.parent.mkdir(parents=True, exist_ok=True)
            # Fixtures test integrity and ownership; these are not experiment media.
            gif.write_bytes(b"unit-test-placeholder")
            media = {"task_id": name, "recorded_grade": status,
                     "kind": "actual_training_saved_observations_gif", "gif_sha256": file_hash(gif),
                     "optimizer_updates_added": 0, "sampling_draws_added": 0}
            tasks.append({**{key: measurement[key] for key in ("metrics", "compatibility_key", "cost", "reason")},
                          "task_id": name, "attempt_id": attempt, "status": status, "importance": importance,
                          "receipt": receipt_path.as_posix(), "receipt_sha256": file_hash(tmp_path / receipt_path),
                          "gif_receipt": media})
            trial_tasks.append({"task": name, "importance": importance, "qualification_tier": 1,
                                "gate_status": status, "compatibility_key": stable_hash((index, name))})
        trial = {"candidate_id": candidate, "candidate_revision": revision, "configuration_id": configuration,
                 "source_digest": SOURCE, "submission_status": "completed", "tasks": trial_tasks}
        trials.append(trial)
        plan["trials"].append({"candidate_id": candidate, "candidate_revision": revision})
        candidates.append({"candidate_id": candidate, "candidate_revision": revision,
            "configuration_id": configuration, "required_total": 6, "tasks": tasks,
            "cost": {"new_paid_wall_seconds": 1}})
        rows.append({"candidate_id": candidate, "candidate_revision": revision,
            "bindings": {"source_digest": SOURCE}, "runtime_cohort": {"execution_backend": "cuda"},
            "tasks": [{"task_id": t["task_id"], "status": t["status"]} for t in tasks if t["importance"] == "required"],
            "nonrequired_tasks": [{"task_id": "audit", "status": "PASS"}],
            "attempt_ids": [t["attempt_id"] for t in tasks]})
    for index, spec_path in enumerate(plan["specs"]):
        atomic_json(tmp_path / "reports/forge/configuration-search" / (Path(spec_path).stem + ".json"),
                    {"source_digest": SOURCE, "trials": trials[2*index:2*index+2]})
    readout = {"scope": "finite_pure_bcap_initial_readout", "round": plan["round"],
        "qualification_input": False, "default_adoption": False, "source_digest": SOURCE,
        "source_commit": COMMIT, "view_revision": 5, "selection": select_configuration(trials, 1),
        "candidates": candidates, "unique_paid_attempts": 70, "required_counts": dict(counts),
        "paid_attempts": paid,
        "diagnostic_counts": {"PASS": 10}, "paid_wall_seconds": 10, "maximum_paid_seconds": 100,
        "cost_note": "Shared campaign totals overlap; count unique paid attempts once."}
    atomic_json(tmp_path / publication.REPORT / "plans.json", plan)
    _save_readout(tmp_path, readout)
    return tmp_path, plan, readout, {"rows": rows}


def _save_readout(root, readout):
    value = deepcopy(readout)
    value.pop("input_digest", None)
    value["input_digest"] = stable_hash(value)
    atomic_json(root / publication.REPORT / "readout.json", value)


def test_complete_readout_uses_whole_recipe_counts_and_hash_tie_break(completed):
    root, plan, readout, frozen = completed
    _, verified, selection = publication.completed_readout(root)
    assert selection["selected_candidate_id"] == readout["candidates"][2]["candidate_id"]
    assert selection["required_pass_count"] == 2 and selection["required_total"] == 6
    assert verified["paid_wall_seconds"] == 10  # not five repeated campaign totals
    pin = publication.selected_pin(frozen, plan, verified, selection)
    assert pin["selection_kind"] == "current_measurement"
    assert pin["measurement_views"] == ["discriminator_stability"]
    assert pin["measurement_tasks"] == ["audit"]
    assert pin["scientific_row_sha256"] == scientific_row_hash(frozen["rows"][2])


@pytest.mark.parametrize("change", ["missing_candidate", "missing_task", "duplicate_attempt", "counts", "cost", "media", "source", "selection"])
def test_incomplete_or_rebound_readout_is_rejected_before_publication(completed, change):
    root, _, readout, _ = completed
    if change == "missing_candidate":
        readout["candidates"].pop()
    elif change == "missing_task":
        readout["candidates"][0]["tasks"].pop()
    elif change == "duplicate_attempt":
        readout["candidates"][1]["tasks"][0]["attempt_id"] = readout["candidates"][0]["tasks"][0]["attempt_id"]
    elif change == "counts":
        readout["required_counts"]["PASS"] += 1
    elif change == "cost":
        readout["paid_wall_seconds"] *= 5
    elif change == "media":
        readout["candidates"][0]["tasks"][0]["gif_receipt"]["sampling_draws_added"] = 1
    elif change == "source":
        readout["source_digest"] = "changed"
    elif change == "selection":
        readout["selection"]["selected_candidate_id"] = readout["candidates"][0]["candidate_id"]
    _save_readout(root, readout)
    with pytest.raises(ValueError):
        publication.completed_readout(root)


def test_selected_row_cannot_pool_a_passing_cell_from_another_candidate(completed):
    _, plan, readout, frozen = completed
    frozen["rows"][2]["tasks"][5]["status"] = "PASS"
    with pytest.raises(ValueError, match="whole-recipe"):
        publication.selected_pin(frozen, plan, readout, readout["selection"])


def _strict_paid_fixture(completed, monkeypatch):
    """Bind a 70+39 roster to immutable synthetic certified projections."""
    root, plan, readout, _ = completed
    original = deepcopy(plan)
    context = {"candidate_id": "original-canceled-context", "candidate_revision": stable_hash("original-context")}
    original["trials"].append(context)
    atomic_json(root / publication.REPORT / "plans.json", original)
    for index in range(39):
        attempt = f"original-context-{index}"
        path = publication.REPORT / "receipts" / (attempt + ".json")
        receipt = deepcopy(read_json(root / readout["paid_attempts"][0]["receipt"]))
        receipt.update(attempt_id=attempt, **context)
        receipt["provenance"]["canonical_result_hash"] = stable_hash(attempt)
        receipt["task_results"][0]["cost"]["execution_seconds"] = .25
        atomic_json(root / path, receipt)
        readout["paid_attempts"].append({"attempt_id": attempt, **context,
            "round": plan["round"], "campaign_id": plan["round"], "attempt_status": "completed",
            "source_digest": SOURCE, "source_commit": COMMIT, "paid_wall_seconds": .25,
            "canonical_result_hash": stable_hash(attempt), "task_statuses": {"task-0": "PASS"},
            "receipt": path.as_posix(), "receipt_sha256": file_hash(root / path)})
    readout.update(original_unselected_paid_attempt_ids=[f"original-context-{index}" for index in range(39)],
                   original_unselected_paid_wall_seconds=9.75, paid_wall_seconds=19.75, unique_paid_attempts=109)
    plan.update(expected_paid_attempt_count=109, selected_measurement_attempt_count=70,
                original_unselected_paid_attempt_count=39,
                execution_rounds=[{"round": plan["round"], "campaign_cap_seconds": 100,
                    "paid_attempt_ids_frozen": [record["attempt_id"] for record in readout["paid_attempts"]],
                    "paid_wall_seconds_frozen": 19.75}])
    atomic_json(root / publication.REPORT / "publication-plans.json", plan)
    projections = {record["attempt_id"]: read_json(root / record["receipt"]) for record in readout["paid_attempts"]}
    monkeypatch.setattr(publication.inventory, "project_receipt", lambda root, attempt: deepcopy(projections[attempt]))
    _save_readout(root, readout)
    return root, readout


def test_strict_final_roster_retains_all_109_certified_paid_attempts(completed, monkeypatch):
    root, _ = _strict_paid_fixture(completed, monkeypatch)
    _, verified, _ = publication.completed_readout(root)
    assert len(verified["paid_attempts"]) == 109 and verified["paid_wall_seconds"] == 19.75


@pytest.mark.parametrize("change", ["drop_context", "duplicate", "receipt_hash", "source", "revision",
                                    "canonical_result", "task_status", "charge", "nonfinite_charge",
                                    "rebound_projection", "displayed_status", "displayed_metrics",
                                    "displayed_compatibility"])
def test_strict_final_roster_rejects_missing_or_rebound_evidence(completed, monkeypatch, change):
    root, readout = _strict_paid_fixture(completed, monkeypatch)
    record = readout["paid_attempts"][-1]
    if change == "drop_context":
        readout["paid_attempts"] = readout["paid_attempts"][:70]
    elif change == "duplicate":
        readout["paid_attempts"][-1] = deepcopy(readout["paid_attempts"][-2])
    elif change == "receipt_hash":
        record["receipt_sha256"] = "0" * 64
    elif change == "source":
        record["source_digest"] = "changed"
    elif change == "revision":
        record["candidate_revision"] = "changed"
    elif change == "canonical_result":
        record["canonical_result_hash"] = "changed"
    elif change == "task_status":
        record["task_statuses"]["task-0"] = "FAIL"
    elif change == "charge":
        record["paid_wall_seconds"] += 1
    elif change == "nonfinite_charge":
        record["paid_wall_seconds"] = float("inf")
        with pytest.raises(ValueError):
            _save_readout(root, readout)
        return
    elif change == "rebound_projection":
        receipt = read_json(root / record["receipt"])
        receipt["provenance"]["canonical_result_hash"] = "changed"
        atomic_json(root / record["receipt"], receipt)
        record.update(canonical_result_hash="changed", receipt_sha256=file_hash(root / record["receipt"]))
    else:
        # A nonwinner's scalar projection must remain tied to its sealed receipt.
        task = readout["candidates"][9]["tasks"][1]
        if change == "displayed_status":
            task["status"] = "PASS"; task["gif_receipt"]["recorded_grade"] = "PASS"
            readout["required_counts"]["PASS"] += 1; readout["required_counts"]["FAIL"] -= 1
        elif change == "displayed_metrics":
            task["metrics"] = {"fabricated": 1}
        else:
            task["compatibility_key"] = "changed"
    _save_readout(root, readout)
    with pytest.raises(ValueError):
        publication.completed_readout(root)


def _composition_fixture(completed, monkeypatch, *, dual=False):
    root, plan, readout, raw = completed
    real_snapshot = publication.inventory._snapshot
    before = _pending_fixture(root, monkeypatch)
    before["tier_requirements"] = {"1": plan["required_task_ids"]}
    manifest = read_json(root / publication.inventory.EVIDENCE_MANIFEST)
    manifest["tier_requirements"] = before["tier_requirements"]
    atomic_json(root / publication.inventory.EVIDENCE_MANIFEST, manifest)
    before["provenance"]["evidence_manifest_sha256"] = stable_hash(manifest)
    before["provenance"].pop("input_digest")
    before["provenance"]["input_digest"] = stable_hash(before)
    atomic_json(root / publication.inventory.CURRENT_PREFIX.with_suffix(".json"), before)
    source2, commit2 = "c" * 64, "d" * 40
    groups = [(SOURCE, COMMIT, raw["rows"][:2]), (source2, commit2, raw["rows"][2:])] if dual else [(SOURCE, COMMIT, raw["rows"])]
    reports = {}
    for source, commit, rows in groups:
        proofs = {}
        for row in rows:
            row["bindings"]["source_digest"] = source
            row["tiers"] = {"1": {"total": 6, "passed": sum(task["status"] == "PASS" for task in row["tasks"])}}
            row.update(status="FAIL", qualified_tier=0, cohort=stable_hash((row["candidate_id"], source)))
            for attempt in row["attempt_ids"]:
                candidate = next(candidate for candidate in readout["candidates"] if candidate["candidate_id"] == row["candidate_id"])
                task = next(task for task in candidate["tasks"] if task["attempt_id"] == attempt)
                summary = read_json(root / task["receipt"])
                summary.update(candidate_id=row["candidate_id"], qualification_reuse=False)
                summary["provenance"].update(source_digest=source, source_origin_commit=commit,
                    canonical_result_hash=stable_hash(attempt), original_files={"result": {"sha256": stable_hash((attempt, "original"))}})
                atomic_json(root / task["receipt"], summary); task["receipt_sha256"] = file_hash(root / task["receipt"])
                record = next(record for record in readout["paid_attempts"] if record["attempt_id"] == attempt)
                record.update(source_digest=source, source_commit=commit, receipt_sha256=task["receipt_sha256"])
                atomic_json(root / "reports/forge/technique-receipts" / (attempt + ".json"), summary)
                atomic_json(root / "reports/forge/attempts" / attempt / "result.json", {"raw": {"finished_at": "2026-10-05T00:00:00Z"}})
                proofs[attempt] = {"canonical_result_hash": summary["provenance"]["canonical_result_hash"],
                    "source_digest": source, "original_file_sha256": {"result": summary["provenance"]["original_files"]["result"]["sha256"]}}
        report = {**{key: before[key] for key in publication.inventory.POLICY_FIELDS},
            "publication_scope": "frozen_source", "frozen_source": {"commit": commit, "source_digests": [source]},
            "rows": rows, "provenance": {"qualified_receipts": proofs}}
        report["provenance"]["input_digest"] = stable_hash(report)
        reports[commit] = report
    if dual:
        cohorts = [{"source_commit": commit, "source_digest": source, "candidate_ids": [row["candidate_id"] for row in rows]}
                   for source, commit, rows in groups]
        plan.pop("source_digest"); plan["source_cohorts"] = cohorts
        atomic_json(root / publication.REPORT / "publication-plans.json", plan)
        # The historical original plan stays the old-source identity.
        old_plan = read_json(root / publication.REPORT / "plans.json")
        assert old_plan["source_digest"] == SOURCE
        readout.pop("source_digest"); readout.pop("source_commit"); readout["source_cohorts"] = cohorts
        trials = []
        for index, spec_path in enumerate(plan["specs"]):
            path = root / "reports/forge/configuration-search" / (Path(spec_path).stem + ".json")
            summary = read_json(path)
            if index:
                summary["source_digest"] = source2
                for trial in summary["trials"]: trial["source_digest"] = source2
            atomic_json(path, summary); trials.extend(summary["trials"])
        readout["selection"] = select_configuration(trials, 1)
    _save_readout(root, readout)
    def snapshot(root, entry, manifest):
        if entry["snapshot"] == "saved.json":
            return {}, {row["candidate_id"]: row for row in before["rows"]}
        return real_snapshot(root, entry, manifest)
    monkeypatch.setattr(publication.inventory, "_snapshot", snapshot)
    def regenerate(*args, **kwargs):
        path = Path(str(kwargs["output_prefix"]) + ".json")
        atomic_json(path, reports[kwargs["source_commit"]])
        return {"json": str(path)}
    monkeypatch.setattr(publication.inventory, "regenerate", regenerate)
    calls = []
    def fresh_pin(root, family, rows, pin, **kwargs):
        assert family == "bcap-pure"  # Retained families must not be readmitted.
        calls.append(pin["selection_kind"])
        selected = next(row for row in rows if row["candidate_id"] == pin["candidate_id"])
        assert publication.family_row_pin(selected, selection_kind=pin["selection_kind"], reason=pin["reason"],
            measurement_views=pin.get("measurement_views"), measurement_tasks=pin.get("measurement_tasks")) == pin
        return selected, {"selection_kind": pin["selection_kind"], "qualified": False, "default_adoption": False}
    monkeypatch.setattr("experiments.forge.trainer_families._current_pin", fresh_pin)
    monkeypatch.setattr(publication.inventory, "_common26_display_projection", lambda *args: {"fixture": True})
    monkeypatch.setattr(publication.inventory, "_current_markdown",
                        lambda result, root, path: "One current table\n" + publication.display_section(root, path))
    from experiments.forge.trainer_families import REGISTRY
    atomic_json(root / REGISTRY, {"schema_version": 1, "families": [{"id": "bcap-pure", "label": "Pure BCAP",
        "candidates": ["canonical"], "canonical_candidate": "canonical"}]})
    return root, plan, readout, before, reports, calls


@pytest.mark.parametrize("current_id", ["bcap-pure", "bcap"])
def test_publication_composes_one_new_pin_without_readmitting_retained_science(completed, monkeypatch, current_id):
    root, _, readout, before, _, calls = _composition_fixture(completed, monkeypatch)
    if current_id == "bcap":
        from experiments.forge.trainer_families import REGISTRY
        registry = read_json(root / REGISTRY)
        registry["families"][0].update(id=current_id, recorded_id="bcap-pure")
        registry["families"].append({"id": "bcap-with-k3p", "recorded_id": "bcap", "label": "BCAP with K3P",
                                      "candidates": ["saved-bcap"], "canonical_candidate": "saved-bcap"})
        atomic_json(root / REGISTRY, registry)
        for key in ("rows", "configuration_rows"):
            before[key] = [{**row, "trainer_family": "bcap-with-k3p"} for row in before[key]]
        before["provenance"].pop("input_digest")
        before["provenance"]["input_digest"] = stable_hash(before)
        atomic_json(root / publication.inventory.CURRENT_PREFIX.with_suffix(".json"), before)
    old_card = read_json(root / CURRENT_SELECTION)
    result = publication.publish(root)
    after = read_json(root / publication.inventory.CURRENT_PREFIX.with_suffix(".json"))
    assert after["rows"][-1]["trainer_family"] == current_id
    assert after["rows"][:1] == before["rows"]
    assert after["task_contracts"] == before["task_contracts"]
    assert read_json(root / CURRENT_SELECTION)["selections"][:1] == old_card["selections"]
    assert calls == ["current_measurement"]
    assert result["preserved_families"] == 1 and result["paid_wall_seconds"] == 10
    assert after["publication_refresh"]["new_sources_registered"] is True
    text = publication.display_section(root, root / "reports/forge/technique-inventory.md")
    assert "2/6" in text and "70 unique attempts" in text and "|" not in text
    first_card = (root / CURRENT_SELECTION).read_bytes()
    again = publication.publish(root)
    assert again == result and (root / CURRENT_SELECTION).read_bytes() == first_card


def test_publication_rolls_back_every_public_byte_on_late_navigation_failure(completed, monkeypatch):
    root, _, _, _, _, _ = _composition_fixture(completed, monkeypatch)
    originals = {path: path.read_bytes() for path in publication._publication_paths(root) if path.is_file()}
    monkeypatch.setattr(publication.inventory, "_family_pages", lambda *args: (_ for _ in ()).throw(ValueError("navigation failed")))
    with pytest.raises(ValueError, match="navigation failed"):
        publication.publish(root)
    assert {path: path.read_bytes() for path in publication._publication_paths(root) if path.is_file()} == originals


def test_publication_rejects_tampered_frozen_snapshot_before_selection_change(completed, monkeypatch):
    root, _, _, _, reports, _ = _composition_fixture(completed, monkeypatch)
    before = (root / CURRENT_SELECTION).read_bytes()
    reports[COMMIT]["rows"][2]["tasks"][4]["status"] = "PASS"
    with pytest.raises(ValueError, match="snapshot input digest"):
        publication.publish(root)
    assert (root / CURRENT_SELECTION).read_bytes() == before


def test_preservation_includes_denominators_and_metrics():
    before = {"view": "view", "view_revision": 1, "policy_fingerprint": "policy", "tier_requirements": {"1": ["task"]},
              "rows": [{"trainer_family": "bcap", "candidate_id": "old", "metrics": {"cdf_ks": .2}}]}
    after = deepcopy(before)
    after["rows"][0]["metrics"]["cdf_ks"] = .01
    with pytest.raises(ValueError, match="existing selected"):
        publication.preserved_science(before, after)


def test_current_generator_adds_only_bound_readout_navigation(completed, monkeypatch):
    root, _, readout, _ = completed
    receipt = {"scope": "pure_bcap_initial_current_measurement", "qualification_input": False,
        "default_adoption": False, "readout": (publication.REPORT / "readout.json").as_posix(),
        "readout_sha256": file_hash(root / publication.REPORT / "readout.json"),
        "selection": readout["selection"], "source_digest": SOURCE, "source_commit": COMMIT,
        "unique_paid_attempts": 70, "paid_wall_seconds": 10}
    atomic_json(root / publication.REPORT / "publication.json", receipt)
    (root / publication.REPORT / "publish.py").write_bytes(Path(publication.__file__).read_bytes())
    monkeypatch.setattr("experiments.forge.family_reports.render_leaderboard",
                        lambda root, result, path: "| One current family table |\n")
    result = {"family_progress": {"synthetic_navigation_fixture": True}}
    before = deepcopy(result)
    text = publication.inventory._current_markdown(result, root, root / "reports/forge/technique-inventory.md")
    assert text.count("| One current family table |") == 1
    assert "[Pure BCAP initial readout](pure-bcap/README.md)" in text
    assert "2/6" in text and result == before
    receipt["paid_wall_seconds"] = 50
    atomic_json(root / publication.REPORT / "publication.json", receipt)
    with pytest.raises(ValueError, match="navigation binding"):
        publication.inventory._current_markdown(result, root, root / "reports/forge/technique-inventory.md")


def test_two_source_composition_keeps_both_snapshots_and_original_plan(completed, monkeypatch):
    root, plan, readout, before, _, calls = _composition_fixture(completed, monkeypatch, dual=True)
    old_plan = (root / publication.REPORT / "plans.json").read_bytes()
    result = publication.publish(root)
    assert calls == ["current_measurement"] and result["paid_wall_seconds"] == 10
    manifest = read_json(root / publication.inventory.EVIDENCE_MANIFEST)
    assert {entry.get("source_commit") for entry in manifest["cohorts"][1:]} == {COMMIT, "d" * 40}
    assert (root / publication.REPORT / "plans.json").read_bytes() == old_plan
    after = read_json(root / publication.inventory.CURRENT_PREFIX.with_suffix(".json"))
    assert after["rows"][:1] == before["rows"]
    assert len(after["evidence_rows"]) == len(before["evidence_rows"]) + 10


def test_original_source_winner_keeps_historical_pin_without_live_word_claim(completed, monkeypatch):
    root, _, readout, _, _, calls = _composition_fixture(completed, monkeypatch, dual=True)
    # Exercise a complete original-source winner using consistent synthetic
    # grades and the same whole-recipe selector as production.
    summaries = []
    for path in sorted((root / "reports/forge/configuration-search").glob("*.json")):
        summary = read_json(path)
        for trial in summary["trials"]:
            trial["configuration_id"] = "0" * 64 if trial["candidate_id"] == readout["candidates"][0]["candidate_id"] else "f" * 64
            if trial["candidate_id"] == readout["candidates"][0]["candidate_id"]:
                # Its original PASS count becomes the shared maximum2.
                trial["tasks"][1]["gate_status"] = "PASS"
        atomic_json(path, summary); summaries.extend(summary["trials"])
    original = readout["candidates"][0]
    original["tasks"][1]["status"] = "PASS"
    original["configuration_id"] = "0" * 64
    # Rebind the synthetic fixture's receipt/GIF scalar metadata consistently.
    task = original["tasks"][1]
    task["gif_receipt"]["recorded_grade"] = "PASS"
    compact = read_json(root / task["receipt"])
    compact["task_results"][0]["gate_status"] = "PASS"
    atomic_json(root / task["receipt"], compact)
    task["receipt_sha256"] = file_hash(root / task["receipt"])
    record = next(record for record in readout["paid_attempts"] if record["attempt_id"] == task["attempt_id"])
    record.update(task_statuses={task["task_id"]: "PASS"}, receipt_sha256=task["receipt_sha256"])
    for task in original["tasks"]:
        old = root / publication.REPORT / "media" / (f"{0:064x}"[:12]) / (task["task_id"] + ".gif")
        target = root / publication.REPORT / "media" / original["configuration_id"][:12] / (task["task_id"] + ".gif")
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(old.read_bytes())
    readout["required_counts"]["PASS"] += 1; readout["required_counts"]["FAIL"] -= 1
    readout["selection"] = select_configuration(summaries, 1)
    _save_readout(root, readout)
    # Reconstructed fixture report must have the same saved numerical grade.
    # Its snapshots are controlled test data, never experiment evidence.
    # Reuse source snapshots from the helper by intercepting the saved function.
    previous = publication.inventory.regenerate
    def regrade(*args, **kwargs):
        metadata = previous(*args, **kwargs)
        path = Path(metadata["json"]); report = read_json(path)
        if kwargs["source_commit"] == COMMIT:
            row = next(row for row in report["rows"] if row["candidate_id"] == original["candidate_id"])
            row["tasks"][1]["status"] = "PASS"; row["tiers"]["1"]["passed"] = 2
            report["provenance"].pop("input_digest"); report["provenance"]["input_digest"] = stable_hash(report)
            atomic_json(path, report)
        return metadata
    monkeypatch.setattr(publication.inventory, "regenerate", regrade)
    publication.publish(root)
    pin = read_json(root / CURRENT_SELECTION)["selections"][-1]
    receipt = read_json(root / publication.REPORT / "publication.json")
    assert pin["selection_kind"] == "historical_incumbent" and "measurement_views" not in pin
    assert calls == ["historical_incumbent"]
    assert receipt["selected_live_measurement_contracts_validated"] is False
    assert "no latest-checkout" in receipt["selection_scope"]


def _export_fixture(completed, monkeypatch):
    """Synthetic physical receipts exercise reporting without training."""
    from dataclasses import asdict
    from particlegan.recipes import get_recipe
    root, plan, readout, _ = completed
    records, projected, trials = [], {}, []
    for index, spec_path in enumerate(plan["specs"]):
        path = root / "reports/forge/configuration-search" / (Path(spec_path).stem + ".json")
        summary = read_json(path)
        for trial in summary["trials"]:
            candidate = next(candidate for candidate in readout["candidates"] if candidate["candidate_id"] == trial["candidate_id"])
            recipe = asdict(get_recipe("bcap"))
            trial["declaration"] = {"resolved_configuration_recipe": recipe}
            trial["cost"] = candidate["cost"]
            for task, displayed in zip(trial["tasks"], candidate["tasks"]):
                attempt = displayed["attempt_id"]
                task.update(attempt_id=attempt, metrics={"scalar": .2}, cost={"wall_seconds": 1/7})
                receipt = read_json(root / displayed["receipt"])
                receipt.update(candidate_id=candidate["candidate_id"], campaign_id=plan["round"],
                    attempt_status="completed", task_results=[{"task_id": displayed["task_id"], "gate_status": displayed["status"],
                        "metrics": task["metrics"], "cost": task["cost"], "compatibility_key": task["compatibility_key"], "reason": None}])
                receipt["provenance"]["canonical_result_hash"] = stable_hash(attempt)
                projected[attempt] = receipt
                records.append({"attempt_id": attempt, "candidate_id": candidate["candidate_id"],
                    "candidate_revision": candidate["candidate_revision"], "campaign_id": plan["round"],
                    "round": plan["round"], "paid_wall_seconds": 1/7, "source_digest": SOURCE,
                    "source_commit": COMMIT, "attempt_status": "completed",
                    "canonical_result_hash": stable_hash(attempt), "task_statuses": {displayed["task_id"]: displayed["status"]}})
            trials.append(trial)
        summary["selection"] = select_configuration(summary["trials"], 1)
        atomic_json(path, summary)
    monkeypatch.setattr(exporter, "paid_attempts", lambda root, plan: deepcopy(records))
    monkeypatch.setattr(exporter, "project_receipt", lambda root, attempt: deepcopy(projected[attempt]))
    def media(directory, destination):
        attempt = Path(directory).name
        task = projected[attempt]["task_results"][0]
        path = Path(destination) / (task["task_id"] + ".gif")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-media-fixture")
        return [{"task_id": task["task_id"], "recorded_grade": task["gate_status"],
            "kind": "actual_training_saved_observations_gif", "gif_sha256": file_hash(path),
            "optimizer_updates_added": 0, "sampling_draws_added": 0}]
    monkeypatch.setattr(exporter, "export_attempt", media)
    return root, plan, readout, records, projected, trials


def test_complete_export_keeps_two_whole_sources_and_original_cancelled_cost(completed, monkeypatch):
    root, plan, _, records, projected, _ = _export_fixture(completed, monkeypatch)
    repaired_source, repaired_commit = "c" * 64, "d" * 40
    original_ids = [candidate["candidate_id"] for candidate in plan["trials"][:2]]
    repaired_ids = [candidate["candidate_id"] for candidate in plan["trials"][2:]]
    for record in records:
        round_id = "repair-round" if record["candidate_id"] in repaired_ids else "original-round"
        record.update(round=round_id, campaign_id=round_id)
        projected[record["attempt_id"]]["campaign_id"] = round_id
        if record["candidate_id"] in repaired_ids:
            record.update(source_digest=repaired_source, source_commit=repaired_commit)
            projected[record["attempt_id"]]["provenance"].update(source_digest=repaired_source,
                                                                source_origin_commit=repaired_commit)
    for spec_path in plan["specs"][1:]:
        path = root / "reports/forge/configuration-search" / (Path(spec_path).stem + ".json")
        summary = read_json(path); summary["source_digest"] = repaired_source
        for trial in summary["trials"]:
            trial["source_digest"] = repaired_source
        atomic_json(path, summary)
    attempt = "original-cancelled-paid"
    projected[attempt] = {"attempt_id": attempt, "candidate_id": "omitted-original-candidate", "candidate_revision": "old",
        "qualification_input": False, "certificate_validated": True, "campaign_id": "original-round",
        "attempt_status": "cancelled", "task_results": [{"task_id": "word", "gate_status": "INCOMPLETE", "cost": {"wall_seconds": 2}}],
        "provenance": {"source_digest": SOURCE, "source_origin_commit": COMMIT, "canonical_result_hash": stable_hash(attempt)}}
    records.append({"attempt_id": attempt, "candidate_id": "omitted-original-candidate", "paid_wall_seconds": 2,
                    "source_digest": SOURCE, "source_commit": COMMIT, "candidate_revision": "old",
                    "campaign_id": "original-round", "round": "original-round", "attempt_status": "cancelled",
                    "canonical_result_hash": stable_hash(attempt), "task_statuses": {"word": "INCOMPLETE"}})
    old_plan = read_json(root / publication.REPORT / "plans.json")
    old_plan["trials"].append({"candidate_id": "omitted-original-candidate", "candidate_revision": "old"})
    atomic_json(root / publication.REPORT / "plans.json", old_plan)
    plan.pop("source_digest")
    plan["source_cohorts"] = [{"source_digest": SOURCE, "source_commit": COMMIT, "candidate_ids": original_ids},
        {"source_digest": repaired_source, "source_commit": repaired_commit, "candidate_ids": repaired_ids}]
    plan["execution_rounds"] = [{"round": "original-round", "campaign_id": "original-round", "campaign_cap_seconds": 100},
        {"round": "repair-round", "campaign_id": "repair-round", "campaign_cap_seconds": 100}]
    atomic_json(root / publication.REPORT / "publication-plans.json", plan)
    result = exporter.export_readout(root)
    assert result["selected_measurement_attempts"] == 70 and result["unique_paid_attempts"] == 71
    assert result["paid_wall_seconds"] == pytest.approx(12)
    assert result["original_unselected_paid_wall_seconds"] == 2
    assert result["original_unselected_paid_attempt_ids"] == [attempt]
    assert result["source_cohorts"] == plan["source_cohorts"]
    assert (root / publication.REPORT / "receipts" / (attempt + ".json")).is_file()
    _, verified, _ = publication.completed_readout(root)
    assert verified["paid_wall_seconds"] == pytest.approx(12)


def test_partial_export_keeps_unknown_cells_and_cannot_select_or_publish(completed, monkeypatch):
    root, plan, _, records, _, _ = _export_fixture(completed, monkeypatch)
    path = root / "reports/forge/configuration-search" / (Path(plan["specs"][0]).stem + ".json")
    summary = read_json(path)
    missing = summary["trials"][0]["tasks"].pop()
    records[:] = [record for record in records if record["attempt_id"] != missing["attempt_id"]]
    summary["trials"][0]["tasks"].append({**missing, "gate_status": "UNKNOWN", "attempt_id": None})
    summary["trials"][0]["submission_status"] = "cancelled"
    summary["trials"][0]["cost"]["new_paid_wall_seconds"] = 6/7
    summary["selection"] = select_configuration(summary["trials"], 1)
    atomic_json(path, summary)
    result = exporter.export_readout(root, partial=True)
    assert result["scope"] == "partial_pure_bcap_initial_readout" and "selection" not in result
    assert result["unique_paid_attempts"] == 69 and result["paid_wall_seconds"] == pytest.approx(10 - 1/7)
    candidate = next(candidate for candidate in result["candidates"] if candidate["candidate_id"] == summary["trials"][0]["candidate_id"])
    assert candidate["unmeasured_task_ids"] == [missing["task"]]
    assert not (root / publication.REPORT / "publication.json").exists()
    with pytest.raises(ValueError, match="terminal"):
        exporter.export_readout(root)


def test_export_rejects_source_mixed_inside_one_whole_candidate(completed, monkeypatch):
    root, _, _, _, projected, _ = _export_fixture(completed, monkeypatch)
    projected[next(iter(projected))]["provenance"]["source_origin_commit"] = "changed-origin"
    with pytest.raises(ValueError, match="source cohort"):
        exporter.export_readout(root)


def test_archive_includes_cancelled_cost_context_and_byte_exact_relocation(tmp_path):
    import tarfile
    root = tmp_path / "checkout"; root.mkdir()
    old = tmp_path / "original"; old.mkdir()
    old_artifact = old / "attempt"; old_artifact.mkdir(); (old_artifact / "stdout.log").write_bytes(b"raw\x00stdout\n")
    old_source = old / "snapshot"; old_source.mkdir(); (old_source / "source.py").write_bytes(b"frozen = 1\n")
    attempt, round_id = "paid-cancelled", "original-round"
    queue = root / "runs/forge" / round_id / "queue"
    local = queue / round_id / attempt; local.mkdir(parents=True)
    (local / "stdout.log").write_bytes((old_artifact / "stdout.log").read_bytes())
    source = queue / "snapshots" / SOURCE; source.mkdir(parents=True)
    (source / "source.py").write_bytes((old_source / "source.py").read_bytes())
    directory = root / "reports/forge/attempts" / attempt
    atomic_json(directory / "request.json", {"request": {"campaign_id": round_id,
        "source": {"digest": SOURCE, "snapshot_path": str(old_source)}}})
    atomic_json(directory / "evidence.json", {"local_artifact_root": str(old_artifact)})
    atomic_json(directory / "result.json", {"attempt_id": attempt, "status": "INCOMPLETE"})
    readout = {"round": round_id, "scope": "partial_pure_bcap_initial_readout", "unique_paid_attempts": 1,
        "paid_attempts": [{"attempt_id": attempt}], "execution_rounds": [{"round": round_id}],
        "source_cohorts": [{"source_digest": SOURCE, "source_commit": COMMIT, "candidate_ids": []}]}
    _save_readout(root, readout)
    receipt = archiver.archive_readout(root)
    assert receipt["unique_attempts"] == 1 and receipt["files_verified"] == 6
    assert receipt["original_absolute_roots"][str(old_artifact)] == local.relative_to(root).as_posix()
    with tarfile.open(receipt["archive"]) as bundle:
        assert bundle.extractfile(local.relative_to(root).as_posix() + "/stdout.log").read() == b"raw\x00stdout\n"
    with pytest.raises(ValueError, match="already exists"):
        archiver.archive_readout(root)
    (local / "stdout.log").write_bytes(b"corrupted relocation")
    with pytest.raises(ValueError, match="original bytes"):
        archiver.archive_readout(root, archive="runs/archives/changed.tar.gz")


def _pending_fixture(tmp_path, monkeypatch):
    contract = {"evaluation_sha256": "original-evaluation"}
    contract_hash = stable_hash(contract)
    row = {"trainer_family": "bcap", "candidate_id": "saved-bcap", "candidate_revision": "saved-revision",
        "runtime_cohort": {"execution_backend": "cuda"}, "tasks": [{"task_id": "word", "status": "FAIL"}],
        "bindings": {"source_digest": SOURCE, "task_contracts": {"word": contract_hash}},
        "metrics": {"mass_tv": .3}}
    pin = publication.family_row_pin(row, selection_kind="current_measurement", reason="Saved numerical measurement",
                                     measurement_views=["discriminator_stability"])
    atomic_json(tmp_path / CURRENT_SELECTION, {"selections": [pin]})
    manifest = {"view": "discriminator_stability", "view_revision": 5, "policy_fingerprint": "saved-policy",
        "tier_requirements": {"1": ["word"]}, "cohorts": [{"snapshot": "saved.json"}]}
    atomic_json(tmp_path / publication.inventory.EVIDENCE_MANIFEST, manifest)
    result = {**{key: manifest[key] for key in publication.inventory.POLICY_FIELDS},
        "publication_scope": "current_technique_inventory", "rows": [row], "configuration_rows": [row],
        "evidence_rows": [row], "archived_evidence_rows": [], "historical_family_rows": [],
        "task_contracts": {contract_hash: contract},
        "provenance": {"evidence_manifest_sha256": stable_hash(manifest),
            "family_current_selection_sha256": file_hash(tmp_path / CURRENT_SELECTION)}}
    result["provenance"]["input_digest"] = stable_hash(result)
    atomic_json(tmp_path / publication.inventory.CURRENT_PREFIX.with_suffix(".json"), result)
    monkeypatch.setattr(publication.inventory, "_snapshot", lambda *args: ({}, {row["candidate_id"]: row}))
    monkeypatch.setattr(publication.inventory, "_archived_reports", lambda *args: [])
    monkeypatch.setattr("experiments.forge.family_reports.build_progress",
                        lambda root, result: {"word": {"current_contract": "CHANGED", "recorded_status": "FAIL"}})
    monkeypatch.setattr(publication.inventory, "_current_markdown", lambda *args: "One current leaderboard\n")
    monkeypatch.setattr(publication.inventory, "_family_pages", lambda *args: {})
    monkeypatch.setattr(publication.inventory, "_shared_score_intro", lambda *args: None)
    monkeypatch.setattr("experiments.forge.trainer_families.select_family_rows",
                        lambda *args, **kwargs: pytest.fail("Pending refresh must not admit current family pins"))
    monkeypatch.setattr(publication.inventory, "regenerate",
                        lambda *args, **kwargs: pytest.fail("Pending refresh must not regrade scientific evidence"))
    return result


def test_pending_refresh_changes_navigation_and_preserves_all_registered_science(tmp_path, monkeypatch):
    before = _pending_fixture(tmp_path, monkeypatch)
    selected_bytes = (tmp_path / CURRENT_SELECTION).read_bytes()
    receipt = publication.refresh_pending(tmp_path)
    after = read_json(tmp_path / publication.inventory.CURRENT_PREFIX.with_suffix(".json"))
    for key in ["rows", "configuration_rows", "evidence_rows", "archived_evidence_rows",
                "historical_family_rows", "task_contracts"]:
        assert after[key] == before[key]
    assert after["family_progress"]["word"]["current_contract"] == "CHANGED"
    assert receipt["qualification_regraded"] is False and receipt["new_sources_registered"] is False
    assert (tmp_path / CURRENT_SELECTION).read_bytes() == selected_bytes
    assert not (tmp_path / publication.REPORT / "publication.json").exists()
    assert publication.refresh_pending(tmp_path)["input_digest"] == receipt["input_digest"]


def test_pending_refresh_rejects_rebound_selections_before_display_writes(tmp_path, monkeypatch):
    _pending_fixture(tmp_path, monkeypatch)
    current = tmp_path / publication.inventory.CURRENT_PREFIX.with_suffix(".json")
    before = current.read_bytes()
    selection = read_json(tmp_path / CURRENT_SELECTION)
    selection["selections"][0]["reason"] = "Changed after publication"
    atomic_json(tmp_path / CURRENT_SELECTION, selection)
    with pytest.raises(ValueError, match="existing family selections"):
        publication.refresh_pending(tmp_path)
    assert current.read_bytes() == before


def test_pending_refresh_rejects_unregistered_science_before_display_writes(tmp_path, monkeypatch):
    result = _pending_fixture(tmp_path, monkeypatch)
    result["evidence_rows"] = [{**result["rows"][0], "metrics": {"mass_tv": 0}}]
    result["provenance"].pop("input_digest")
    result["provenance"]["input_digest"] = stable_hash(result)
    path = tmp_path / publication.inventory.CURRENT_PREFIX.with_suffix(".json")
    atomic_json(path, result)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unregistered scientific"):
        publication.refresh_pending(tmp_path)
    assert path.read_bytes() == before


def test_pending_display_retains_zero_credit_declarations_but_rejects_a_pass(tmp_path, monkeypatch):
    result = _pending_fixture(tmp_path, monkeypatch)
    declaration = {"candidate_id": "unmeasured-declaration", "attempt_ids": [], "qualified_tier": 0,
        "tasks": [{"task_id": "word", "status": "UNKNOWN"}], "tiers": {"1": {"passed": 0}},
        "cost": {"wall_seconds": None, "measured_tasks": 0}}
    result["configuration_rows"].append(declaration)
    result["provenance"].pop("input_digest")
    result["provenance"]["input_digest"] = stable_hash(result)
    path = tmp_path / publication.inventory.CURRENT_PREFIX.with_suffix(".json")
    atomic_json(path, result)
    publication.refresh_pending(tmp_path)
    after = read_json(path)
    assert after["configuration_rows"][-1] == declaration
    after["configuration_rows"][-1]["tasks"][0]["status"] = "PASS"
    after["provenance"].pop("input_digest")
    after["provenance"]["input_digest"] = stable_hash(after)
    atomic_json(path, after)
    with pytest.raises(ValueError, match="unregistered measured configuration"):
        publication.refresh_pending(tmp_path)


def test_current_generator_links_partial_readout_without_selection(completed, monkeypatch):
    root, _, _, _ = completed
    result = {"scope": "partial_pure_bcap_initial_readout", "qualification_input": False,
        "default_adoption": False, "unique_paid_attempts": 53, "paid_wall_seconds": 1038.752}
    result["input_digest"] = stable_hash(result)
    atomic_json(root / publication.REPORT / "original-initial-readout.json", result)
    (root / publication.REPORT / "publish.py").write_bytes(Path(publication.__file__).read_bytes())
    monkeypatch.setattr("experiments.forge.family_reports.render_leaderboard",
                        lambda *args: "| One current family table |\n")
    text = publication.inventory._current_markdown({"family_progress": {"fixture": True}}, root,
                                                 root / "reports/forge/technique-inventory.md")
    assert text.count("| One current family table |") == 1
    assert "[Pure BCAP initial readout](pure-bcap/README.md)" in text
    assert "53 original paid attempts" in text and "no new selection" in text
    assert not (root / publication.REPORT / "publication.json").exists()


def test_new_measurement_pin_still_rejects_live_contract_drift_before_publication(completed, monkeypatch):
    root, _, _, _, _, _ = _composition_fixture(completed, monkeypatch)
    original = {path: path.read_bytes() for path in publication._publication_paths(root) if path.is_file()}
    def reject(*args, **kwargs):
        assert args[1] == "bcap-pure"
        raise ValueError("current measurement requires current execution, evaluation and budget contracts")
    monkeypatch.setattr("experiments.forge.trainer_families._current_pin", reject)
    with pytest.raises(ValueError, match="current execution, evaluation and budget"):
        publication.publish(root)
    assert {path: path.read_bytes() for path in publication._publication_paths(root) if path.is_file()} == original

"""Register the completed Pure BCAP round in the one current family board.

Only original certified receipts are qualification inputs to the independent
frozen-source regrade. The readout and publication receipt are display metadata.
This script never trains, samples, changes task gates, or adopts a default.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import math
import hashlib
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin, scientific_row_hash
from reports.forge import regenerate_technique_inventory as inventory

REPORT = Path("reports/forge/pure-bcap")
FAMILY = "bcap-pure"
VIEW = "discriminator_stability"


def source_cohorts(readout):
    """Every final candidate belongs to one complete executed-source cohort."""
    cohorts = readout.get("source_cohorts")
    if cohorts is None:
        cohorts = [{"source_commit": readout["source_commit"], "source_digest": readout["source_digest"],
                    "candidate_ids": [candidate["candidate_id"] for candidate in readout["candidates"]]}]
    sources = {}
    for cohort in cohorts:
        for candidate in cohort["candidate_ids"]:
            if candidate in sources:
                raise ValueError("Pure BCAP final candidates cannot pool executed sources")
            sources[candidate] = {key: cohort[key] for key in ("source_commit", "source_digest")}
    if set(sources) != {candidate["candidate_id"] for candidate in readout["candidates"]}:
        raise ValueError("Pure BCAP executed sources differ from the final candidate roster")
    return cohorts, sources


def validate_paid_roster(root, plan, readout, selected_attempts, prior_attempts, sources):
    """Bind every physical charge, including canceled context, to its receipt."""
    records = readout.get("paid_attempts", [])
    ids = [record["attempt_id"] for record in records]
    expected = selected_attempts | set(prior_attempts)
    if (len(ids) != len(set(ids)) or set(ids) != expected
            or len(ids) != readout["unique_paid_attempts"]
            or len(ids) != plan.get("expected_paid_attempt_count", len(expected))
            or len(selected_attempts) != plan.get("selected_measurement_attempt_count", len(selected_attempts))
            or len(prior_attempts) != plan.get("original_unselected_paid_attempt_count", len(prior_attempts))):
        raise ValueError("Pure BCAP paid roster must retain every unique selected and original context attempt")
    originals = read_json(root / REPORT / "plans.json")
    revisions = {trial["candidate_id"]: trial["candidate_revision"]
                 for trial in [*originals["trials"], *plan["trials"]]}
    cohort_pairs = {(cohort["source_digest"], cohort["source_commit"])
                    for cohort in source_cohorts(readout)[0]}
    candidate_costs, total, prior_cost = Counter(), 0.0, 0.0
    for record in records:
        attempt = record["attempt_id"]
        relative = REPORT / "receipts" / (attempt + ".json")
        if (record["receipt"] != relative.as_posix()
                or file_hash(root / relative) != record["receipt_sha256"]):
            raise ValueError("Pure BCAP paid roster receipt reference or hash differs")
        receipt = read_json(root / relative)
        if (plan.get("expected_paid_attempt_count") is not None
                and receipt != inventory.project_receipt(root, attempt)):
            raise ValueError("Pure BCAP paid compact receipt differs from its original certified projection")
        provenance = receipt["provenance"]
        seconds = record["paid_wall_seconds"]
        charged = [task.get("cost", {}).get("execution_seconds", task.get("cost", {}).get("wall_seconds"))
                   for task in receipt.get("task_results", [])]
        if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0
                or not charged or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                                      or value < 0 for value in charged)
                or not math.isclose(seconds, sum(charged), rel_tol=0, abs_tol=1e-8)
                or receipt.get("certificate_validated") is not True or receipt.get("qualification_input") is not False
                or any(record[key] != receipt.get(key) for key in
                       ("attempt_id", "candidate_id", "candidate_revision", "campaign_id", "attempt_status"))
                or record["candidate_revision"] != revisions.get(record["candidate_id"])
                or record["source_digest"] != provenance["source_digest"]
                or record["source_commit"] != provenance["source_origin_commit"]
                or record["canonical_result_hash"] != provenance["canonical_result_hash"]
                or record["task_statuses"] != {task["task_id"]: task["gate_status"] for task in receipt["task_results"]}
                or (record["source_digest"], record["source_commit"]) not in cohort_pairs):
            raise ValueError("Pure BCAP paid roster differs from its certified receipt identity, source, or charge")
        if attempt in selected_attempts:
            source = sources.get(record["candidate_id"])
            if source != {"source_digest": record["source_digest"], "source_commit": record["source_commit"]}:
                raise ValueError("Pure BCAP paid selected attempt belongs to another complete source cohort")
            candidate_costs[record["candidate_id"]] += seconds
        else:
            if record["source_digest"] != originals["source_digest"]:
                raise ValueError("Pure BCAP original paid context belongs to another executed source")
            prior_cost += seconds
        total += seconds
    if (not math.isclose(total, readout["paid_wall_seconds"], rel_tol=0, abs_tol=1e-8)
            or not math.isclose(prior_cost, readout.get("original_unselected_paid_wall_seconds", 0), rel_tol=0, abs_tol=1e-8)
            or any(not math.isclose(candidate_costs[candidate["candidate_id"]],
                                    candidate["cost"]["new_paid_wall_seconds"], rel_tol=0, abs_tol=1e-8)
                   for candidate in readout["candidates"])):
        raise ValueError("Pure BCAP paid roster charge totals differ")
    for cohort in plan.get("execution_rounds", []):
        cohort_records = [record for record in records if record["round"] == cohort["round"]]
        campaign = cohort.get("campaign_id", cohort["round"])
        if (not cohort_records or any(record["campaign_id"] != campaign for record in cohort_records)
                or sum(record["paid_wall_seconds"] for record in cohort_records) > cohort["campaign_cap_seconds"]
                or (cohort.get("paid_attempt_ids_frozen") is not None
                    and ({record["attempt_id"] for record in cohort_records} != set(cohort["paid_attempt_ids_frozen"])
                         or not math.isclose(sum(record["paid_wall_seconds"] for record in cohort_records),
                                             cohort["paid_wall_seconds_frozen"], rel_tol=0, abs_tol=1e-8)))):
            raise ValueError("Pure BCAP paid roster differs from the immutable execution-round accounting")


def completed_readout(root):
    """Verify all ten complete recipes and count shared paid attempts once."""
    root = Path(root).resolve()
    plan_path = root / REPORT / "publication-plans.json"
    if not plan_path.is_file():
        plan_path = root / REPORT / "plans.json"
    plan = read_json(plan_path)
    readout = read_json(root / REPORT / "readout.json")
    cohorts, sources = source_cohorts(readout)
    copied = deepcopy(readout)
    if copied.pop("input_digest", None) != stable_hash(copied):
        raise ValueError("Pure BCAP readout digest differs")
    if (readout.get("scope") != "finite_pure_bcap_initial_readout"
            or readout.get("qualification_input") is not False
            or readout.get("default_adoption") is not False
            or ("source_cohorts" in plan and cohorts != plan["source_cohorts"])
            or ("source_cohorts" not in plan and {c["source_digest"] for c in cohorts} != {plan["source_digest"]})
            or readout.get("round") != plan["round"]
            or readout.get("view_revision") != plan["view_revision"]):
        raise ValueError("Pure BCAP readout differs from the frozen round")
    summaries = [read_json(root / "reports/forge/configuration-search" / (Path(spec).stem + ".json"))
                 for spec in plan["specs"]]
    trials = [trial for summary in summaries for trial in summary["trials"]]
    expected = {trial["candidate_id"]: trial["candidate_revision"] for trial in plan["trials"]}
    observed = {trial["candidate_id"]: trial["candidate_revision"] for trial in trials}
    candidates = {candidate["candidate_id"]: candidate for candidate in readout["candidates"]}
    if (len(trials) != len(observed) or len(candidates) != len(readout["candidates"])
            or observed != expected or set(candidates) != set(expected)
            or len(expected) != plan["configuration_count"]
            or any(trial["source_digest"] != sources[trial["candidate_id"]]["source_digest"] for trial in trials)
            or any(summary["source_digest"] != sources[trial["candidate_id"]]["source_digest"]
                   for summary in summaries for trial in summary["trials"])):
        raise ValueError("Pure BCAP publication requires the entire frozen candidate roster")
    selection = select_configuration(trials, 1)
    if (not selection["all_trials_terminal"] or selection != readout["selection"]
            or selection["default_adoption"] is not False):
        raise ValueError("Pure BCAP whole-recipe selection is incomplete or differs")
    attempts, required, diagnostic = set(), Counter(), Counter()
    for candidate in candidates.values():
        source = sources[candidate["candidate_id"]]
        if candidate["candidate_revision"] != expected[candidate["candidate_id"]]:
            raise ValueError("Pure BCAP readout candidate revision differs")
        tasks = candidate["tasks"]
        if (len(tasks) != len(plan["task_ids"]) or {task["task_id"] for task in tasks} != set(plan["task_ids"])
                or candidate["required_total"] != len(plan["required_task_ids"])):
            raise ValueError("Pure BCAP readout must retain every declared Tier 1 peer")
        for task in tasks:
            if task["status"] not in {"PASS", "FAIL"}:
                raise ValueError("Pure BCAP initial publication requires complete numerical measurements")
            is_required = task["task_id"] in plan["required_task_ids"]
            if task["importance"] != ("required" if is_required else "diagnostic"):
                raise ValueError("Pure BCAP task importance differs from the frozen plan")
            (required if is_required else diagnostic)[task["status"]] += 1
            if task["attempt_id"] in attempts:
                raise ValueError("Pure BCAP attempts and campaign cost must be counted once")
            attempts.add(task["attempt_id"])
            path = root / task["receipt"]
            if file_hash(path) != task["receipt_sha256"]:
                raise ValueError("Pure BCAP compact receipt identity differs")
            receipt = read_json(path)
            if (receipt.get("qualification_input") is not False
                    or receipt.get("certificate_validated") is not True
                    or receipt["attempt_id"] != task["attempt_id"]
                    or receipt["candidate_revision"] != candidate["candidate_revision"]
                    or receipt["provenance"]["source_digest"] != source["source_digest"]
                    or receipt["provenance"]["source_origin_commit"] != source["source_commit"]):
                raise ValueError("Pure BCAP compact receipt provenance differs")
            measured = [row for row in receipt.get("task_results", []) if row["task_id"] == task["task_id"]]
            if (len(measured) != 1 or measured[0]["gate_status"] != task["status"]
                    or any(task.get(key) != measured[0].get(key)
                           for key in ("compatibility_key", "metrics", "cost", "reason"))):
                raise ValueError("Pure BCAP displayed task differs from its sealed compact measurement")
            gif = root / REPORT / "media" / candidate["configuration_id"][:12] / (task["task_id"] + ".gif")
            media = task["gif_receipt"]
            if (file_hash(gif) != media["gif_sha256"]
                    or media["task_id"] != task["task_id"] or media["recorded_grade"] != task["status"]
                    or media.get("kind") != "actual_training_saved_observations_gif"
                    or media.get("optimizer_updates_added") != 0 or media.get("sampling_draws_added") != 0):
                raise ValueError("Pure BCAP actual-training GIF binding differs")
    paid = readout["paid_wall_seconds"]
    prior_attempts = readout.get("original_unselected_paid_attempt_ids", [])
    if (len(prior_attempts) != len(set(prior_attempts)) or attempts & set(prior_attempts)
            or not math.isfinite(readout.get("original_unselected_paid_wall_seconds", 0))
            or readout.get("original_unselected_paid_wall_seconds", 0) < 0):
        raise ValueError("Pure BCAP original repair context must retain disjoint paid attempts")
    if (not isinstance(paid, (int, float)) or not math.isfinite(paid) or paid < 0
            or paid > plan["campaign_cap_seconds"]
            or not math.isclose(paid, sum(candidate["cost"]["new_paid_wall_seconds"]
                                         for candidate in candidates.values())
                               + readout.get("original_unselected_paid_wall_seconds", 0), abs_tol=1e-5)
            or readout["maximum_paid_seconds"] != plan["campaign_cap_seconds"]
            or readout["unique_paid_attempts"] != len(attempts) + len(prior_attempts)
            or dict(required) != readout["required_counts"] or dict(diagnostic) != readout["diagnostic_counts"]):
        raise ValueError("Pure BCAP campaign accounting or numerical counts differ")
    validate_paid_roster(root, plan, readout, attempts, prior_attempts, sources)
    return plan, readout, selection


def selected_pin(frozen, plan, readout, selection, *, selection_kind="current_measurement"):
    """Select one full candidate by the predeclared objective, never by cell."""
    _, sources = source_cohorts(readout)
    source = sources[selection["selected_candidate_id"]]
    matches = [row for row in frozen["rows"] if row["candidate_id"] == selection["selected_candidate_id"]
               and row.get("bindings", {}).get("source_digest") == source["source_digest"]
               and row.get("runtime_cohort", {}).get("execution_backend") == "cuda"]
    if len(matches) != 1:
        raise ValueError("Pure BCAP selected candidate needs one verified CUDA evidence row")
    row = deepcopy(matches[0])
    row["trainer_family"] = FAMILY
    candidate = next(item for item in readout["candidates"] if item["candidate_id"] == row["candidate_id"])
    measured = row["tasks"] + row.get("nonrequired_tasks", [])
    observed = {task["task_id"]: task["status"] for task in measured}
    if (row.get("bindings", {}).get("source_digest") != source["source_digest"]
            or row["candidate_revision"] != candidate["candidate_revision"]
            or any(observed.get(task["task_id"]) != task["status"] for task in candidate["tasks"])
            or set(row["attempt_ids"]) != {task["attempt_id"] for task in candidate["tasks"]}):
        raise ValueError("Pure BCAP selected row differs from its certified whole-recipe readout")
    return family_row_pin(row, selection_kind=selection_kind,
        reason="Initial finite Pure BCAP round: required Tier 1 PASS count descending, then configuration hash ascending. One complete recipe; no calibrated default adoption.",
        measurement_views=[VIEW] if selection_kind == "current_measurement" else None,
        measurement_tasks=(sorted(set(plan["task_ids"]) - set(plan["required_task_ids"]))
                           if selection_kind == "current_measurement" else None))


def preserved_science(before, after):
    """Adding the new family cannot edit any already selected scientific row."""
    expected = {row["trainer_family"]: scientific_row_hash(row) for row in before["rows"]
                if row["trainer_family"] != FAMILY}
    actual = {row["trainer_family"]: scientific_row_hash(row) for row in after["rows"]
              if row["trainer_family"] != FAMILY}
    if expected != actual:
        raise ValueError("Pure BCAP publication changed existing selected family science")
    for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements"):
        if before[key] != after[key]:
            raise ValueError("Pure BCAP publication changed the recorded task policy")
    for key in ("evidence_rows", "archived_evidence_rows", "historical_family_rows"):
        old_hashes = {scientific_row_hash(row) for row in before.get(key, [])}
        if not old_hashes <= {scientific_row_hash(row) for row in after.get(key, [])}:
            raise ValueError("Pure BCAP publication changed existing registered evidence")
    return expected



def _publication_paths(root):
    return {root / CURRENT_SELECTION, root / inventory.EVIDENCE_MANIFEST,
            root / inventory.CURRENT_PREFIX.with_suffix(".json"), root / inventory.CURRENT_PREFIX.with_suffix(".md"),
            root / REPORT / "publication.json", root / REPORT / "current-media-index.json",
            root / "reports/forge/scoped-publications.json", root / "reports/forge/EXPERIMENTS_BY_TIER.md",
            *(root / "reports/forge/families").glob("*.md"),
            *(root / "reports/forge/shared-score-index-20261003").glob("*.md"),
            *(root / "reports/forge/technique-evidence").glob("*.json"),
            *(root / "reports/forge/technique-receipts").glob("*.json")}


def _register_media(root, readout):
    """Extend navigation through a new index; original media stays immutable."""
    registry_path = root / "reports/forge/scoped-publications.json"
    if not registry_path.is_file():
        return
    registry = read_json(registry_path)
    old = registry.get("media")
    if old:
        if file_hash(root / old["path"]) != old["sha256"]:
            raise ValueError("Registered actual-training media index differs")
        index = read_json(root / old["path"])
    else:
        index = {"qualification_input": False, "items": []}
    identities = {(item["family"], item["task_id"], item["attempt_id"]): item for item in index["items"]}
    for candidate in readout["candidates"]:
        for task in candidate["tasks"]:
            gif = REPORT / "media" / candidate["configuration_id"][:12] / (task["task_id"] + ".gif")
            item = {**task["gif_receipt"], "family": FAMILY, "attempt_id": task["attempt_id"], "gif": gif.as_posix()}
            key = FAMILY, task["task_id"], task["attempt_id"]
            if key in identities and identities[key] != item:
                raise ValueError("Pure BCAP actual-training media identity changed")
            if key not in identities:
                index["items"].append(item); identities[key] = item
    path = REPORT / "current-media-index.json"
    atomic_json(root / path, index)
    registry["media"] = {"path": path.as_posix(), "sha256": file_hash(root / path)}
    atomic_json(registry_path, registry)


def compose_scientific_publication(root, before, frozen, plan, readout, selection):
    """Append recorded evidence while retaining every existing scientific row.

    Existing pins are validated against registered snapshots, rather than
    readmitted under a changed live task. Only the new measurement pin goes
    through the strict public fresh-pin contract check.
    """
    from experiments.forge.family_reports import build_progress
    from experiments.forge.trainer_families import REGISTRY, _current_pin, comparison_cohort, load_families
    root = Path(root).resolve()
    manifest = read_json(root / inventory.EVIDENCE_MANIFEST)
    card = read_json(root / CURRENT_SELECTION)
    old_pins = [pin for pin in card["selections"] if pin["trainer_family"] != FAMILY]
    result = deepcopy(before)
    snapshots, eligible = {}, []
    paid = readout.get("paid_attempts", [])
    for cohort, report in frozen:
        if (report.get("publication_scope") != "frozen_source"
                or report.get("frozen_source", {}).get("commit") != cohort["source_commit"]):
            raise ValueError("Pure BCAP composition requires independently reconstructed source snapshots")
        check = deepcopy(report)
        if check["provenance"].pop("input_digest", None) != stable_hash(check):
            raise ValueError("Pure BCAP source snapshot input digest differs")
        if any(report.get(key) != manifest[key] for key in inventory.POLICY_FIELDS):
            raise ValueError("Pure BCAP source snapshot changed the recorded task policy")
        names = {record["candidate_id"] for record in paid if record["source_digest"] == cohort["source_digest"]}
        names = names or set(cohort["candidate_ids"])
        rows = [row for row in report["rows"] if row["candidate_id"] in names and row.get("attempt_ids")
                and row.get("bindings", {}).get("source_digest") == cohort["source_digest"]
                and row.get("runtime_cohort", {}).get("execution_backend") == "cuda"]
        if len(rows) != len(names) or {row["candidate_id"] for row in rows} != names:
            raise ValueError("Pure BCAP snapshot must retain every paid candidate in its complete source cohort")
        recorded = {}
        for row in rows:
            inventory._validate_published_row(root, report, row)
            records = [record for record in paid if record["candidate_id"] == row["candidate_id"]
                       and record["source_digest"] == cohort["source_digest"]]
            if records and ({record["attempt_id"] for record in records} != set(row["attempt_ids"])
                            or any(record["candidate_revision"] != row["candidate_revision"] for record in records)):
                raise ValueError("Pure BCAP snapshot differs from its exact paid receipt roster")
            recorded[row["candidate_id"]] = max(read_json(root / "reports/forge/attempts" / attempt / "result.json")
                                               ["raw"]["finished_at"] for attempt in row["attempt_ids"])
        encoded = inventory._json_text(report)
        relative = inventory.EVIDENCE_MANIFEST.parent / (report["provenance"]["input_digest"] + ".json")
        entry = {"snapshot": relative.as_posix(), "json_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                 "source_commit": cohort["source_commit"], "candidates": recorded}
        if entry not in manifest["cohorts"]:
            manifest["cohorts"].append(entry)
        snapshots[root / relative] = encoded
        result.setdefault("evidence_sources", {})[entry["json_sha256"]] = deepcopy(entry)
        for row in rows:
            copied = deepcopy(row)
            copied.update(publication_key=entry["json_sha256"], qualification_input=False, qualification_reuse=False,
                          trainer_family=FAMILY)
            eligible.append(copied)
        for catalog in inventory.CONTRACT_CATALOGS:
            combined = result.setdefault(catalog, {})
            for digest, contract in report.get(catalog, {}).items():
                if digest != stable_hash(contract) or (digest in combined and combined[digest] != contract):
                    raise ValueError("Pure BCAP composition contains conflicting scientific contracts")
                combined[digest] = deepcopy(contract)
    cohorts, _ = source_cohorts(readout)
    original_source = read_json(root / REPORT / "plans.json").get("source_digest")
    old_winner = len(cohorts) > 1 and selection["source_digest"] == original_source
    pin = selected_pin({"rows": eligible}, plan, readout, selection,
                       selection_kind="historical_incumbent" if old_winner else "current_measurement")
    selected, metadata = _current_pin(root, FAMILY, eligible, pin, view_id=VIEW, catalogs=result)
    metadata.update(qualification_scope="recorded_source", qualifies_latest_checkout=False)
    if old_winner:
        metadata["qualified"] = False
    existing = [item for item in card["selections"] if item["trainer_family"] == FAMILY]
    if existing and existing != [pin]:
        raise ValueError("Pure BCAP publication cannot replace its exact frozen family selection")
    if not existing:
        card["selections"].append(pin)
    label = load_families(root)[FAMILY]["label"]
    winner = deepcopy(selected)
    winner.update(technique=label, selection=metadata,
                  configuration_id=selection["selected_configuration_id"], comparison_cohort=comparison_cohort(selected, result))
    result["rows"] = [row for row in result["rows"] if row.get("trainer_family") != FAMILY] + [winner]
    result.setdefault("trainer_families", {})[FAMILY] = load_families(root)[FAMILY]
    for key in ("configuration_rows", "evidence_rows"):
        seen = {(scientific_row_hash(row), row.get("publication_key")) for row in result.get(key, [])}
        target = result.setdefault(key, [])
        for row in eligible:
            identity = scientific_row_hash(row), row["publication_key"]
            if identity not in seen:
                variant = deepcopy(row)
                if key == "configuration_rows":
                    variant.update(selected_configuration=scientific_row_hash(row) == scientific_row_hash(winner),
                                   alternative_scope="selected" if scientific_row_hash(row) == scientific_row_hash(winner)
                                   else "archived_alternative")
                target.append(variant); seen.add(identity)
    preserved = preserved_science(before, result)
    if [pin for pin in card["selections"] if pin["trainer_family"] != FAMILY] != old_pins:
        raise ValueError("Pure BCAP composition changed an existing selected pin")
    for key in inventory.CONTRACT_CATALOGS:
        if any(result[key].get(digest) != value for digest, value in before.get(key, {}).items()):
            raise ValueError("Pure BCAP composition changed a recorded scientific contract")
    # Store navigation inputs before generating pages. The surrounding
    # transaction restores every public byte if any later validation fails.
    atomic_json(root / CURRENT_SELECTION, card)
    _register_media(root, readout)
    result["publication_refresh"] = {"scientific_rows_preserved": True, "family_selections_preserved": True,
        "qualification_regraded": False, "new_sources_registered": True, "training_launched": False,
        "new_sources_independently_regraded": [cohort["source_commit"] for cohort, _ in frozen],
        "old_word_contracts": "Retained source-bound outcomes; live progress separately reports changed contracts."}
    result["family_progress"] = build_progress(root, result)
    result["common26_display"] = inventory._common26_display_projection(result, root)
    result["provenance"].update(evidence_manifest_sha256=stable_hash(manifest),
        family_current_selection_sha256=file_hash(root / CURRENT_SELECTION),
        trainer_family_registry_sha256=file_hash(root / REGISTRY), selected_rows_sha256=stable_hash(result["rows"]),
        preserved_scientific_composition_reducer_sha256=file_hash(Path(__file__)))
    result["provenance"].pop("input_digest", None)
    result["provenance"]["input_digest"] = stable_hash(result)
    return result, manifest, snapshots, pin, preserved


def publish(root=ROOT):
    root = Path(root).resolve()
    plan, readout, selection = completed_readout(root)
    print(f"Validated {readout['unique_paid_attempts']} paid receipt identities and charges", flush=True)
    before = verified_registered_publication(root)
    print(f"Verified {len(before['rows'])} registered selected scientific rows", flush=True)
    cohorts, _ = source_cohorts(readout)
    originals = {path: path.read_bytes() for path in _publication_paths(root) if path.is_file()}
    try:
        frozen = []
        with tempfile.TemporaryDirectory(prefix="forge-pure-bcap-composition-") as temporary:
            for index, cohort in enumerate(cohorts):
                print(f"Independently regrading executed source {cohort['source_commit']}", flush=True)
                regraded = inventory.regenerate(root, view_id=VIEW, execution_backend="cuda",
                    source_commit=cohort["source_commit"], output_prefix=Path(temporary) / f"evidence-{index}")
                frozen.append((cohort, read_json(regraded["json"])))
                print(f"Verified source {cohort['source_commit']}", flush=True)
        result, manifest, snapshots, pin, preserved = compose_scientific_publication(
            root, before, frozen, plan, readout, selection)
        receipt = {"schema_version": 2, "scope": "pure_bcap_initial_recorded_source_selection",
            "qualification_input": False, "default_adoption": False, "source_cohorts": cohorts,
            "readout": (REPORT / "readout.json").as_posix(), "readout_sha256": file_hash(root / REPORT / "readout.json"),
            "plan_sha256": file_hash(root / REPORT / ("publication-plans.json" if
                                     (root / REPORT / "publication-plans.json").is_file() else "plans.json")),
            "selection": selection, "family_pin": pin, "preserved_family_scientific_rows": preserved,
            "unique_paid_attempts": readout["unique_paid_attempts"], "paid_wall_seconds": readout["paid_wall_seconds"],
            "cost_note": readout["cost_note"], "board": inventory.CURRENT_PREFIX.with_suffix(".md").as_posix(),
            "selection_scope": "One complete executed source cohort; no latest-checkout or default qualification.",
            "selected_live_measurement_contracts_validated": pin["selection_kind"] == "current_measurement",
            "composition": "Exact existing scientific rows and pins are retained; only the new complete Pure BCAP family row is selected."}
        atomic_json(root / REPORT / "publication.json", receipt)
        markdown_path = root / inventory.CURRENT_PREFIX.with_suffix(".md")
        markdown = inventory._current_markdown(result, root, markdown_path)
        pages = inventory._family_pages(root, result)
        shared = inventory._shared_score_intro(root, result)
        for path, content in snapshots.items():
            if path.exists() and path.read_text() != content:
                raise ValueError("Registered source snapshots are immutable")
            inventory._write_changed(path, content)
        atomic_json(root / inventory.EVIDENCE_MANIFEST, manifest)
        inventory._write_changed(root / inventory.CURRENT_PREFIX.with_suffix(".json"), inventory._json_text(result))
        inventory._write_changed(markdown_path, markdown)
        inventory._write_family_pages(root, pages)
        if shared:
            inventory._write_changed(*shared)
    except BaseException:
        for path in _publication_paths(root) - originals.keys():
            if path.is_file():
                path.unlink()
        for path, content in originals.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        raise
    return {"json": str(root / inventory.CURRENT_PREFIX.with_suffix(".json")),
            "report": str(root / inventory.CURRENT_PREFIX.with_suffix(".md")), "rows": len(result["rows"]),
            "input_digest": result["provenance"]["input_digest"], "selected_candidate_id": pin["candidate_id"],
            "preserved_families": len(preserved), "paid_wall_seconds": readout["paid_wall_seconds"]}


def verified_registered_publication(root):
    """Verify existing registered science without admitting current contracts."""
    root = Path(root).resolve()
    manifest = read_json(root / inventory.EVIDENCE_MANIFEST)
    result = read_json(root / inventory.CURRENT_PREFIX.with_suffix(".json"))
    copied = deepcopy(result)
    if copied["provenance"].pop("input_digest", None) != stable_hash(copied):
        raise ValueError("Current publication input digest differs")
    if (result.get("publication_scope") != "current_technique_inventory" or result.get("recorded_policy")
            or any(result.get(key) != manifest[key] for key in inventory.POLICY_FIELDS)
            or result["provenance"]["evidence_manifest_sha256"] != stable_hash(manifest)):
        raise ValueError("Pending display requires the unchanged registered publication policy and evidence")
    selection = root / CURRENT_SELECTION
    if (not result["provenance"].get("family_current_selection_sha256")
            or file_hash(selection) != result["provenance"]["family_current_selection_sha256"]):
        raise ValueError("Pending display cannot change existing family selections")
    for catalog in inventory.CONTRACT_CATALOGS:
        if any(digest != stable_hash(contract) for digest, contract in result.get(catalog, {}).items()):
            raise ValueError("Pending display contains an invalid recorded scientific contract")
    for pin in read_json(selection)["selections"]:
        matches = [row for row in result["rows"] if family_row_pin(row,
            selection_kind=pin["selection_kind"], reason=pin["reason"],
            measurement_views=pin.get("measurement_views"), measurement_tasks=pin.get("measurement_tasks")) == pin]
        if len(matches) != 1:
            raise ValueError("Pending display requires each pin's exact registered whole scientific row")
    registered = set()
    for entry in manifest["cohorts"]:
        _, rows = inventory._snapshot(root, entry, manifest)
        registered.update(scientific_row_hash(row) for row in rows.values())
    for key in ("rows", "evidence_rows"):
        if any(scientific_row_hash(row) not in registered for row in result.get(key, [])):
            raise ValueError("Pending display contains unregistered scientific rows")
    for row in result.get("configuration_rows", []):
        if scientific_row_hash(row) in registered:
            continue
        # Existing inventories include unmeasured declaration projections.
        # Their saved input digest binds them, and they carry no grade credit.
        statuses = {task["status"] for task in row.get("tasks", []) + row.get("nonrequired_tasks", [])}
        if (row.get("attempt_ids") or row.get("qualified_tier", 0) != 0
                or not statuses or not statuses <= {"UNKNOWN", "BLOCKED", "NOT_RUN"}
                or any(tier.get("passed", 0) for tier in row.get("tiers", {}).values())
                or row.get("cost", {}).get("measured_tasks", 0)
                or row.get("cost", {}).get("wall_seconds") not in {None, 0}):
            raise ValueError("Pending display contains an unregistered measured configuration")
    archived = set()
    for _, _, _, rows in inventory._archived_reports(root, manifest):
        archived.update(scientific_row_hash(row) for row in rows.values())
    for original in result.get("archived_evidence_rows", []):
        row = deepcopy(original)
        row.pop("evidence_policy", None)
        if scientific_row_hash(row) not in archived:
            raise ValueError("Pending display contains unregistered archived science")
    if any(scientific_row_hash(row) not in registered | archived for row in result.get("historical_family_rows", [])):
        raise ValueError("Pending display contains unregistered historical science")
    return result


def refresh_pending(root=ROOT):
    """Update navigation over exact saved rows; do not select or register work.

The live word implementation changed to supply both BiGAN gradients. Existing
pins keep their original contracts and grades; the progress display reports
contract drift. Strict fresh-pin admission remains in trainer_families.
"""
    from experiments.forge.family_reports import build_progress
    from experiments.forge.trainer_families import REGISTRY
    root = Path(root).resolve()
    before = verified_registered_publication(root)
    result = deepcopy(before)
    result["family_progress"] = build_progress(root, result)
    word = root / "configs/forge/tasks/five_word_joint_acquisition.json"
    result["publication_refresh"] = {"scientific_rows_preserved": True, "family_selections_preserved": True,
        "qualification_regraded": False, "new_sources_registered": False, "training_launched": False,
        "pending_word_contract": "Repaired joint-loss GPU evidence is pending; original word grades retain their saved contracts.",
        "live_word_task_sha256": file_hash(word) if word.is_file() else None,
        "live_family_registry_sha256": file_hash(root / REGISTRY) if (root / REGISTRY).is_file() else None}
    scientific = (*inventory.CONTRACT_CATALOGS, "rows", "configuration_rows", "evidence_rows",
                  "archived_evidence_rows", "historical_family_rows", "trainer_families")
    if any(result.get(key) != before.get(key) for key in scientific):
        raise ValueError("Pending navigation changed registered science")
    result["provenance"].pop("input_digest", None)
    result["provenance"]["input_digest"] = stable_hash(result)
    markdown = root / inventory.CURRENT_PREFIX.with_suffix(".md")
    text = inventory._current_markdown(result, root, markdown)
    pages = inventory._family_pages(root, result)
    shared = inventory._shared_score_intro(root, result)
    # Validate all generated content before replacing display artifacts.
    inventory._write_changed(root / inventory.CURRENT_PREFIX.with_suffix(".json"), inventory._json_text(result))
    inventory._write_changed(markdown, text)
    inventory._write_family_pages(root, pages)
    if shared:
        inventory._write_changed(*shared)
    return {"json": str(root / inventory.CURRENT_PREFIX.with_suffix(".json")), "report": str(markdown),
            "rows": len(result["rows"]), "input_digest": result["provenance"]["input_digest"],
            **result["publication_refresh"]}


def partial_display_section(root, page):
    """Link audited partial work without ranking or assigning gate credit."""
    root, page = Path(root).resolve(), Path(page)
    path = root / REPORT / "original-initial-readout.json"
    if not path.is_file():
        return ""
    result = read_json(path)
    copied = deepcopy(result)
    if (copied.pop("input_digest", None) != stable_hash(copied)
            or result.get("scope") != "partial_pure_bcap_initial_readout"
            or result.get("qualification_input") is not False or result.get("default_adoption") is not False
            or "selection" in result):
        raise ValueError("Partial Pure BCAP display differs from its non-selecting readout")
    link = os.path.relpath(root / REPORT / "README.md", page.resolve().parent)
    return (f"\n[BCAP initial Adam study]({link}): {result['unique_paid_attempts']} original paid attempts cost "
        f"{result['paid_wall_seconds']:.3f} seconds, counted once. The original two relativistic recipes are complete; "
        "the other loss variants were cancelled after the word host's missing encoder gradient was found. "
        "The corrected study awaits GPU execution. Existing family rows retain their original source and task contracts; "
        "no new selection or default adoption.\n")


def display_section(root, page):
    """A compact readout link, with no second leaderboard or pooled credit."""
    root, page = Path(root).resolve(), Path(page)
    path = root / REPORT / "publication.json"
    if not path.is_file():
        return partial_display_section(root, page)
    receipt = read_json(path)
    readout = read_json(root / receipt["readout"])
    cohorts, _ = source_cohorts(readout)
    receipt_cohorts = receipt.get("source_cohorts") or [{"source_commit": receipt.get("source_commit"),
        "source_digest": receipt.get("source_digest"), "candidate_ids": [c["candidate_id"] for c in readout["candidates"]]}]
    if (receipt.get("scope") not in {"pure_bcap_initial_current_measurement", "pure_bcap_initial_recorded_source_selection"}
            or receipt.get("qualification_input") is not False or receipt.get("default_adoption") is not False
            or file_hash(root / receipt["readout"]) != receipt["readout_sha256"]
            or readout["selection"] != receipt["selection"]
            or cohorts != receipt_cohorts
            or readout["unique_paid_attempts"] != receipt["unique_paid_attempts"]
            or readout["paid_wall_seconds"] != receipt["paid_wall_seconds"]):
        raise ValueError("Pure BCAP publication navigation binding differs")
    link = os.path.relpath(root / REPORT / "README.md", page.resolve().parent)
    selected = receipt["selection"]
    sources_text = ", ".join("`" + cohort["source_commit"] + "`" for cohort in cohorts)
    return (f"\n[BCAP initial Adam study]({link}): five adversarial losses at two constant Adam rates; "
            f"{selected['required_pass_count']}/{selected['required_total']} required Tier 1 passes for one "
            f"selected whole recipe. {receipt['unique_paid_attempts']} unique attempts cost "
            f"{receipt['paid_wall_seconds']:.3f} paid seconds, counted once across the shared campaign. "
            f"Executed source cohorts {sources_text}; each candidate keeps its complete cohort. No default adoption.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pending", action="store_true", help="Refresh existing display navigation without registering repaired evidence")
    args = parser.parse_args()
    print(refresh_pending(args.root) if args.pending else publish(args.root), flush=True)

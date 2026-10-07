"""Prepare exact whole-row family display pins from saved publication metadata.

This helper never resolves a training request, grades samples, or publishes the
inventory. It writes a proposed selection card and an audit to an ignored local
directory; the coordinator reviews/copies the card before --advance-policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import (
    CURRENT_SELECTION, _current_pin, family_for_candidate, family_row_pin,
    load_families, scientific_row_hash,
)
from reports.forge.regenerate_technique_inventory import (
    EVIDENCE_MANIFEST, POLICY_FIELDS, _archived_reports, _published_report,
    _snapshot, _validate_published_row,
)


def _check_digest(report, label):
    copy = deepcopy(report)
    claimed = copy.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(copy):
        raise ValueError(f"{label} input digest mismatch")


def _unique_rows(rows):
    result = {}
    for row in rows:
        result.setdefault(scientific_row_hash(row), row)
    return list(result.values())


def _exact_old_row(pin, rows):
    matches = []
    for original in rows:
        row = {**deepcopy(original), "trainer_family": pin["trainer_family"]}
        if family_row_pin(row, selection_kind=pin["selection_kind"], reason=pin["reason"],
                          measurement_views=pin.get("measurement_views"),
                          measurement_tasks=pin.get("measurement_tasks")) == pin:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"old pin needs one exact verified row: {pin['trainer_family']}")
    return matches[0]


def propose(root, staged, old_board, old_selection, round_definition, manifest,
            archived_rows, declarations, *, configured_standard=False, refresh_source=False):
    """Pure saved-metadata transformation; never selects by observed outcomes."""
    root = Path(root)
    _check_digest(staged, "staged publication")
    _check_digest(old_board, "old inventory")
    view = read_json(root / "configs/forge/views" / (round_definition["view"] + ".json"))
    if (staged.get("publication_scope") != "frozen_source"
            or not staged.get("frozen_source", {}).get("commit")
            or staged.get("policy_fingerprint") != stable_hash(view)
            or staged.get("view_revision") != view["revision"]
            or staged.get("view") != view["id"]
            or round_definition.get("view_revision") != view["revision"]
            or round_definition.get("execution_backend") != "cuda"
            or staged.get("execution_backend") != "cuda"):
        raise ValueError("staged report and frozen round must bind the current CUDA view")
    if (old_selection.get("policy_fingerprint") != manifest["policy_fingerprint"]
            or old_selection.get("view") != manifest["view"]
            or old_selection.get("schema_version") != 1
            or old_selection.get("scope") != "whole_candidate_family_current"
            or old_selection.get("default_adoption") is not False
            or any(old_board.get(key) != manifest[key] for key in POLICY_FIELDS)):
        raise ValueError("old selection/inventory differs from its immutable evidence policy")
    if refresh_source:
        if any(staged.get(key) != manifest[key] for key in POLICY_FIELDS):
            raise ValueError("source refresh requires the identical registered view policy and tier requirements")
        registered = [*manifest.get("cohorts", []),
                      *(entry for policy in manifest.get("archived_policies", []) for entry in policy["cohorts"])]
        old_digests = {row.get("bindings", {}).get("source_digest") for row in [*archived_rows, *old_board.get("rows", [])]}
        new_digests = set(staged["frozen_source"].get("source_digests", []))
        if (not new_digests or new_digests & old_digests
                or staged["frozen_source"]["commit"] in {entry["source_commit"] for entry in registered}):
            raise ValueError("source refresh requires a new executed frozen source identity")
    elif view["revision"] <= manifest["view_revision"]:
        raise ValueError("pin preparation requires a later view revision")
    required = {item["task"] for item in view["assignments"]
                if item["importance"] == "required" and item["qualification_tier"] == 1}
    denominator = [sum(item["importance"] == "required" and item["qualification_tier"] == tier
                       for item in view["assignments"]) for tier in (1, 2, 3)]
    if (not required or denominator != round_definition["required_denominator_by_tier"]
            or set(staged["tier_requirements"]["1"]) != required):
        raise ValueError("frozen round/staging changed the required task denominator")
    roster = round_definition["candidate_ids"]
    preselected = round_definition["configuration_ids"]
    if len(roster) != len(set(roster)) or len(preselected) != len(set(preselected)) or not set(preselected) <= set(roster):
        raise ValueError("round requires distinct exact candidate and preselected IDs")
    for kind, field in (("idea", "idea_card_hashes"), ("configuration", "configuration_card_hashes")):
        for name, digest in round_definition.get(field, {}).items():
            if name not in declarations or stable_hash(declarations[name]) != digest:
                raise ValueError(f"frozen {kind} declaration changed: {name}")
    families = load_families(root)
    by_family = {}
    for name in roster:
        family = family_for_candidate(root, name, declarations.get(name), current_presentation=True)
        by_family.setdefault(family["id"], {"family": family, "candidates": []})["candidates"].append(name)
    preferred = {}
    for name in preselected:
        family = family_for_candidate(root, name, declarations.get(name), current_presentation=True)["id"]
        if family in preferred:
            raise ValueError(f"round preselects multiple recipes for family {family}")
        preferred[family] = name
    for pin in old_selection["selections"]:
        name = pin["candidate_id"]
        if name in roster:
            family = family_for_candidate(root, name, declarations.get(name), current_presentation=True)["id"]
            preferred.setdefault(family, name)

    rows = []
    for original in staged["rows"]:
        row = deepcopy(original)
        row["trainer_family"] = family_for_candidate(
            root, row["candidate_id"], declarations.get(row["candidate_id"]), current_presentation=True)["id"]
        rows.append(row)
    rows = _unique_rows(rows)
    source_digests = set(staged["frozen_source"].get("source_digests", []))
    measured = [row for row in rows if row.get("attempt_ids")]
    if not measured:
        raise ValueError("staged source has no measured rows; do not publish a preflight source as executed")
    for row in rows:
        if row.get("runtime_cohort", {}).get("execution_backend") != "cuda":
            raise ValueError("staged row uses a different backend")
        if row.get("bindings", {}).get("source_digest") not in source_digests:
            raise ValueError("staged row is absent from the validated frozen source manifests")
        if (row.get("attempt_ids") and staged["frozen_source"]["commit"] not in
                row.get("bindings", {}).get("recorded_source_origin_commits", [])):
            raise ValueError("measured row has no receipt from the declared executed source commit")
        observed = {task["task_id"]: task["status"] for task in row.get("tasks", [])}
        if len(observed) != len(row.get("tasks", [])):
            raise ValueError(f"duplicate task cells: {row['candidate_id']}")
        for tier, names in staged["tier_requirements"].items():
            cell = row.get("tiers", {}).get(tier, {})
            if (not set(names) <= set(observed) or cell.get("total") != len(names)
                    or cell.get("passed") != sum(observed[name] == "PASS" for name in names)):
                raise ValueError("staged row's full task cells disagree with its tier summary")
    for digest, contract in staged.get("task_contracts", {}).items():
        if digest != stable_hash(contract):
            raise ValueError("staged task contract digest mismatch")
    selections, decisions, unavailable = [], [], []
    for family_id, group in sorted(by_family.items()):
        family = group["family"]
        name = preferred.get(family_id, family["canonical_candidate"])
        matches = [row for row in rows if row["candidate_id"] == name]
        if len(matches) > 1:
            raise ValueError(f"multiple exact-source rows for intended candidate {name}; choose no runtime silently")
        if not matches:
            unresolved = [row for row in staged.get("unresolved_configuration_rows", [])
                          if row["candidate_id"] == name]
            unavailable.append({"trainer_family": family_id, "candidate_id": name,
                                "inventory_visible": family.get("inventory_visible", True),
                                "canonical_candidate": family["canonical_candidate"],
                                "active_search_by_backend": deepcopy(family.get("active_search_by_backend", {})),
                                "reason": "No registrable frozen row for the pre-run choice; retain a declaration-only fallback, no substitute recipe.",
                                "unresolved_blockers": [reason for row in unresolved for reason in row.get("blockers", [])],
                                "available_other_candidate_ids": sorted({row["candidate_id"] for row in rows
                                                                         if row["trainer_family"] == family_id})})
            continue
        row = matches[0]
        statuses = {task["task_id"]: task["status"] for task in row.get("tasks", [])}
        if len(statuses) != len(row.get("tasks", [])):
            raise ValueError(f"duplicate task cells: {name}")
        complete = bool(row.get("attempt_ids")) and all(statuses.get(task) in {"PASS", "FAIL"} for task in required)
        all_pass = complete and all(statuses[task] == "PASS" for task in required)
        kind = "configured_standard" if all_pass and configured_standard else "current_measurement" if complete else "historical_incumbent"
        reason = ("Preserve the round's pre-run whole candidate choice in its freshly measured CUDA source cohort; "
                  "no task pooling, outcome-based recipe reselection, calibration or default adoption."
                  if complete else
                  "Current-source pre-run whole candidate choice is partially BLOCKED or unmeasured; "
                  "retain exact display evidence with no complete current-measurement, qualification or default-adoption credit.")
        pin = family_row_pin(row, selection_kind=kind, reason=reason,
                             measurement_views=[view["id"]] if kind == "current_measurement" else None)
        selected, metadata = _current_pin(root, family_id, rows, pin, view_id=view["id"], catalogs=staged)
        if scientific_row_hash(selected) != scientific_row_hash(row):
            raise ValueError("selected pin changed its scientific row")
        selections.append(pin)
        decisions.append({"trainer_family": family_id, "candidate_id": name, "selection_kind": kind,
                          "scientific_row_sha256": scientific_row_hash(row),
                          "required_tier1_statuses": {task: statuses.get(task, "UNKNOWN") for task in sorted(required)},
                          "qualification_recorded": row.get("qualified_tier"),
                          "complete_current_measurement": complete, "inventory_visible": family.get("inventory_visible", True),
                          "attempt_ids": row.get("attempt_ids", []), "default_adoption": metadata["default_adoption"]})

    history, archived = {}, []
    for pin in [*old_selection.get("historical_selections", []), *old_selection["selections"]]:
        # Match registered, receipt-verified snapshots rather than trusting a
        # display board's self-reported hash. Do not hide duplicate matches:
        # the public historical selector also requires exactly one row.
        row = _exact_old_row(pin, archived_rows)
        converted = pin["selection_kind"] != "historical_incumbent" or "measurement_views" in pin or "measurement_tasks" in pin
        reason = (f"Preserve the exact archived revision-{manifest['view_revision']} measurement and its original policy qualification; "
                  "no current-policy measurement or default-adoption credit.") if converted else pin["reason"]
        historical = family_row_pin(row, selection_kind="historical_incumbent", reason=reason)
        if historical["scientific_row_sha256"] != pin["scientific_row_sha256"]:
            raise ValueError("archival conversion changed scientific evidence")
        # Mirror the public historical matcher, which deliberately excludes measurement fields.
        if family_row_pin(row, selection_kind=historical["selection_kind"], reason=historical["reason"]) != historical:
            raise ValueError("historical display pin cannot match its exact row")
        key = historical["trainer_family"], historical["scientific_row_sha256"]
        history.setdefault(key, historical)
        archived.append({"trainer_family": pin["trainer_family"], "candidate_id": pin["candidate_id"],
                         "scientific_row_sha256": pin["scientific_row_sha256"],
                         "old_selection_kind": pin["selection_kind"], "converted_display_metadata": converted,
                         "original_qualified_tier": row.get("qualified_tier")})
    card = {"schema_version": 1, "scope": "whole_candidate_family_current", "default_adoption": False,
            "view": view["id"], "policy_fingerprint": stable_hash(view),
            "selections": selections, "historical_selections": [history[key] for key in sorted(history)]}
    audit = {"schema_version": 1, "scope": "saved_metadata_only_family_pin_proposal",
             "preparation_mode": "same_view_source_refresh" if refresh_source else "later_view_policy",
             "frozen_source": deepcopy(staged["frozen_source"]), "view_revision": view["revision"],
             "required_denominator_by_tier": denominator, "selection_rule": round_definition.get("selection_rule"),
             "registered_family_count": len(families), "round_family_count": len(by_family),
             "selected_family_count": len(selections), "selection_kind_counts": dict(Counter(pin["selection_kind"] for pin in selections)),
             "unavailable_selections": unavailable, "decisions": decisions, "archived_pins": archived,
             "original_policy": {key: deepcopy(manifest[key]) for key in POLICY_FIELDS},
             "evidence_manifest_unchanged_sha256": stable_hash(manifest),
             "qualification_input": False, "qualification_reuse": False, "default_adoption": False,
             "publication_performed": False, "selection_card_sha256": stable_hash(card)}
    return card, audit


def _validate_refresh_sources(root, staged):
    """Check saved source bytes and original request origins without regrading."""
    from experiments.forge.sources import verify_snapshot
    expected = staged["frozen_source"]
    verified = {}
    for row in staged["rows"]:
        for attempt in row.get("attempt_ids", []):
            request = read_json(root / "reports/forge/attempts" / attempt / "request.json")["request"]
            source = request["source"]
            proof = read_json(root / "reports/forge/technique-receipts" / (attempt + ".json"))["provenance"]
            if (source["origin_commit"] != expected["commit"]
                    or source["digest"] not in expected["source_digests"]
                    or source["digest"] != row["bindings"]["source_digest"]
                    or proof.get("source_origin_commit") != source["origin_commit"]
                    or proof.get("source_digest") != source["digest"]
                    or request["candidate"]["id"] != row["candidate_id"]
                    or request["candidate_revision"] != row["candidate_revision"]):
                raise ValueError("source refresh original request differs from its exact staged receipt/source identity")
            snapshot = Path(source["snapshot_path"])
            snapshot = snapshot if snapshot.is_absolute() else root / snapshot
            manifest_path = snapshot / "forge-source.json"
            frozen = read_json(manifest_path)
            if frozen != {key: value for key, value in source.items() if key != "snapshot_path"}:
                raise ValueError("source refresh snapshot manifest differs from its original request")
            key = source["digest"], str(snapshot.resolve())
            if key not in verified:
                verify_snapshot(snapshot, source)
                verified[key] = {"source_origin_commit": source["origin_commit"],
                                 "source_digest": source["digest"], "snapshot": str(snapshot.resolve()),
                                 "manifest_sha256": file_hash(manifest_path), "source_bytes_verified": True}
    if not verified:
        raise ValueError("source refresh has no measured original source to verify")
    if {key[0] for key in verified} != set(expected["source_digests"]):
        raise ValueError("source refresh claims a frozen manifest without a measured original source")
    return [verified[key] for key in sorted(verified)]


def prepare(root, *, staged_path, round_path, old_board_path, old_selection_path,
            manifest_path, configured_standard=False, refresh_source=False):
    """Validate saved digests/proofs, then prepare a proposal without publishing."""
    root = Path(root).resolve()
    paths = {"staged": Path(staged_path), "round": Path(round_path), "old_board": Path(old_board_path),
             "old_selection": Path(old_selection_path), "evidence_manifest": Path(manifest_path)}
    paths = {key: path if path.is_absolute() else root / path for key, path in paths.items()}
    staged, _ = _published_report(root, paths["staged"])
    origins = {}
    for row in staged["rows"]:
        _validate_published_row(root, staged, row)
        for attempt in row.get("attempt_ids", []):
            summary = read_json(root / "reports/forge/technique-receipts" / (attempt + ".json"))
            origin = summary["provenance"].get("source_origin_commit")
            if origin is not None:
                origins.setdefault(summary["provenance"]["source_digest"], set()).add(origin)
    for row in staged["rows"]:
        # Match regenerate's manifest-wide origin binding; compatible reuse
        # still preserves every actual attempt and its original receipt.
        recorded = sorted(origins.get(row.get("bindings", {}).get("source_digest"), set()))
        if row.get("attempt_ids") and recorded != row.get("bindings", {}).get("recorded_source_origin_commits"):
            raise ValueError("staged row origin metadata differs from its saved receipt proofs")
    manifest = read_json(paths["evidence_manifest"])
    archived_rows = []
    for entry in manifest["cohorts"]:
        _, selected = _snapshot(root, entry, manifest)
        archived_rows.extend(selected.values())
    for _, _, _, selected in _archived_reports(root, manifest):
        archived_rows.extend(selected.values())
    from experiments.forge.planning import declaration_paths
    declarations = {path.stem: read_json(path) for path in declaration_paths(root)}
    card, audit = propose(root, staged, read_json(paths["old_board"]), read_json(paths["old_selection"]),
                          read_json(paths["round"]), manifest, archived_rows, declarations,
                          configured_standard=configured_standard, refresh_source=refresh_source)
    if refresh_source:
        audit["verified_refresh_sources"] = _validate_refresh_sources(root, staged)
    audit["input_files"] = {key: {"path": str(path), "sha256": file_hash(path)} for key, path in paths.items()}
    audit["helper_sha256"] = file_hash(Path(__file__))
    return card, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--round", dest="round_path", type=Path, required=True)
    parser.add_argument("--old-board", type=Path, default=Path("reports/forge/technique-inventory.json"))
    parser.add_argument("--old-selection", type=Path, default=CURRENT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=EVIDENCE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True, help="ignored local directory, or a directory outside the repository")
    parser.add_argument("--configured-standard", action="store_true", help="label a complete 6/6 row configured_standard; default is current_measurement")
    parser.add_argument("--refresh-source", action="store_true",
                        help="explicit new-source refresh under the identical registered view; preserve pre-run recipe choices")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    targets = [output / "selection.json", output / "audit.json"]
    for target in targets:
        if target.is_relative_to(root) and subprocess.run(
                ["git", "check-ignore", "--quiet", "--", str(target)], cwd=root).returncode != 0:
            parser.error("output must be ignored or outside the repository; the helper never writes publication inputs")
    card, audit = prepare(root, staged_path=args.staged, round_path=args.round_path,
                          old_board_path=args.old_board, old_selection_path=args.old_selection,
                          manifest_path=args.manifest, configured_standard=args.configured_standard,
                          refresh_source=args.refresh_source)
    atomic_json(targets[0], card)
    atomic_json(targets[1], audit)
    print(json.dumps({"selection": str(targets[0]), "audit": str(targets[1]),
                      "selected_families": audit["selected_family_count"],
                      "unavailable": audit["unavailable_selections"], "publication_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()

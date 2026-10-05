"""Freeze repair alternatives and register navigation; never train or select a family.

The numerical snapshot is a verification artifact, not another leaderboard or
qualification input. It keeps whole recipes and their recorded source cohorts
separate from the incumbent's older qualification policy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash

DIRECTORY = Path("reports/forge/bcap-tier1-repair")
STUDY = "bcap-tier1-repair-rates-v1"
STUDIES = {STUDY: ("plans.json", 8), "bcap-tier1-repair-moments-v1": ("moment-plans.json", 4)}
DIAGNOSTIC = "bcap-original-horizon-diagnostics-v1"
SNAPSHOT = DIRECTORY / "rates-evidence.json"
REGISTRY = DIRECTORY / "publication-links.json"
TERMINAL = {"completed", "blocked", "cancelled", "concluded"}


def _publisher(root):
    path = root / "reports/forge/regenerate_technique_inventory.py"
    spec = importlib.util.spec_from_file_location("repair_inventory_publisher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("publication requires an existing repository file: " + str(relative))
    return path


def _stopped_campaign(root, plans):
    """Read admitted requests, including draining jobs behind cancelled submissions."""
    from experiments.forge.queue import Queue
    queue = Queue(root / "runs/forge/bcap-tier1-repair/queue")
    state = queue.inspect()
    expected = {trial["candidate"]: trial for trial in plans["trials"]}
    entries = [entry for entry in state["submissions"].values()
               if entry["request"].get("campaign_id") == plans["study"]]
    total = STUDIES[plans["study"]][1]
    if (len(expected) != total or len(entries) != total or
            {entry["request"]["candidate"]["id"] for entry in entries} != set(expected)):
        raise ValueError("publication must retain every declared repair configuration")
    campaign = state["campaigns"].get(plans["study"], {})
    if campaign.get("reserved_seconds") != 0 or any(entry["status"] not in TERMINAL for entry in entries):
        raise ValueError("finish or cancel repair submissions before publication")
    requests = {}
    for entry in entries:
        request = entry["request"]
        candidate = request["candidate"]["id"]
        if request["source"]["digest"] != plans["source_digest"]:
            raise ValueError("admitted repair source differs from its reviewed plan")
        names = {a["task"] for a in request["view"]["assignments"]
                 if a["qualification_tier"] == 1 and a["importance"] == "required"}
        if names != set(expected[candidate].get("tasks", names)) or len(names) != 7:
            raise ValueError("repair publication must preserve all seven required tasks")
        for job in request["jobs"]:
            if Queue._authorized(request, job) and state["jobs"][job["compatibility_key"]]["status"] == "running":
                raise ValueError("repair worker is still draining after submission termination")
        requests[candidate] = request
    return requests


def _description(report):
    required = set(report["tier_requirements"]["1"])
    studies = report["repair_studies"]
    if STUDY not in studies or set(studies) - STUDIES.keys() or any(
            len(definition["candidates"]) != STUDIES[study][1] for study, definition in studies.items()):
        raise ValueError("repair snapshot changed its declared study membership")
    expected = {candidate for study in studies.values() for candidate in study["candidates"]}
    if (len(required) != 7 or len(report["rows"]) != len(expected) or
            {row["candidate_id"] for row in report["rows"]} != expected):
        raise ValueError("repair snapshot must retain every recipe and the seven-task denominator")
    descriptions = []
    for row in report["rows"]:
        tasks = {task["task_id"]: task["status"] for task in row["tasks"] if task["task_id"] in required}
        if set(tasks) != required or row["tiers"]["1"]["total"] != len(required):
            raise ValueError("repair row changed its required denominator")
        passed = sum(status == "PASS" for status in tasks.values())
        if row["tiers"]["1"]["passed"] != passed:
            raise ValueError("repair row pass count differs from its task statuses")
        descriptions.append({"candidate_id": row["candidate_id"],
            "candidate_revision": row["candidate_revision"],
            "configuration_id": row["bindings"].get("configuration_id") or row["candidate_id"],
            "required_passes": passed, "required_total": len(required), "task_statuses": tasks})
    # This is the declared Tier 1 search objective and configuration-hash tie
    # break. A partial best observation is never a selected family replacement.
    best = {study: min((item for item in descriptions if item["candidate_id"] in definition["candidates"]),
                      key=lambda item: (-item["required_passes"], item["configuration_id"]))
            for study, definition in sorted(studies.items())}
    complete = sorted(item["candidate_id"] for item in descriptions if item["required_passes"] == len(required))
    return {"best_observed_by_study": best, "configurations": sorted(descriptions, key=lambda item: item["candidate_id"]),
            "all_required_pass_candidates": complete, "family_selection_changed": False,
            "default_adoption": False, "independent_confirmation": "not_performed"}


def freeze(root, source_commits=None, expected_studies=None):
    root = Path(root).resolve()
    publisher = _publisher(root)
    results = read_json(root / DIRECTORY / "results.json")
    if results.get("status") != "COMPLETE":
        raise ValueError("freeze the repair evidence only after the full readout is COMPLETE")
    admitted_studies = {row["campaign"] for row in results["candidates"] if row["campaign"] in STUDIES}
    from experiments.forge.queue import Queue
    actual = Queue(root / "runs/forge/bcap-tier1-repair/queue").inspect()
    queue_studies = {entry["request"]["campaign_id"] for entry in actual["submissions"].values()
                     if entry["request"]["campaign_id"] in STUDIES}
    if queue_studies != admitted_studies or (expected_studies is not None and set(expected_studies) != admitted_studies):
        raise ValueError("COMPLETE readout does not retain the complete expected admitted study set")
    if STUDY not in admitted_studies:
        raise ValueError("the original rate study must remain in the final readout")
    requests, studies = {}, {}
    for study in sorted(admitted_studies):
        plans = read_json(root / DIRECTORY / STUDIES[study][0])
        if plans["study"] != study or file_hash(_path(root, plans["spec"])) != plans["spec_sha256"]:
            raise ValueError("repair search declaration differs from its reviewed plan")
        for trial in plans["trials"]:
            if file_hash(_path(root, trial["declaration"])) != trial["declaration_sha256"]:
                raise ValueError("repair configuration declaration differs from its reviewed plan")
        admitted = _stopped_campaign(root, plans)
        requests.update(admitted)
        studies[study] = {"candidates": sorted(admitted), "plan": str(DIRECTORY / STUDIES[study][0]),
                          "plan_sha256": file_hash(root / DIRECTORY / STUDIES[study][0])}
    commits = sorted({request["source"]["origin_commit"] for request in requests.values()})
    if source_commits is not None:
        if isinstance(source_commits, str):
            source_commits = [source_commits]
        if {publisher._resolve_commit(root, commit) for commit in source_commits} != set(commits):
            raise ValueError("explicit publication sources differ from the complete admitted repair cohorts")
    report, cohorts = None, []
    for commit in commits:
        with tempfile.TemporaryDirectory(prefix="forge-bcap-repair-publication-") as temporary:
            _, measured, reconstructed, digests = publisher._frozen_report(root, commit,
                view_id="discriminator_stability", execution_backend="cuda", temporary=Path(temporary))
        cohort_requests = {candidate: request for candidate, request in requests.items()
                           if request["source"]["origin_commit"] == commit}
        required_digests = {request["source"]["digest"] for request in cohort_requests.values()}
        if reconstructed != commit or not required_digests.issubset(digests):
            raise ValueError("frozen repair report differs from the admitted source")
        from experiments.forge.views import view_fingerprint
        if any(view_fingerprint(request["view"]) != measured["policy_fingerprint"]
               for request in cohort_requests.values()):
            raise ValueError("frozen repair report differs from its admitted policy")
        measured["rows"] = [row for row in measured["rows"] if row["candidate_id"] in cohort_requests]
        if report is None:
            report = measured
        else:
            if any(report[field] != measured[field] for field in ("view", "view_revision", "policy_fingerprint", "tier_requirements")):
                raise ValueError("repair studies must retain the same declared view and required criteria")
            report["rows"].extend(measured["rows"])
            for field in ("task_contracts", "recipe_contracts", "protocol_contracts", "status_reasons"):
                report.setdefault(field, {}).update(measured.get(field, {}))
        cohorts.append({"commit": commit, "source_digests": sorted(required_digests)})
    if len(report["rows"]) != len(requests) or {row["candidate_id"] for row in report["rows"]} != set(requests):
        raise ValueError("independent regrade did not reconstruct every repair configuration")
    for row in report["rows"]:
        request = requests[row["candidate_id"]]
        if (row["candidate_revision"] != request["candidate_revision"] or
                row["bindings"]["source_digest"] != request["source"]["digest"]):
            raise ValueError("independent repair row differs from its admitted cohort")
    for field in ("configuration_rows", "archived_rows", "unresolved_configuration_rows"):
        report.pop(field, None)
    attempts = sorted({attempt for row in report["rows"] for attempt in row["attempt_ids"]})
    diagnostics = [row for row in results["candidates"] if row["campaign"] == DIAGNOSTIC]
    if len(diagnostics) != 1 or len(diagnostics[0]["tasks"]) != 2:
        raise ValueError("retain the separate two-task duration diagnostic cohort")
    attempts = sorted(set(attempts) | {task["attempt_id"] for row in diagnostics for task in row["tasks"]})
    summaries = {attempt: publisher.project_receipt(root, attempt) for attempt in attempts}
    metric_proofs = {}
    for row in results["candidates"]:
        for task in row["tasks"]:
            attempt = task["attempt_id"]
            original = read_json(root / "reports/forge/attempts" / attempt / "result.json")
            measured = [item for item in original["task_results"] if item["task_id"] == task["task_id"]]
            if (len(measured) != 1 or stable_hash(original) != summaries[attempt]["provenance"]["canonical_result_hash"] or
                    measured[0].get("metrics", {}) != task["metrics"]):
                raise ValueError("final numerical readout differs from its exact certified original metrics")
            # Compact receipts deliberately omit arbitrary arrays. Hash the
            # full final metric mapping once while originals are hydrated;
            # later display verification does not need the raw execution data.
            metric_proofs[attempt + "/" + task["task_id"]] = stable_hash(task["metrics"])
    report.update(publication_scope="frozen_source", qualification_input=False, qualification_reuse=False,
        publication_role="bcap_repair_alternative_verification", repair_studies=studies,
        duration_diagnostics={"qualification_input": False, "cohorts": diagnostics},
        readout_metric_sha256=metric_proofs,
        frozen_source={"cohorts": cohorts, "qualifies_latest_checkout": False})
    report["repair_readout"] = _description(report)
    publisher._publication_provenance(report, summaries)
    # Validate staged proofs before touching shared compact receipt files.
    for summary in summaries.values():
        if summary["campaign_id"] == DIAGNOSTIC:
            continue
        if summary["provenance"]["source_digest"] != requests[summary["candidate_id"]]["source"]["digest"]:
            raise ValueError("repair projection crossed source cohorts")
    for attempt, summary in summaries.items():
        publisher._write_changed(root / "reports/forge/technique-receipts" / (attempt + ".json"),
                                 publisher._json_text(summary))
    for row in report["rows"]:
        publisher._validate_published_row(root, report, row)
    _verified_readout(root, report)
    publisher._write_changed(root / SNAPSHOT, publisher._json_text(report))
    return {"snapshot": str(SNAPSHOT), "sha256": file_hash(root / SNAPSHOT),
            "input_digest": report["provenance"]["input_digest"], **report["repair_readout"]}


def _verified_snapshot(root):
    report = read_json(root / SNAPSHOT)
    copied = deepcopy(report)
    claimed = copied.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(copied):
        raise ValueError("repair numerical snapshot input digest mismatch")
    if (report.get("publication_scope") != "frozen_source" or
            report.get("publication_role") != "bcap_repair_alternative_verification" or
            report.get("qualification_input") is not False or report.get("qualification_reuse") is not False):
        raise ValueError("repair snapshot must remain a display-only verification artifact")
    if report.get("repair_readout") != _description(report):
        raise ValueError("repair descriptive result differs from its numerical rows")
    publisher = _publisher(root)
    for row in report["rows"]:
        publisher._validate_published_row(root, report, row)
    return report


def _verified_readout(root, report):
    publisher = _publisher(root)
    results = read_json(root / DIRECTORY / "results.json")
    if results.get("qualification_input") is not False or results.get("status") != "COMPLETE":
        raise ValueError("repair readout must be COMPLETE and display-only")
    if any(c["reserved_seconds"] for c in results["campaign_accounting"].values()):
        raise ValueError("repair readout still has reserved work")
    if any(row["submission_status"] not in TERMINAL for row in results["candidates"]):
        raise ValueError("repair readout still has active submissions")
    from experiments.forge.queue import Queue
    actual = Queue(root / "runs/forge/bcap-tier1-repair/queue").inspect()
    actual_studies = {entry["request"]["campaign_id"] for entry in actual["submissions"].values()
                      if entry["request"]["campaign_id"] in STUDIES}
    if actual_studies != set(report["repair_studies"]):
        raise ValueError("repair publication omits an admitted study")
    if (any(entry["status"] not in TERMINAL for entry in actual["submissions"].values()) or
            any(job["status"] == "running" for job in actual["jobs"].values()) or
            any(campaign["reserved_seconds"] for campaign in actual["campaigns"].values())):
        raise ValueError("repair queue still has active or draining work")
    if {row["campaign"] for row in results["candidates"]} != set(report["repair_studies"]) | {DIAGNOSTIC}:
        raise ValueError("repair readout changed its admitted study membership")
    rates = [row for row in results["candidates"] if row["campaign"] in STUDIES]
    expected = {row["candidate_id"]: row for row in report["rows"]}
    if len(rates) != len(expected) or {row["candidate"] for row in rates} != set(expected):
        raise ValueError("final repair readout omits a declared configuration")
    for row in rates:
        original = expected[row["candidate"]]
        statuses = {task["task_id"]: task["status"] for task in row["tasks"]}
        scored = {task["task_id"]: task["status"] for task in original["tasks"]
                  if task["task_id"] in report["tier_requirements"]["1"]}
        if (row["candidate_revision"] != original["candidate_revision"] or
                row["executed_commit"] != original["bindings"]["source_origin_commit"] or
                row["source_digest"] != original["bindings"]["source_digest"] or statuses != scored):
            raise ValueError("final repair readout differs from independent frozen regrade")
    diagnostic = report.get("duration_diagnostics", {})
    rows = [row for row in results["candidates"] if row["campaign"] == DIAGNOSTIC]
    if diagnostic.get("qualification_input") is not False or rows != diagnostic.get("cohorts"):
        raise ValueError("duration diagnostic cohort changed or entered ordinary qualification")
    for row in results["candidates"]:
        for task in row["tasks"]:
            summary = read_json(root / "reports/forge/technique-receipts" / (task["attempt_id"] + ".json"))
            provenance = summary["provenance"]
            certified = [item for item in summary["task_results"] if item["task_id"] == task["task_id"]]
            proof = report["provenance"]["qualified_receipts"].get(task["attempt_id"])
            actual = {"canonical_result_hash": provenance["canonical_result_hash"],
                      "source_digest": provenance["source_digest"],
                      "original_file_sha256": {name: item["sha256"] for name, item in provenance["original_files"].items()}}
            if (len(certified) != 1 or summary.get("certificate_validated") is not True or actual != proof or
                    summary.get("qualification_input") is not False or summary.get("qualification_reuse") is not False or
                    summary["candidate_id"] != row["candidate"] or summary["candidate_revision"] != row["candidate_revision"] or
                    summary["campaign_id"] != row["campaign"] or provenance["source_digest"] != row["source_digest"] or
                    provenance["source_origin_commit"] != row["executed_commit"] or
                    provenance["canonical_result_hash"] != task["canonical_result_sha256"] or
                    certified[0]["gate_status"] != task["status"] or
                    certified[0]["metrics"] != publisher._scalars(task["metrics"]) or
                    report["readout_metric_sha256"].get(task["attempt_id"] + "/" + task["task_id"]) != stable_hash(task["metrics"])):
                raise ValueError("repair readout differs from its certified numerical receipt")
    return results


def register(root, readout=DIRECTORY / "README.md"):
    root = Path(root).resolve()
    report = _verified_snapshot(root)
    _verified_readout(root, report)
    readout_path = _path(root, readout)
    paths = {"evidence": root / SNAPSHOT, "results": root / DIRECTORY / "results.json", "readout": readout_path}
    registry = {"schema_version": 1, "scope": "bcap_repair_display_navigation", "qualification_input": False,
                "files": {name: {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
                          for name, path in paths.items()}}
    atomic_json(root / REGISTRY, registry)
    return registry


def display_section(root, page):
    """Optional native-generator section; absent registration renders nothing."""
    root, page = Path(root).resolve(), Path(page).resolve()
    path = root / REGISTRY
    if not path.is_file():
        return ""
    registry = read_json(path)
    if (registry.get("schema_version") != 1 or registry.get("scope") != "bcap_repair_display_navigation" or
            registry.get("qualification_input") is not False or set(registry.get("files", {})) != {"evidence", "results", "readout"}):
        raise ValueError("invalid repair publication navigation")
    for name, pointer in registry["files"].items():
        if file_hash(_path(root, pointer["path"])) != pointer["sha256"]:
            raise ValueError("repair publication navigation hash mismatch: " + name)
    if (registry["files"]["evidence"]["path"] != str(SNAPSHOT) or
            registry["files"]["results"]["path"] != str(DIRECTORY / "results.json")):
        raise ValueError("repair navigation must bind the registered numerical snapshot and final results")
    report = _verified_snapshot(root)
    _verified_readout(root, report)
    def link(name):
        return os.path.relpath(root / registry["files"][name]["path"], page.parent)
    outcome = ("A whole recipe passed all required checks; independent confirmation and family-selection review remain pending."
               if report["repair_readout"]["all_required_pass_candidates"] else
               "No whole recipe passed all required checks; the selected incumbent is retained.")
    observed = []
    for study, best in report["repair_readout"]["best_observed_by_study"].items():
        statuses = ", ".join("`" + task + "` " + status for task, status in sorted(best["task_statuses"].items()))
        observed.append(f"The best observed `{study}` recipe has **{best['required_passes']}/{best['required_total']} PASS** "
                        f"(`{best['candidate_id']}`): {statuses}.")
    return ("\n\nBCAP's [bounded repair readout](" + link("readout") + ") records " + str(len(report["rows"])) + " whole configurations under "
            f"view revision {report['view_revision']}. " + " ".join(observed) + " " + outcome + " "
            "The [source-bound numerical snapshot](" + link("evidence") + ") and [final outcomes](" + link("results") + ") "
            "retain every configuration and separate duration diagnostics. These observations supply no qualification "
            "for the older recorded board or the latest checkout.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "register"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit", action="append",
                        help="optional exact allowed executed source; repeat for distinct study cohorts")
    parser.add_argument("--expected-study", action="append", choices=sorted(STUDIES),
                        help="require exactly these admitted ordinary studies; repeat for the complete round")
    parser.add_argument("--readout", type=Path, default=DIRECTORY / "README.md")
    args = parser.parse_args()
    if args.stage == "freeze":
        outcome = freeze(args.root, args.source_commit, args.expected_study)
    else:
        outcome = register(args.root, args.readout)
    import json
    print(json.dumps(outcome, sort_keys=True), flush=True)

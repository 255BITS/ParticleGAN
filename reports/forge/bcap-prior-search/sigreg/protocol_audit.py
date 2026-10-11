"""Saved-only audit of SIGReg's frozen four-arm comparison; no training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workflow
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.knowledge import _attempts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    progress = read_json(args.archive / "progress.json")
    state = read_json(args.archive / "queue/queue/state.json")
    require = workflow.require
    require(progress["phase"] == "complete", "Wait for the true completed round")
    require(not any(e["status"] in {"running", "queued", "paused"} for e in state["submissions"].values())
            and not any(j["status"] == "running" for j in state["jobs"].values()), "Active work remains")
    records, _ = _attempts(ROOT)
    attempts = {record["attempt_id"]: record for record in records}
    entries = [state["submissions"][request_id] for request_id in
               [progress["control_request"], *progress["search_requests"]]]
    requests = [entry["request"] for entry in entries]
    control = requests[0]
    base_recipe = workflow.resolved_recipe(control["candidate"])
    rows = [workflow.validated_submission_rows(state, entry, attempts) for entry in entries]
    baseline_rows = {row["task_id"]: row for row in rows[0]}
    cohorts, initial_pairs, data_pairs, variable_length, applied_bindings = [], [], [], [], []
    request_evidence = []
    for index, (request, measured) in enumerate(zip(requests, rows)):
        require(request["source"]["digest"] == progress["source_digest"], "Source digest mismatch")
        require(request["source"]["origin_commit"] == progress["source_commit"], "Executed commit mismatch")
        require(request["source"]["files"] == control["source"]["files"], "Scientific source bytes changed")
        require(request["protocol"] == control["protocol"] and request["protocol"]["seed"] == 0, "Protocol changed")
        require(request["compute_profiles"] == control["compute_profiles"], "Compute cohort changed")
        require(request["view"] == control["view"], "Gate view changed")
        for task, card in control["tasks"].items():
            for key in ("adapter", "execution", "evaluation", "resources", "dependencies"):
                require(card.get(key) == request["tasks"][task].get(key), f"Task condition changed: {task}/{key}")
        recipe = workflow.resolved_recipe(request["candidate"])
        delta = {key: {"control": value, "candidate": recipe[key]} for key, value in base_recipe.items()
                 if recipe[key] != value}
        if index:
            require(set(delta) == {"prior_regularizer", "prior_reg"}, "Undeclared recipe delta")
            require(recipe["prior_regularizer"] == "sigreg" and recipe["prior_reg"] in [.0001, .001, .01], "Unregistered strength")
        cohorts.append({"candidate": request["candidate"]["id"], "delta": delta,
                        "certified_task_statuses": dict(Counter(row["gate_status"] for row in measured))})
        request_evidence.append({"request_id": request["request_id"], "candidate_revision": request["candidate_revision"],
                                 "request_sha256": stable_hash(request), "certified_rows_sha256": stable_hash(measured)})
        for row in measured:
            if row["task_id"] == "clockfree_audit_measurement_v1":
                continue
            applied = workflow.compact_row(row)
            actual_recipe = applied.get("recipe", {})
            require(actual_recipe.get("prior_reg", 0.0) == recipe["prior_reg"], "Applied coefficient differs from admitted request")
            require(actual_recipe.get("prior_regularizer", "vicreg") == recipe["prior_regularizer"], "Applied regularizer differs from admitted request")
            applied_bindings.append({"candidate": request["candidate"]["id"], "task": row["task_id"],
                                     "regularizer": actual_recipe.get("prior_regularizer", "vicreg"),
                                     "weight": actual_recipe.get("prior_reg", 0.0),
                                     "prior_policy": applied.get("prior_policy")})
        if not index:
            continue
        for row in measured:
            task = row["task_id"]
            if task == "clockfree_audit_measurement_v1":
                continue  # This separately scoped smaller probe supplies no feature-invariance credit.
            baseline = baseline_rows[task]
            actual = row.get("applied") or row
            original = baseline.get("applied") or baseline
            init, initial = actual.get("initialization"), original.get("initialization")
            if init is not None and initial is not None:
                require(init == initial, f"Initial state mismatch: {task}")
                initial_pairs.append({"candidate": request["candidate"]["id"], "task": task,
                                      "initialization_sha256": stable_hash(init)})
            evidence, reference = row.get("evidence", {}), baseline.get("evidence", {})
            steps, ref_steps = evidence.get("completed_steps", row.get("cost", {}).get("completed_steps")), reference.get("completed_steps", baseline.get("cost", {}).get("completed_steps"))
            if steps != ref_steps:
                variable_length.append({"task": task, "candidate": request["candidate"]["id"],
                                        "control_steps": ref_steps, "candidate_steps": steps,
                                        "scope": "same declared adaptive stopping; full consumed prefixes differ"})
            if evidence.get("data_sha256") is not None and steps == ref_steps:
                require(evidence["data_sha256"] == reference.get("data_sha256"), f"Seen data mismatch: {task}")
                data_pairs.append({"candidate": request["candidate"]["id"], "task": task,
                                   "completed_steps": steps, "data_sha256": evidence["data_sha256"]})
    document = {"schema_version": 1, "scope": "protocol/provenance audit; original certifications retained",
        "executed_source_commit": progress["source_commit"], "source_digest": progress["source_digest"],
        "passed": True, "request_evidence": request_evidence, "cohorts": cohorts,
        "checks": {"same_source_bytes": "PASS", "same_protocol_seed0": "PASS", "same_compute_cohort": "PASS",
                   "same_task_conditions_and_gates": "PASS", "only_declared_global_recipe_deltas": "PASS",
                   "certified_initial_state_pairs": len(initial_pairs), "identical_seen_data_digest_pairs": len(data_pairs),
                   "applied_recipe_bindings": len(applied_bindings)},
        "applied_recipe_bindings": applied_bindings,
        "initial_state_pairs": initial_pairs, "seen_data_pairs": data_pairs, "adaptive_execution_lengths": variable_length,
        "clock_probe": "optional smaller own-recipe diagnostic; no SIGReg feature-invariance credit",
        "publication_training_updates": 0, "publication_sampling_draws": 0}
    archive_path = ROOT / "reports/forge/recipe-prior-refactor/publication/results.json"
    fresh_path = Path(__file__).with_name("results.json")
    archived, fresh = read_json(archive_path), read_json(fresh_path)
    require(fresh["source_digest"] == progress["source_digest"], "Published source mismatch")
    reference = {row["task_id"]: row for row in archived["certified_results"]
                 if row["family"] == "bcap-pure" and row["scope"] == "ordinary"}
    actual = {row["task_id"]: row for row in fresh["control_results"]}
    require(set(actual) == set(reference) == set(baseline_rows), "Baseline task coverage differs")
    matches = []
    for task, before in sorted(reference.items()):
        after = actual[task]
        validated = workflow.compact_row(baseline_rows[task])
        require(after == validated, f"Published control differs from its original certified row: {task}")
        require(after["receipt_binding"]["valid_receipt"], f"Uncertified control: {task}")
        changed = [key for key in sorted(before["metrics"])
                   if key not in after["metrics"] or before["metrics"][key] != after["metrics"][key]]
        require(not changed and before["gate_status"] == after["gate_status"], f"Archived baseline changed: {task}")
        matches.append({"task": task, "gate_equal": True, "valid_fresh_receipt": True,
                        "changed_archived_metrics": changed,
                        "additional_metrics": sorted(set(after["metrics"]) - set(before["metrics"])),
                        "fresh_attempt_id": after["receipt_binding"]["attempt_id"],
                        "archived_attempt_id": before["attempt_id"]})
    document["baseline_reproduction_diagnostic"] = {
        "scope": "saved_evidence_baseline_reproduction_diagnostic",
        "baseline_path": str(archive_path.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archived_source_digest": archived["source_digest"],
        "fresh_results_sha256": hashlib.sha256(fresh_path.read_bytes()).hexdigest(),
        "exact_gate_matches": len(matches), "exact_archived_metric_matches": len(matches),
        "exact_metric_dictionary_matches": sum(not row["additional_metrics"] for row in matches),
        "tasks": matches, "archived_evidence_reused": False, "qualification_input": False,
        "optimizer_updates_added": 0, "sampling_draws_added": 0}
    atomic_json(Path(__file__).with_name("protocol-audit.json"), document)
    print(json.dumps(document["checks"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

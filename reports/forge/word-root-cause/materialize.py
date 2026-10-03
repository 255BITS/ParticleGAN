"""Publish compact task-only diagnostic receipts and immutable recall records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash

HERE = Path(__file__).resolve().parent


def write(path, value):
    if path.exists() and read_json(path) == value:
        return
    atomic_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = HERE / f"round{args.round}-protocol.json"
    protocol = read_json(protocol_path)
    rows, receipts = [], []
    for arm in protocol["arms"]:
        original = args.runs / arm["id"] / "compact-receipt.json"
        receipt = read_json(original)
        if (receipt["id"] != arm["id"] or receipt["protocol_sha256"] != file_hash(protocol_path)
                or receipt["recipe_delta"] != arm["recipe_delta"] or receipt["qualification_input"] is not False
                or receipt["grade"]["gate_status"] not in ("PASS", "FAIL")
                or receipt["cost"]["completed_steps"] != 20001):
            raise ValueError("diagnostic receipt does not match the frozen complete protocol")
        path = HERE / "receipts" / f"{arm['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and file_hash(path) != file_hash(original):
            raise ValueError("a previously published immutable diagnostic receipt changed")
        if not path.exists():
            shutil.copyfile(original, path)
        relative = str(path.relative_to(ROOT))
        source = {"path": relative, "sha256": file_hash(path)}
        convergence = receipt["grade"]["evaluator_result"]["convergence"]
        rows.append({"id": arm["id"], "family": arm["family"], "recipe_delta": arm["recipe_delta"],
                     "grade": receipt["grade"]["gate_status"], "convergence": convergence,
                     "final_metrics": receipt["final_metrics"], "cost": receipt["cost"], "receipt": source,
                     "source_commit": receipt["source_commit"], "source_digest": receipt["source_digest"],
                     "task_fingerprint": receipt["task_fingerprint"],
                     "guards": {name: receipt["guards"][name] for name in
                         ("all_finite", "hooks_exercised", "unintended_rng_deviations", "optimizer_updates")},
                     "a2": receipt["guards"]["mechanism_audit"]["mechanisms"]["a2"],
                     "initial_hashes": {role: row["initial_state_sha256"]
                         for role, row in receipt["host"]["models"].items()} |
                         {"prior": receipt["host"]["initial_prior_sha256"]},
                     "rng_manifest_sha256": receipt["rng_manifest_sha256"]})
        record_id = "word-diagnostic-" + stable_hash({"protocol": receipt["protocol_sha256"],
            "revision": receipt["candidate_revision"], "source": receipt["source_digest"]})[:24]
        status = receipt["grade"]["gate_status"]
        record = {"schema_version": 1, "record_id": record_id, "record_type": "scientific",
            "candidate_id": arm["id"], "candidate_revision": receipt["candidate_revision"],
            "goal": "discriminator_stability", "evidence_scope": "task_only_diagnostic",
            "lifecycle": "concluded", "qualification_input": False, "qualification_reuse": False,
            "mechanism_class": "floor_constant", "trainer_family": arm["family"],
            "hypothesis": protocol["purpose"], "changed_factors": arm["recipe_delta"],
            "source": source, "provenance": {"source_commit": receipt["source_commit"],
                "source_digest": receipt["source_digest"], "protocol_sha256": receipt["protocol_sha256"],
                "parent": receipt["parent"], "parent_sha256": receipt["parent_sha256"],
                "task_fingerprint": receipt["task_fingerprint"], "runtime": receipt["runtime"],
                "raw_artifacts": receipt["raw_artifacts"]},
            "task_results": [{"task_id": protocol["task"], "gate_status": status,
                "metrics": receipt["final_metrics"], "cost": {"wall_seconds": receipt["cost"]["wall_seconds"]},
                "reason": "Standalone full-budget diagnostic; unchanged 24-check/five-terminal-pass task grader."}],
            "conclusion": f"{status} on the unchanged finite five-word joint/inverse task; "
                f"{convergence['passing_observations']}/24 observations pass and "
                f"the passing terminal suffix is {convergence['passing_suffix']} (required >=5). "
                "This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.",
            "next_action": "Use the completed bounded readout before further research. Preserve original "
                "failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption."}
        write(ROOT / "reports/forge/records" / f"{record_id}.json", record)
        receipts.append(receipt)
    if len({stable_hash(row["initial_hashes"]) for row in rows}) != 1:
        raise ValueError("diagnostic arms did not share their declared initial states")
    if len({row["rng_manifest_sha256"] for row in rows}) != 1:
        raise ValueError("diagnostic arms did not share named-stream bindings")
    result = {"schema_version": 1, "id": protocol["id"], "protocol": {
        "path": str(protocol_path.relative_to(ROOT)), "sha256": file_hash(protocol_path)},
        "scope": protocol["purpose"], "qualification_input": False, "qualification_reuse": False,
        "eligible_for_default": False, "results_in_protocol_order": rows,
        "paid_adapter_seconds": sum(row["cost"]["wall_seconds"] for row in rows),
        "reserved_seconds": protocol["round_reserved_seconds"],
        "cost_scope": "Adapter execution, evaluation and durable state/receipt writing; "
            "source inspection, imports, archive/export and media rendering are excluded. No speed ranking.",
        "source_commits": sorted({row["source_commit"] for row in rows}),
        "source_digests": sorted({row["source_digest"] for row in rows}),
        "historical_metadata_note": "Round 1 request envelopes retain base configuration/search annotations "
            "as ancestor provenance; executed recipes and candidate_revision bind the new arms. "
            "Round 2 strips these annotations from the diagnostic candidate. No historical receipt is changed."}
    outcomes = []
    for prediction in protocol["predictions"]:
        if prediction["id"] == "clean-long-optimization-bundle":
            scoped = [row for row in rows if row["id"] in
                      ("k3p-clean-full-low-rate", "ka2-clean-full-low-rate")]
        elif prediction["id"] == "existing-r1r2-capacity":
            scoped = [row for row in rows if row["family"] == "r1r2"]
        else:
            scoped = [row for row in rows if row["family"] == prediction["id"].split("-", 1)[0]]
        supported = any(row["grade"] == "PASS" for row in scoped)
        outcomes.append({**prediction, "prediction_met": supported, "falsifier_triggered": not supported,
                         "observed_passing_arms": [row["id"] for row in scoped if row["grade"] == "PASS"]})
    result["prediction_results"] = outcomes
    write(HERE / f"round{args.round}-readout.json", result)
    print(json.dumps({"round": args.round, "completed_runs": len(rows),
                      "passing_runs": sum(row["grade"] == "PASS" for row in rows),
                      "paid_adapter_seconds": result["paid_adapter_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

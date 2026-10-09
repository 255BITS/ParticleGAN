"""Run one unattended Forge drain, then evaluate every frozen BCAP candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash, utc_now
from experiments.forge.queue import Queue, drain
from experiments.forge.search_space import enqueue_compilation, report_compilation
from reports.forge.regenerate_technique_inventory import project_receipt

STUDY = "bcap-tier2-search-v1"
CONFIGURATIONS = 72
OUTPUT = ROOT / "reports/forge/bcap-tier2-search"
MANIFEST = OUTPUT / "compiled.json"
LOCAL = ROOT / "runs/software/bcap-tier2-search"


def evaluate_all(queue):
    """Read results only after the standard coordinator has concluded the campaign."""
    state = queue.inspect()
    entries = [entry for entry in state["submissions"].values()
               if entry["request"].get("campaign_id") == STUDY]
    assert len(entries) == CONFIGURATIONS
    assert all(e["status"] not in {"queued", "running", "paused"} for e in entries)
    report = report_compilation(ROOT, queue.root, MANIFEST, queue=queue)
    assert report["selection"]["selection_complete"]
    campaign = state["campaigns"][STUDY]
    assert campaign["reserved_seconds"] == 0
    jobs = [job for job in state["jobs"].values()
            if (job.get("cost_owner") or {}).get("campaign") == STUDY]
    assert all(len(job["attempts"]) <= 1 for job in jobs), "No scientific retries"
    attempts = sorted(a["attempt_id"] for job in jobs for a in job["attempts"])
    charges = [row for row in state["charges"] if row["attempt_id"] in attempts]
    assert len(charges) == len(attempts) == len(set(attempts))
    assert math.isclose(sum(row["seconds"] for row in charges), campaign["spent_seconds"], abs_tol=1e-7)

    # Original receipt certificates bind every result, source and task key.
    receipts = [project_receipt(ROOT, attempt) for attempt in attempts]
    sources = {r["provenance"]["source_digest"] for r in receipts}
    assert sources == {read_json(OUTPUT / "plan.json")["source_digest"]}
    compact_receipts = [{"attempt_id": r["attempt_id"], "candidate_id": r["candidate_id"],
                         "candidate_revision": r["candidate_revision"], "provenance": r["provenance"],
                         "attempt_status": r["attempt_status"],
                         "task_results": r["task_results"]} for r in receipts]
    atomic_json(OUTPUT / "receipts.json", compact_receipts)

    # Task-owned conditions must match. Trainer-derived ownership annotations
    # are explicitly excluded from this authored-task comparison.
    conditions = {}
    for entry in entries:
        request = entry["request"]
        assert request["protocol"]["seed"] == 0
        assert request["execution_policy"]["mode"] == "complete_current_tier"
        recipe = request["candidate"]["resolved_recipe"]
        assert recipe["reg_arm"] == "b_cap" and recipe["reg_coeff"] > 0 and recipe["reg_kappa"] > 0
        for name, task in request["tasks"].items():
            task = {k: v for k, v in task.items() if k != "field_ownership"}
            conditions.setdefault(name, set()).add(stable_hash(task))
    assert all(len(hashes) == 1 for hashes in conditions.values())

    trials = []
    for trial in report["trials"]:
        tasks = [{key: row[key] for key in ("task", "qualification_tier", "importance", "gate_status",
                                          "metrics", "cost", "compatibility_key") if key in row}
                 for row in trial["tasks"]]
        counts = [sum(row["importance"] == "required" and row["qualification_tier"] == tier
                      and row["gate_status"] == "PASS" for row in tasks) for tier in (1, 2, 3)]
        trials.append({"candidate_id": trial["candidate_id"], "configuration_id": trial["configuration_id"],
            "candidate_revision": trial["candidate_revision"], "settings": trial["settings"],
            "optimizer": trial["resolved_recipe"]["optimizer_family"], "loss": trial["resolved_recipe"]["loss"],
            "submission_status": trial["submission_status"], "submission_blockers": trial["submission_blockers"],
            "required_passes_by_tier": counts, "qualified_tier": trial["qualification"]["qualified_tier"],
            "cost": trial["cost"], "attempt_ids": trial["attempt_ids"], "tasks": tasks})
    trials.sort(key=lambda t: (-t["required_passes_by_tier"][0], -t["required_passes_by_tier"][1],
                               t["configuration_id"]))
    selected = trials[0]
    assert selected["candidate_id"] == report["selection"]["selected_candidate_id"]
    gate_counts = dict(Counter(row["gate_status"] for trial in trials for row in trial["tasks"]
                               if row["importance"] == "required" and row["qualification_tier"] <= 2))
    output = {"schema_version": 1, "study_id": STUDY, "goal": "discriminator_stability",
        "view_revision": entries[0]["request"]["view"]["revision"], "required_counts": [6, 21, 2],
        "source_digest": next(iter(sources)),
        "source_origin_commits": sorted({r["provenance"]["source_origin_commit"] for r in receipts}),
        "runtime": entries[0]["request"]["runtime"],
        "compute_profiles": entries[0]["request"]["compute_profiles"],
        "manifest_hash": read_json(MANIFEST)["manifest_hash"], "queue_root": str(queue.root),
        "campaign": campaign, "selection": report["selection"], "trials": trials,
        "tier1_survivors": sum(t["required_passes_by_tier"][0] == 6 for t in trials),
        "high_score_target": 10, "target_met": selected["required_passes_by_tier"][0] == 6
                                                    and selected["required_passes_by_tier"][1] >= 10,
        "gate_counts": gate_counts, "attempt_count": len(attempts), "scientific_retries": 0,
        "receipt_certificates_validated": len(receipts), "all_bcap_on": True,
        "matched_task_contract_hashes": {name: next(iter(hashes)) for name, hashes in sorted(conditions.items())},
        "evaluated_after_all_terminal": True, "intermediate_scientific_interventions": 0,
        "default_adoption": False, "qualification_input": False,
        "independent_confirmation": "not_performed", "completed_at": utc_now()}
    atomic_json(OUTPUT / "readout.json", output)
    atomic_json(LOCAL / "full-report.json", report)
    print({"stage": "all_results_evaluated", "configurations": CONFIGURATIONS,
           "tier1_survivors": output["tier1_survivors"],
           "selected_passes_by_tier": selected["required_passes_by_tier"],
           "target_met": output["target_met"], "paid_seconds": campaign["spent_seconds"]}, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    queue = Queue(args.queue_root.resolve(), report_root=ROOT / "reports/forge", on_completion=None)
    LOCAL.mkdir(parents=True, exist_ok=True)
    try:
        if not args.report_only:
            assert args.gpus == "0,1", "This study declares both A6000 GPUs, one worker each"
            # Ensure room for original artifacts before any scientific spend.
            assert shutil.disk_usage(queue.root).free >= 20 * 1024**3
            admitted = enqueue_compilation(ROOT, queue.root, MANIFEST, queue=queue)
            assert admitted["submitted_count"] == CONFIGURATIONS
            assert all(not t["submission_blockers"] for t in admitted["trials"])
            atomic_json(LOCAL / "started.json", {"study_id": STUDY, "started_at": utc_now(),
                "queue_root": str(queue.root), "manifest_hash": admitted["manifest_hash"],
                "source_origin_commit": __import__("subprocess").check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "physical_gpus": [0, 1], "configurations": CONFIGURATIONS})
            print({"stage": "admitted", "configurations": CONFIGURATIONS, "queue_root": str(queue.root)}, flush=True)
            # One blocking coordinator call: no intermediate reports, ranking,
            # callbacks, new hypotheses or agent-managed worker interventions.
            drain(queue, args.gpus.split(","), campaign=STUDY, poll_seconds=2.)
        output = evaluate_all(queue)
        atomic_json(LOCAL / "completed.json", {"study_id": STUDY, "completed_at": utc_now(),
            "selected_candidate_id": output["selection"]["selected_candidate_id"],
            "target_met": output["target_met"], "readout": str(OUTPUT / "readout.json"),
            "readout_sha256": file_hash(OUTPUT / "readout.json")})
    except Exception:
        atomic_json(LOCAL / "failed.json", {"study_id": STUDY, "failed_at": utc_now(),
            "traceback": traceback.format_exc(), "automatic_scientific_retry": False})
        raise


if __name__ == "__main__":
    main()

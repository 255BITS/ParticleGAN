"""Wait for the original study, then run the separately frozen overnight batch."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash, utc_now
from experiments.forge.queue import Queue, drain
from experiments.forge.search_space import enqueue_compilation, report_compilation

STUDY = "bcap-overnight-search-v1"
OUTPUT = ROOT / "reports/forge/bcap-tier2-search/overnight"
LOCAL = ROOT / "runs/software/bcap-overnight-search"
PREDECESSOR = ROOT / "runs/software/bcap-tier2-search/completed.json"
CUTOFF = datetime(2026, 10, 9, 6, tzinfo=ZoneInfo("America/Denver"))


def load_runner():
    spec = importlib.util.spec_from_file_location("bcap_shared_runner", ROOT / "reports/forge/bcap-tier2-search/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STUDY, module.CONFIGURATIONS = STUDY, 24
    module.OUTPUT, module.MANIFEST, module.LOCAL = OUTPUT, OUTPUT / "compiled.json", LOCAL
    return module


def combine():
    original = read_json(ROOT / "runs/software/bcap-tier2-search/full-report.json")
    extra = read_json(LOCAL / "full-report.json")
    trials = original["trials"] + extra["trials"]
    assert len(trials) == len({t["configuration_id"] for t in trials}) == 96
    assert len({t["source_digest"] for t in trials}) == 1
    for field in ("runtime_cohort", "protocol_hash", "policy_fingerprint"):
        assert len({stable_hash(t[field]) for t in trials}) == 1
    selection = select_configuration(trials, 2)
    assert selection["selection_complete"]
    original_readout = read_json(ROOT / "reports/forge/bcap-tier2-search/readout.json")
    extra_readout = read_json(OUTPUT / "readout.json")
    atomic_json(ROOT / "reports/forge/bcap-tier2-search/combined-readout.json", {
        "schema_version": 1, "study_ids": ["bcap-tier2-search-v1", STUDY], "configurations": 96,
        "source_digest": trials[0]["source_digest"], "selection": selection,
        "tier1_survivors": original_readout["tier1_survivors"] + extra_readout["tier1_survivors"],
        "paid_worker_seconds": original_readout["campaign"]["spent_seconds"] + extra_readout["campaign"]["spent_seconds"],
        "attempt_count": original_readout["attempt_count"] + extra_readout["attempt_count"],
        "scientific_retries": 0, "domain_adapted_to_interim_scientific_results": False,
        "original_72_declarations_changed": False, "matched_source_runtime_protocol": True,
        "default_adoption": False, "qualification_input": False,
        "completed_at": utc_now(), "campaign_readouts": ["readout.json", "overnight/readout.json"]})
    print({"stage": "combined_evaluation", "configurations": 96,
           "selected_passes_by_tier": selection["required_passes_by_tier"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--finish-policy", choices=("complete_batch", "pause_at_6"), default="complete_batch")
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    atomic_json(LOCAL / "armed.json", {"study_id": STUDY, "armed_at": utc_now(),
        "predecessor_completion": str(PREDECESSOR), "finish_policy": args.finish_policy,
        "cutoff_local": CUTOFF.isoformat(), "configurations": 24,
        "predecessor_changed": False, "scientific_selection_performed": False})
    print({"stage": "waiting_for_original_72", "finish_policy": args.finish_policy}, flush=True)
    try:
        while not PREDECESSOR.exists():
            if (PREDECESSOR.parent / "failed.json").exists():
                raise RuntimeError("Original study failed operationally; extension remains unadmitted")
            if args.finish_policy == "pause_at_6" and datetime.now(CUTOFF.tzinfo) >= CUTOFF:
                atomic_json(LOCAL / "not-started.json", {"reason": "06:00 cutoff reached before original completion",
                                                       "scientific_updates_added": 0})
                return
            time.sleep(10)
        if args.finish_policy == "pause_at_6" and datetime.now(CUTOFF.tzinfo) >= CUTOFF:
            atomic_json(LOCAL / "not-started.json", {"reason": "Original finished after cutoff", "scientific_updates_added": 0})
            return
        runner = load_runner()
        queue = Queue(args.queue_root.resolve(), report_root=ROOT / "reports/forge", on_completion=None)
        admitted = enqueue_compilation(ROOT, queue.root, runner.MANIFEST, queue=queue)
        assert admitted["submitted_count"] == 24 and all(not t["submission_blockers"] for t in admitted["trials"])
        atomic_json(LOCAL / "started.json", {"study_id": STUDY, "started_at": utc_now(),
            "predecessor_completion_sha256": file_hash(PREDECESSOR), "finish_policy": args.finish_policy,
            "manifest_hash": admitted["manifest_hash"], "configurations": 24, "physical_gpus": [0, 1]})
        print({"stage": "admitted", "configurations": 24}, flush=True)
        stop_timer = threading.Event()

        def pause_at_cutoff():
            while not stop_timer.wait(2):
                if datetime.now(CUTOFF.tzinfo) >= CUTOFF:
                    queue.pause(STUDY, True)
                    return

        if args.finish_policy == "pause_at_6":
            threading.Thread(target=pause_at_cutoff, daemon=True).start()
        drain(queue, ["0", "1"], campaign=STUDY, poll_seconds=2.)
        stop_timer.set()
        if queue.inspect()["campaigns"][STUDY]["paused"]:
            report = report_compilation(ROOT, queue.root, runner.MANIFEST, queue=queue)
            atomic_json(LOCAL / "partial-report.json", report)
            atomic_json(LOCAL / "paused.json", {"study_id": STUDY, "paused_at": utc_now(),
                "reason": "User-selected 06:00 new-launch cutoff; active tasks completed",
                "remaining_cells": "unmeasured; immutable queue retained for explicit resume"})
            return
        result = runner.evaluate_all(queue)
        combine()
        atomic_json(LOCAL / "completed.json", {"study_id": STUDY, "completed_at": utc_now(),
            "readout_sha256": file_hash(OUTPUT / "readout.json"),
            "combined_readout": str(OUTPUT.parent / "combined-readout.json"),
            "target_met": result["target_met"]})
    except Exception:
        atomic_json(LOCAL / "failed.json", {"study_id": STUDY, "failed_at": utc_now(),
            "traceback": traceback.format_exc(), "automatic_scientific_retry": False})
        raise


if __name__ == "__main__":
    main()

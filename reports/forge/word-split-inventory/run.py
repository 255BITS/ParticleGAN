"""Replay the five selected configurations through the ordinary CUDA tier gates.

Run from the repository root. Raw output belongs in runs/, not in Git.
Registration has one fixed configuration per family; no tuning occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.forge.configuration_search import enqueue_search, plan_search, report_search
from experiments.forge.queue import Queue, drain
from experiments.forge.trainer_families import current_family_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--stage", choices=("plan", "run", "report"), default="plan")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    root = args.root.resolve()
    queue_root = (args.queue_root or root / "runs/forge-word-split").resolve()
    devices = args.gpus.split(",")
    if not devices or any(not device.isdecimal() for device in devices):
        parser.error("physical CUDA indices are required")
    specs = sorted((root / "configs/forge/searches").glob("word-split-inventory-*-v1.json"))
    if len(specs) != 5:
        raise ValueError("expected the five frozen runnable family registrations")
    queue = Queue(queue_root, report_root=root / "reports/forge", on_completion=None)
    if args.stage != "report":
        # Check every recipe before admitting any work. Saved v1 configuration
        # identities retain their original prior/protocol metadata.
        selected = {item["candidate_id"] for item in current_family_candidates(root)}
        for path in specs:
            plan = plan_search(root, queue_root, path, queue=queue)
            if len(plan["trials"]) != 1 or plan["trials"][0]["candidate_id"] not in selected:
                raise ValueError("registration differs from the selected whole configuration: " + str(path))
            if plan["trials"][0]["submission_blockers"]:
                raise ValueError(str(plan["trials"][0]["submission_blockers"]))
            print(json.dumps({"study": plan["study_id"], "candidate": plan["trials"][0]["candidate_id"],
                              "stage": "planned", "through_tier": 2}), flush=True)
    if args.stage == "plan":
        return
    if args.stage == "run":
        for path in specs:
            result = enqueue_search(root, queue_root, path, queue=queue)
            print(json.dumps({"study": result["study_id"], "stage": result["stage"]}), flush=True)
        drain(queue, devices, campaign="technique-inventory-word-split-v1")
    for path in specs:
        result = report_search(root, queue_root, path, queue=queue)
        print(json.dumps({"study": result["study_id"], "stage": result["stage"],
                          "qualification": result["trials"][0]["qualification"]}), flush=True)


if __name__ == "__main__":
    main()

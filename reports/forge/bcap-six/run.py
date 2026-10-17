"""Run the declared BCAP comparison through Forge; keep bulk output in runs/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import enqueue_search, plan_search, report_search
from experiments.forge.queue import Queue, drain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--stage", choices=("plan", "run", "report"), default="plan")
    parser.add_argument("--search", default="bcap-six-smoothing-v1")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    root = args.root.resolve()
    queue_root = (args.queue_root or root / "runs/forge").resolve()
    queue = Queue(queue_root, report_root=root / "reports/forge", on_completion=None)
    if args.stage == "plan":
        print(json.dumps(plan_search(root, queue_root, args.search, queue=queue), indent=2))
        return
    if args.stage == "run":
        planned = plan_search(root, queue_root, args.search, queue=queue)
        if any(trial["submission_blockers"] for trial in planned["trials"]):
            raise ValueError("Resolve all declared submission blockers before spending")
        admitted = enqueue_search(root, queue_root, args.search, queue=queue)
        print(json.dumps({"study": admitted["study_id"], "stage": admitted["stage"]}), flush=True)
        drain(queue, args.gpus.split(","), campaign=admitted["campaign"]["id"])
    print(json.dumps(report_search(root, queue_root, args.search, queue=queue), indent=2), flush=True)


if __name__ == "__main__":
    main()

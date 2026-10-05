"""Execute only the frozen Pure BCAP initial round on GPU 0 and GPU 1."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import enqueue_search, plan_search, report_search
from experiments.forge.contracts import read_json
from experiments.forge.queue import Queue, drain


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise ValueError("Execute the reviewed source commit")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=True)
    plan = read_json(ROOT / "reports/forge/pure-bcap/plans.json")
    queue_root = ROOT / "runs/forge" / plan["round"] / "queue"
    queue = Queue(queue_root, report_root=ROOT / "reports/forge", on_completion=None)
    for spec in plan["specs"]:
        reviewed = plan_search(ROOT, queue_root, ROOT / spec, queue=queue)
        assert reviewed["source_digest"] == plan["source_digest"]
        assert all(t["submission_status"] == "READY" for t in reviewed["trials"])
    for spec in plan["specs"]:
        summary = enqueue_search(ROOT, queue_root, ROOT / spec, queue=queue)
        assert summary["blocked_count"] == 0 and summary["submitted_count"] == 2
        assert summary["source_digest"] == plan["source_digest"]
        emit({"event": "search_enqueued", "study": summary["study_id"], "candidates": 2, "source_commit": head})
    emit({"event": "round_started", "gpus": [0, 1], "cpu_workers": 1,
          "logs": str(queue_root / "events.jsonl"), "candidate_count": 10,
          "maximum_paid_seconds": plan["campaign_cap_seconds"]})
    drain(queue, ["0", "1"], workers_per_gpu=1, allow_sharing=False, campaign=plan["round"])
    for spec in plan["specs"]:
        summary = report_search(ROOT, queue_root, ROOT / spec, queue=queue)
        assert summary["selection"]["all_trials_terminal"]
        emit({"event": "search_complete", "study": summary["study_id"], "selection": summary["selection"],
              "trials": [{"candidate": t["candidate_id"], "settings": t["settings"], "cost": t["cost"]} for t in summary["trials"]]})


if __name__ == "__main__":
    main()

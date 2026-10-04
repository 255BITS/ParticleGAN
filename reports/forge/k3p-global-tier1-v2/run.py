"""Execute the frozen twelve-configuration ordinary Tier 1 search.

Uses existing Forge search/queue admission, independent grading and budget
accounting. GPU0 and the coordinator's CPU fallback each permit one worker.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import enqueue_search, report_search
from experiments.forge.contracts import read_json
from experiments.forge.queue import Queue, drain


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or sys.executable != "/usr/bin/python":
        raise ValueError("Run the reviewed commit with the declared scientific Python")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=True)
    plan = read_json(ROOT / "reports/forge/k3p-global-tier1-v2/plans.json")
    queue_root = ROOT / "runs/forge/k3p-global-tier1-v2/queue"
    queue = Queue(queue_root, report_root=ROOT / "reports/forge", on_completion=None)
    spec = "configs/forge/searches/k3p-global-tier1-v2.json"
    summary = enqueue_search(ROOT, queue_root, spec, queue=queue)
    assert summary["blocked_count"] == 0 and summary["submitted_count"] == 12
    assert summary["source_digest"] == plan["source_digest"]
    for trial in summary["trials"]:
        request = queue.inspect()["submissions"][trial["request_id"]]["request"]
        assert request["source"]["origin_commit"] == head
        assert request["runtime"]["python"] == "3.14.7"
        assert request["decision_review"]["status"] == "READY"
    emit({"event": "search_enqueued", "source_commit": head,
          "source_digest": summary["source_digest"], "configurations": 12,
          "logs": str(queue_root / "events.jsonl"), "maximum_workers": 2})
    drain(queue, ["0"], workers_per_gpu=1, allow_sharing=False,
          campaign="k3p-global-tier1-v2")
    summary = report_search(ROOT, queue_root, spec, queue=queue)
    assert summary["selection"]["all_trials_terminal"]
    emit({"event": "search_complete", "selection": summary["selection"],
          "trials": [{"candidate": t["candidate_id"], "settings": t["settings"],
                      "status": t["status"], "attempts": t["attempt_ids"],
                      "cost": t["cost"]} for t in summary["trials"]]})


if __name__ == "__main__":
    main()

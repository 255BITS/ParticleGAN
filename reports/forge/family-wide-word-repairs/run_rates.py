"""Run the six frozen ordinary global configurations in one serial batch.

This composes existing strict search admission and Queue/drain APIs, retaining
ordinary prerequisites, independent grading, durable certificates and cost
accounting. Full memory/leaderboard compilation is deferred to publication.
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
        raise ValueError("Run only the reviewed source freeze with the declared scientific Python")
    plan = read_json(ROOT / "reports/forge/family-wide-word-repairs/rates-plans.json")
    assert plan["configuration_count"] == 6
    assert plan["campaign_ceiling_seconds"] == 293400
    queue_root = ROOT / "runs/forge/family-wide-word-repair-rates-v1/queue"
    queue = Queue(queue_root, report_root=ROOT / "reports/forge", on_completion=None)
    count = 0
    for family in plan["families"]:
        summary = enqueue_search(ROOT, queue_root, family["spec"], queue=queue)
        assert summary["blocked_count"] == 0 and summary["submitted_count"] == 2
        assert summary["source_digest"] == plan["source_digest"]
        for trial in summary["trials"]:
            request = queue.inspect()["submissions"][trial["request_id"]]["request"]
            assert request["source"]["origin_commit"] == head
            assert request["runtime"]["python"] == "3.14.7"
            assert request["decision_review"]["status"] == "READY"
        count += summary["submitted_count"]
        emit({"event": "study_enqueued", "study": family["study"], "source_commit": head,
              "source_digest": summary["source_digest"], "configs": 2,
              "logs": str(queue_root / "events.jsonl")})
    assert count == 6
    drain(queue, ["0"], workers_per_gpu=1, allow_sharing=False)
    for family in plan["families"]:
        summary = report_search(ROOT, queue_root, family["spec"], queue=queue)
        emit({"event": "study_complete", "study": family["study"],
              "selection": summary["selection"],
              "trials": [{"candidate": t["candidate_id"], "settings": t["settings"],
                  "status": t["status"], "attempts": t["attempt_ids"],
                  "cost": t["cost"]} for t in summary["trials"]]})
    emit({"event": "round_complete", "configurations": 6,
          "memory_compilation": "deferred summaries-only publication", "source_commit": head})


if __name__ == "__main__":
    main()

"""Run the frozen selected BCAP recipe through Tier 2, reusing its Tier 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.queue import Queue, drain

STUDY = "bcap-tier2-smoothing-v1"
CANDIDATE = "bcap-dualnorm--8db70e3cb9fd3da9b5cc6a117731e8572cba837d64e7ee721d012d3157c9a3fe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "enqueue", "drain"), default="plan")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    queue = Queue(ROOT / "runs/forge", report_root=ROOT / "reports/forge", on_completion=None)
    if args.stage == "drain":
        drain(queue, args.gpus.split(","), campaign=STUDY)
        print(json.dumps(queue.inspect()["campaigns"][STUDY]), flush=True)
        return
    request = resolve_idea(ROOT, CANDIDATE, study=STUDY, queue_root=queue.root,
                           freeze_source=args.stage == "enqueue")
    summary = plan_summary(request, queue.inspect())
    prior = read_json(ROOT / "reports/forge/bcap-six/readout.json")["selection"]
    assert request["candidate_revision"] == "72d9237558743f721c8636feb4642e4790c0f54ac1b139fdd0948daf096dfbf8"
    assert request["source"]["digest"] == prior["source_digest"]
    assert not request["preflight_blockers"]
    assert request["execution_policy"]["mode"] == "complete_current_tier"
    authorized = [task for task in summary["tasks"] if task["permitted_by_tier_cap"]]
    assert all(task["reusable"] for task in authorized if task["qualification_tier"] == 1)
    assert sum(task["qualification_tier"] == 2 and task["importance"] == "required"
               for task in authorized) == 21
    assert summary["worst_case_seconds"] <= 40500
    deltas = request["study_review"]["expected"]["substantive_delta"]
    assert deltas and all(delta["path"][0] == "recipe" and delta["path"][-1] == "optimizer_smoothing"
                          and delta["before"] == 0 and delta["after"] == 1e-5 for delta in deltas)
    if args.stage == "enqueue":
        entry = queue.submit(request, request["study"]["campaign"])
        summary["request_id"] = entry["request"]["request_id"]
        atomic_json(ROOT / "reports/forge/bcap-tier2/plan.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

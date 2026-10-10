"""Execute one reviewed Pure BCAP round on GPU 0 and GPU 1."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import enqueue_search, plan_search, report_search
from experiments.forge.contracts import read_json, stable_hash
from experiments.forge.queue import Queue, drain


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--plan", choices=("plans.json", "repair-plans.json"), default="plans.json")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise ValueError("Execute the reviewed source commit")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT, check=True)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise ValueError("GPU 0 and GPU 1 must be available; no substitute cohort will be admitted")
    if any(torch.cuda.get_device_name(index) != "NVIDIA RTX A6000" for index in (0, 1)):
        raise ValueError("The reviewed cohort requires two NVIDIA RTX A6000 GPUs")
    plan = read_json(ROOT / "reports/forge/pure-bcap" / args.plan)
    queue_root = ROOT / "runs/forge" / plan["round"] / "queue"
    queue = Queue(queue_root, report_root=ROOT / "reports/forge", on_completion=None)
    expected = {trial["candidate_id"]: trial for trial in plan["trials"]}
    observed = set()
    for spec in plan["specs"]:
        if "spec_sha256" in plan:
            require(stable_hash(read_json(ROOT / spec)) == plan["spec_sha256"][spec],
                    "Search declaration or finite campaign differs from the reviewed plan")
        reviewed = plan_search(ROOT, queue_root, ROOT / spec, queue=queue)
        require(reviewed["source_digest"] == plan["source_digest"], "Source differs from the reviewed plan")
        require(all(t["submission_status"] == "READY" for t in reviewed["trials"]), "Review submission blockers before running")
        for trial in reviewed["trials"]:
            frozen = expected[trial["candidate_id"]]
            require(trial["candidate_revision"] == frozen["candidate_revision"], "Candidate revision differs from the reviewed plan")
            require(trial["settings"] == frozen["settings"], "Recipe settings differ from the reviewed plan")
            if "scientific_signature" in frozen:
                require(trial["scientific_signature"] == frozen["scientific_signature"], "Scientific bindings differ from the reviewed plan")
                require(stable_hash(trial["runtime_cohort"]) == frozen["runtime_cohort_sha256"], "Runtime or hardware differs from the reviewed plan")
            observed.add(trial["candidate_id"])
    require(observed == set(expected) and len(observed) == plan["configuration_count"], "Candidate roster differs from the reviewed plan")
    for spec in plan["specs"]:
        summary = enqueue_search(ROOT, queue_root, ROOT / spec, queue=queue)
        require(summary["blocked_count"] == 0 and summary["submitted_count"] == 2, "Search admission is incomplete")
        require(summary["source_digest"] == plan["source_digest"], "Admitted source differs from the reviewed plan")
        emit({"event": "search_enqueued", "study": summary["study_id"], "candidates": 2, "source_commit": head})
    emit({"event": "round_started", "gpus": [0, 1], "cpu_workers": 1,
          "logs": str(queue_root / "events.jsonl"), "candidate_count": plan["configuration_count"],
          "maximum_paid_seconds": plan["campaign_cap_seconds"]})
    drain(queue, ["0", "1"], workers_per_gpu=1, allow_sharing=False, campaign=plan["round"])
    for spec in plan["specs"]:
        summary = report_search(ROOT, queue_root, ROOT / spec, queue=queue)
        require(summary["selection"]["all_trials_terminal"], "Read out incomplete trials before concluding the round")
        emit({"event": "search_complete", "study": summary["study_id"], "selection": summary["selection"],
              "trials": [{"candidate": t["candidate_id"], "settings": t["settings"], "cost": t["cost"]} for t in summary["trials"]]})


if __name__ == "__main__":
    main()

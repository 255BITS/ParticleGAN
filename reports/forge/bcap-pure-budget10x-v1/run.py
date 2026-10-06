"""Freeze and run exactly two owner-requested BCAP-pure budget diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, stable_hash
from experiments.forge.planning import resolve_idea, plan_summary
from experiments.forge.queue import Queue, drain

STUDY = "bcap-pure-budget10x-v1"
CANDIDATE = "bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9"
QUEUE = Path("/home/martyn/dev/ParticleGAN/runs/forge/bcap-pure-budget10x-v1-queue")


def emit(**values):
    print(json.dumps(values, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--queue-root", type=Path, default=QUEUE)
    args = parser.parse_args()
    queue = Queue(args.queue_root, report_root=ROOT / "reports/forge", on_completion=None)
    request = resolve_idea(ROOT, CANDIDATE, queue_root=args.queue_root,
                           freeze_source=args.action == "run", execution_backend="cuda", study=STUDY)
    summary = plan_summary(request, queue.inspect())
    if args.action == "plan":
        args.queue_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.queue_root / "plan.json", summary)
        emit(event="plan", tasks=summary["tasks"],
             worst_case_seconds=summary["worst_case_seconds"],
             preflight_blockers=summary["preflight_blockers"])
        return
    # One declared whole recipe and exactly the two changed-budget hosts.
    expected = {"gaussian1d_acquisition_budget10x_v1": 10_000,
                "ring16_acquisition_budget10x_v1": 4_000}
    if set(request["tasks"]) != set(expected):
        raise ValueError("unexpected additional task in diagnostic request")
    for task_id, steps in expected.items():
        task = request["tasks"][task_id]
        if task["execution"]["steps"] != steps or task["evaluation"]["observations"] != 240:
            raise ValueError("training allowance/cadence differs from owner request")
    if request["protocol"]["seed"] != 0 or request["view"].get("evidence_scope") != "research_diagnostic":
        raise ValueError("wrong protocol or qualification scope")
    campaign = request["study"]["campaign"]
    if campaign["budget_seconds"] != 4200:
        raise ValueError("unexpected campaign reservation")
    entry = queue.submit(request, campaign)
    frozen = entry["request"]
    receipt = {"schema_version": 1, "qualification_input": False,
               "study": STUDY, "candidate_id": CANDIDATE,
               "request_id": frozen["request_id"], "source": {
                   k: frozen["source"][k] for k in ("origin_commit", "digest")},
               "protocol": frozen["protocol"], "tasks": expected,
               "jobs": [{k: j[k] for k in ("task_ids", "compatibility_key", "budget_seconds")}
                        for j in frozen["jobs"]],
               "campaign": campaign, "devices": [0, 1], "automatic_retries": 0}
    receipt["input_digest"] = stable_hash(receipt)
    atomic_json(Path(__file__).parent / "freeze.json", receipt)
    emit(event="submitted", request_id=frozen["request_id"], source=receipt["source"],
         queue=str(args.queue_root), logs=str(args.queue_root / "events.jsonl"))
    stop = threading.Event()

    def progress():
        while not stop.wait(30):
            state = queue.inspect()
            active = state["campaigns"].get(STUDY, {})
            emit(event="progress", spent_seconds=active.get("spent_seconds"),
                 reserved_seconds=active.get("reserved_seconds"),
                 submission_status=state["submissions"].get(frozen["request_id"], {}).get("status"))

    watcher = threading.Thread(target=progress, daemon=True)
    watcher.start()
    try:
        drain(queue, ["0", "1"], campaign=STUDY)
    finally:
        stop.set()
        watcher.join()
    state = queue.inspect()
    emit(event="finished", submission_status=state["submissions"][frozen["request_id"]]["status"],
         spent_seconds=state["campaigns"][STUDY]["spent_seconds"],
         reserved_seconds=state["campaigns"][STUDY]["reserved_seconds"])


if __name__ == "__main__":
    main()

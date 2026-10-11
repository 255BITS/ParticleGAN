"""Off-ladder diagnostic of the two target gates for both frozen requests.

Both registered requests stopped at Tier 1, so their Tier 2 target cells stay
BLOCKED. This replays the unchanged frozen requests' gaussian1d_stability (from
each arm's own certified gaussian1d_smoke state) and mode_hold through the same
run_task path and worker determinism settings. Results are a separate
diagnostic cohort: no qualification credit, no retry of any registered cell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from experiments.forge.adapters import run_task
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.views import grade_result

TASKS = ("mode_hold", "gaussian1d_stability")


def prerequisites(state, request, task_id):
    result = {}
    for dependency in request["tasks"][task_id].get("dependencies", []):
        required = dependency["task"] if isinstance(dependency, dict) else dependency
        definition = next(j for j in request["jobs"] if required in j.get("task_ids", [j["task_id"]]))
        saved = state["jobs"][definition["compatibility_key"]]["result"]
        row = next(row for row in saved["task_results"] if row["task_id"] == required)
        result[required] = {"attempt_id": saved["attempt_id"], "result_hash": stable_hash(saved), "result": row,
                            "candidate_revision": saved["candidate_revision"],
                            "compatibility_key": definition["compatibility_key"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    progress = read_json(args.archive / "progress.json")
    queue = Queue(args.archive / "queue", report_root=ROOT / "reports/forge", on_completion=None)
    state = queue.inspect()
    arms = [("control", progress["control_request"])] + [("idle1.0", rid) for rid in progress["search_requests"]]
    summary = []
    for arm, request_id in arms:
        request = state["submissions"][request_id]["request"]
        for task_id in TASKS:
            job = next(j for j in request["jobs"] if task_id in j.get("task_ids", [j["task_id"]]))
            output = args.archive / "diagnostic" / arm / task_id
            output.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            raw = run_task(request, {**job, "prerequisites": prerequisites(state, request, task_id)}, output, args.device)
            row = {**raw, "task_id": task_id}
            torch.save(raw, output / "diagnostic-result.pt")
            grade = grade_result(request["tasks"][task_id], row)
            idle = row.get("evidence", {}).get("guards", {}).get("mechanism_audit", {}).get(
                "mechanisms", {}).get("generator_idle", {})
            entry = dict(arm=arm, task_id=task_id, request_id=request_id, candidate_revision=request["candidate_revision"],
                         source_digest=request["source"]["digest"], gate_status=grade.get("gate_status"),
                         metrics=grade.get("metrics", {}), evaluator=grade.get("evaluator_result", {}),
                         idle_steps=idle.get("applied"), decisions=idle.get("calls"),
                         wall_seconds=time.monotonic() - started, qualification_credit=False)
            summary.append(entry)
            print(json.dumps({k: entry[k] for k in ("arm", "task_id", "gate_status", "idle_steps", "decisions",
                                                    "wall_seconds")}), flush=True)
    atomic_json(args.archive / "diagnostic" / "summary.json", dict(schema_version=1, cohort="off_ladder_target_diagnostic",
                qualification_credit=False, rows=summary))


if __name__ == "__main__":
    main()

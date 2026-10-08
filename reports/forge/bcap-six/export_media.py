"""Export a named complete BCAP candidate's saved training; add no draws or updates."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.tier1_media import export_attempt


def main():
    import importlib.util
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate")
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = args.candidate or read_json(root / "reports/forge/bcap-six/readout.json")["selection"]["selected_candidate_id"]
    state = read_json(root / "runs/forge/queue/state.json")
    requests = {key: entry["request"] for key, entry in state["submissions"].items()
                if entry["request"]["candidate"]["id"] == candidate}
    assert len(requests) == 1
    required = {assignment["task"] for request in requests.values() for assignment in request["view"]["assignments"]
                if assignment["qualification_tier"] == 1 and assignment["importance"] == "required"}
    jobs = [job for job in state["jobs"].values() if job.get("result") and set(job["subscribers"]) & requests.keys()]
    rows = {row["task_id"]: row for job in jobs for row in job["result"]["task_results"]}
    assert len(required) == 6 and all(rows[name]["gate_status"] == "PASS" for name in required)
    helper = root / "reports/forge/gaussian-smoke-inventory/export_media.py"
    spec = importlib.util.spec_from_file_location("saved_media_guard", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = root / "reports/forge/bcap-six/media"
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    with module.forbid_live_execution():
        for job in sorted(jobs, key=lambda item: item["definition"]["task_id"]):
            attempt = job["result"]["attempt_id"]
            request = requests[next(key for key in job["subscribers"] if key in requests)]
            for receipt in export_attempt(root / "reports/forge/attempts" / attempt, output):
                task = receipt["task_id"]
                entries.append({"task_id": task, "attempt_id": attempt, "recorded_grade": rows[task]["gate_status"],
                    "source_commit": request["source"]["origin_commit"], "source_digest": request["source"]["digest"],
                    "gif": task + ".gif", "gif_sha256": file_hash(output / (task + ".gif")),
                    "renderer_receipt": task + ".json", "renderer_receipt_sha256": file_hash(output / (task + ".json"))})
                print(task, receipt["observation_count"], flush=True)
    assert required <= {entry["task_id"] for entry in entries}
    atomic_json(output / "index.json", {"schema_version": 1, "candidate_id": candidate, "tasks": entries,
        "optimizer_updates_added": 0, "sampling_draws_added": 0, "qualification_input": False,
        "selection_policy": "All executed tasks from the explicitly named complete candidate; no per-task candidate substitution.",
        "live_execution_guard": "Model construction/forward, trainer step/sample and prior sampling disabled during export.",
        "exporter_sha256": file_hash(Path(__file__))})


if __name__ == "__main__":
    main()

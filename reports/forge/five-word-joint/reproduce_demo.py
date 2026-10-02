"""Execute the frozen short API integration demo and export only compact evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import signal
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_audit.api_run import render_gif
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sources import inspect_source
from experiments.forge.views import grade_result, load_tasks, task_fingerprint
from experiments.forge.word_adapter import run_word


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = ROOT / "reports/forge/five-word-joint/demo-protocol.json"
    protocol = read_json(protocol_path)
    task = load_tasks(ROOT)[protocol["task"]]
    candidate_path = ROOT / f"configs/forge/ideas/{protocol['candidate']}.json"
    candidate = read_json(candidate_path)
    # Exclusive output ownership preserves earlier failed attempts.
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "protocol.json", protocol)
    torch.set_num_threads(protocol["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    source = inspect_source(ROOT, extra_paths=[str(protocol_path.relative_to(ROOT)), str(candidate_path.relative_to(ROOT)),
                                              f"configs/forge/tasks/{task['id']}.json", str(Path(__file__).relative_to(ROOT))])
    def timeout(signum, frame):
        raise TimeoutError("preregistered 60-second API demonstration budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, protocol["wall_seconds"])
    request = {"candidate": candidate, "protocol": {"seed": protocol["seed"]},
               "candidate_revision": stable_hash(candidate), "tasks": {task["id"]: task}}
    raw, records = run_word(request, task, args.output, protocol["device"],
                            execution_limit=protocol["execution_limit"], capture_media=True)
    grade = grade_result(task, raw)
    if grade["gate_status"] != "INCOMPLETE":
        raise RuntimeError("a reduced integration demo must not qualify the full task")
    instantaneous = "PASS" if records[-1]["passed"] else "FAIL"
    case = {"id": protocol["id"], "legacy_ids": ["source-family-15", "develop-five_word_joint"],
            "goal": task["description"], "scope": protocol["purpose"], "default_recipe": "ka2",
            "default_steps": task["execution"]["steps"], "eval_samples": protocol["eval_samples"],
            "sampling": {"evaluation": raw["evidence"]["sampling_law"]},
            "thresholds": task["evaluation"]["thresholds"]}
    indices = np.rint(np.linspace(0, len(records) - 1, protocol["media_frames"])).astype(int)
    selected = [records[index] for index in indices]
    render_gif(case, selected, args.output / "goal.gif", full_budget=False,
               requested_steps=protocol["execution_limit"], final_verdict="INCOMPLETE")
    arrays = {f"step{record['step']}_view{index}_{role}": view[role].numpy()
              for record in records for index, view in enumerate(record["views"]) for role in ("target", "samples")}
    np.savez_compressed(args.output / "observations.npz", **arrays)
    artifacts = {name: {"sha256": file_hash(args.output / name), "bytes": (args.output / name).stat().st_size}
                 for name in ("adapter-receipt.json", "state.pt", "observations.npz", "goal.gif")}
    compact = {"id": case["id"], "legacy_ids": case["legacy_ids"], "goal": case["goal"], "scope": case["scope"],
        "execution_status": "COMPLETE", "verdict": "INCOMPLETE", "instantaneous_metric": instantaneous,
        "failed_bounds": records[-1]["failed_bounds"], "full_task_grade": grade,
        "completed_updates": protocol["execution_limit"], "default_updates": task["execution"]["steps"],
        "final_metrics": deepcopy(raw["evidence"]["live"]), "recipe": raw["recipe"], "prior": raw["prior"],
        "guards": raw["evidence"]["guards"], "host": raw["evidence"]["host"],
        "initializer": raw["initializer"], "initialization_sha256": stable_hash(raw["initialization"]),
        "rng_version": raw["rng"]["version"], "rng_manifest_sha256": stable_hash(raw["rng"]),
        "api_components": raw["api_components"], "task_fingerprint": task_fingerprint(task),
        "candidate_revision": request["candidate_revision"], "protocol": protocol,
        "protocol_sha256": file_hash(protocol_path), "candidate_sha256": file_hash(candidate_path),
        "source_commit": source["origin_commit"], "source_identity": source["digest"],
        "source_bindings": {path: source["files"][path] for path in sorted(set(task["evaluation"]["sources"]) |
            {"experiments/forge/word_adapter.py", "experiments/forge/api.py", "experiments/forge/rng.py",
             "experiments/forge/sampling.py", str(Path(__file__).relative_to(ROOT))})},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "device": protocol["device"],
                    "torch_threads": torch.get_num_threads()}, "cost": raw["cost"],
        "raw_artifacts": artifacts, "raw_artifact_directory": str(args.output.resolve()),
        "gif": "goal.gif", "gif_sha256": artifacts["goal.gif"]["sha256"], "qualification_input": False}
    atomic_json(args.output / "compact-receipt.json", compact)
    publication = {"schema_version": 1, "cases": [case], "readouts": [{key: compact[key] for key in
        ("id", "legacy_ids", "goal", "scope", "execution_status", "verdict", "failed_bounds", "completed_updates",
         "default_updates", "source_commit", "gif")}], "runs": [compact],
        "source_identities": {source["digest"]: {"origin_commit": source["origin_commit"], "files": compact["source_bindings"]}}}
    atomic_json(args.output / "publication.json", publication)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"output": str(args.output), "instantaneous_metric": instantaneous,
                      "full_task_grade": grade["gate_status"], "qualification_input": False}), flush=True)


if __name__ == "__main__":
    main()

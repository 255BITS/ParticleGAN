"""One preregistered CUDA smoke and eligible own-checkpoint hold; no seed sweep."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_audit.api_run import render_gif
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sources import inspect_source
from experiments.forge.views import grade_result, load_tasks, task_fingerprint
from experiments.forge.word_adapter import run_word


@reproducible_execution
def execute(output, *, device):
    output.mkdir(parents=True, exist_ok=False)
    protocol = read_json(Path(__file__).with_name("protocol.json"))
    candidate = read_json(ROOT / "configs/forge/configurations" / (protocol["candidate"] + ".json"))
    tasks = {name: task for name, task in load_tasks(ROOT).items() if name in protocol["tasks"]}
    extra = [str(Path(__file__).relative_to(ROOT)), "reports/forge/five-word-tier-split/protocol.json"]
    source = inspect_source(ROOT, extra_paths=extra)
    atomic_json(output / "source-manifest.json", source)
    atomic_json(output / "protocol.json", protocol)
    request = dict(candidate=candidate, candidate_revision=stable_hash(candidate), protocol=dict(seed=0), tasks=tasks,
        jobs=[dict(task_id=name, compatibility_key=stable_hash(dict(task=task_fingerprint(task),
              source=source["digest"], candidate=candidate, protocol=protocol))) for name, task in tasks.items()])
    atomic_json(output / "request.json", request)
    prerequisites, rows = {}, []
    began = time.monotonic()
    for name in protocol["tasks"]:
        if name.endswith("hold") and not prerequisites:
            rows.append(dict(task=name, gate_status="BLOCKED", reason="own smoke did not pass", new_optimizer_updates=0))
            break
        task = tasks[name]
        raw, records = run_word(request, task, output / name, device,
            prerequisites=prerequisites, capture_media=True)
        result = {**raw, **grade_result(task, raw)}
        atomic_json(output / name / "result.json", result)
        case = dict(id=name, legacy_ids=["source-family-15"], goal=task["description"],
            default_steps=task["execution"]["steps"], eval_samples=1024,
            sampling=dict(evaluation=raw["evidence"]["sampling_law"]), thresholds=task["evaluation"]["thresholds"])
        render_gif(case, records, output / name / "goal.gif", full_budget=True,
                   requested_steps=task["execution"]["steps"], final_verdict=result["gate_status"])
        rows.append(dict(task=name, gate_status=result["gate_status"], evaluator_result=result.get("evaluator_result"),
            final_metrics=deepcopy(raw["evidence"]["live"]), checkpoint=raw["evidence"].get("checkpoint"),
            continuity=raw["evidence"].get("continuity"), host=raw["evidence"]["host"], guards=raw["evidence"]["guards"],
            recipe=raw["recipe"], prior=raw["prior"], rng_manifest_sha256=stable_hash(raw["rng"]),
            task_fingerprint=task_fingerprint(task), new_optimizer_updates=raw["cost"]["new_optimizer_updates"],
            completed_steps=raw["cost"]["completed_steps"], adapter_loop_seconds=raw["cost"]["adapter_loop_seconds"],
            gif_path=name + "/goal.gif", gif_sha256=file_hash(output / name / "goal.gif"),
            raw_receipt_sha256=file_hash(output / name / "adapter-receipt.json")))
        if result["gate_status"] == "PASS":
            prerequisites[name] = dict(candidate_revision=request["candidate_revision"],
                compatibility_key=next(job["compatibility_key"] for job in request["jobs"] if job["task_id"] == name), result=result)
    atomic_json(output / "readout.json", dict(schema_version=1, id=protocol["id"], qualification_input=False,
        scope=protocol["scope"], source_commit=source["origin_commit"], source_digest=source["digest"],
        protocol_sha256=file_hash(Path(__file__).with_name("protocol.json")), candidate_revision=request["candidate_revision"],
        seed=0, device=device, runtime=dict(torch=str(torch.__version__), autograd_multithreading=torch.autograd.is_multithreading_enabled()),
        rows=rows, new_optimizer_updates=sum(row["new_optimizer_updates"] for row in rows),
        total_wall_seconds=time.monotonic() - began, reserved_seconds=protocol["maximum_reserved_seconds"],
        original_sustained_task_unchanged=True))
    atomic_json(output.parent / (output.name + "-archive-manifest.json"), manifest_artifacts(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    execute(args.output, device=args.device)


if __name__ == "__main__":
    main()

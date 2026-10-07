"""Reusable CUDA execution for explicitly declared Gaussian architecture studies."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.gaussian_tasks import run_gaussian, STABILITY_KIND
from experiments.forge.sources import inspect_source, runtime_manifest
from .reproducibility import reproducible_execution

ROOT = Path(__file__).resolve().parents[2]


def stability_task(smoke):
    """Explicit materialized continuation variant, retaining all task conditions."""
    value = deepcopy(smoke)
    value["id"] = smoke["id"].replace("smoke", "stability")
    if value["id"] == smoke["id"]:
        value["id"] += "_stability"
    value["execution"].update(steps=6000, original_schedule_horizon=1000, preserve_prefix_steps=1000,
                              continuation_of=smoke["id"], produces_state=True)
    value["evaluation"].pop("confirmation", None)
    value["evaluation"].update(kind=STABILITY_KIND, stationary_end=4000, shift_end=6000, shift_mean=3.,
                              reacquisition_deadline=5000, minimum_stable_checks=5, stationary_checks=72,
                              shift_checks=48, shift_hold_checks=24)
    value["resources"]["timeout_seconds"] = 600
    value["dependencies"] = [{"task": smoke["id"], "kind": "checkpoint"}]
    value["description"] = "Continue this candidate's own Gaussian smoke state: stationary hold, shift and deadline reacquisition."
    return value


@reproducible_execution
def execute(task_path, candidate_path, output, *, device, through_stability=False, stability_path=None, resume_existing=False):
    import torch
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Gaussian architecture study requires CUDA; no CPU fallback")
    task_path, candidate_path, output = Path(task_path), Path(candidate_path), Path(output)
    smoke = json.loads(task_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    continuation = (json.loads(Path(stability_path).read_text()) if stability_path else stability_task(smoke))
    source = inspect_source(ROOT, extra_paths=tuple(str(path.resolve().relative_to(ROOT)) for path in
                            (task_path, candidate_path) + ((Path(stability_path),) if stability_path else ())))
    revision = stable_hash(candidate)
    smoke_key = stable_hash(dict(task=smoke, candidate=candidate, source=source["digest"]))
    request = dict(candidate=candidate, candidate_revision=revision, protocol={"seed": 0},
                   tasks={smoke["id"]: smoke, continuation["id"]: continuation},
                   jobs=[dict(task_id=smoke["id"], compatibility_key=smoke_key)])
    if resume_existing:
        from experiments.forge.artifacts import verify_artifacts
        archived_request = json.loads((output / "request.json").read_text())
        if archived_request["candidate"] != candidate or archived_request["tasks"][smoke["id"]] != smoke:
            raise ValueError("continuation task/candidate differs from the saved prefix")
        request = archived_request
        first = json.loads((output / "smoke" / "adapter-receipt.json").read_text())
        verify_artifacts(first["evidence"]["artifact_root"], first["evidence"]["artifact_manifest"])
        atomic_json(output / "continuation-source.json", source)
        source = {**source, "prefix_source": json.loads((output / "source.json").read_text()),
                  "continued_existing_prefix": True}
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "source.json", source)
        atomic_json(output / "request.json", request)
        first = run_gaussian(request, smoke, output / "smoke", device, diagnostic=True)
    first["gate_status"] = first["gaussian_grade"]["gate_status"]
    result = dict(schema_version=1, qualification_input=False, scope="architecture_diagnostic",
                  runtime=runtime_manifest(), source=source, task_sha256=file_hash(task_path),
                  candidate_sha256=file_hash(candidate_path), smoke=first, stability=None)
    if through_stability:
        # Diagnostic continuation remains explicit even after a smoke failure.
        # It cannot be used as ordinary qualification because diagnostic=True.
        parent = dict(candidate_revision=revision, compatibility_key=smoke_key,
                      result=first)
        second = run_gaussian(request, continuation, output / "stability", device,
                              prerequisites={smoke["id"]: parent}, diagnostic=True)
        second["gate_status"] = second["gaussian_grade"]["gate_status"]
        second["diagnostic_parent_smoke_status"] = first["gate_status"]
        result["stability"] = second
    atomic_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--through-stability", action="store_true")
    parser.add_argument("--stability-task", type=Path)
    parser.add_argument("--resume-existing", action="store_true",
                        help="Continue an explicitly amended saved prefix; never rerun its updates.")
    args = parser.parse_args()
    execute(args.task, args.candidate, args.output, device=args.device,
            through_stability=args.through_stability, stability_path=args.stability_task, resume_existing=args.resume_existing)


if __name__ == "__main__":
    main()

"""Resume three selected GPU prior runs under a new, bounded duration protocol."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.sources import inspect_source, runtime_manifest
from experiments.forge.state import (state_digest, require_same_formulation,
                                      require_optimizer_steps)
from experiments.forge.vectorprofiles import build_vector_models
from .api_vectors import _bounds
from .reproducibility import reproducible_execution
from . import tier1_prior_smoke as parent

ROOT = parent.ROOT
PROTOCOL = ROOT / "reports/forge/tier1-prior-duration/protocol.json"


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    if file_hash(ROOT / protocol["parent_results"]) != protocol["parent_results_sha256"]:
        raise ValueError("parent publication changed")
    parent.declaration()
    return protocol


def build(row, device):
    original = parent.declaration()
    task = json.loads((ROOT / original["tasks"][row["task"]]["path"]).read_text())
    condition = original["arms"][row["arm"]]
    task["execution"]["prior"] = condition["prior"]
    task["execution"]["host_definition"]["particles"] = condition["particles"]
    candidate = json.loads((ROOT / original["candidate_path"]).read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    g, d = build_vector_models(context, task["execution"]["host_definition"])
    trainer = context.build_trainer(g, d, max_steps=row["parent_updates"])
    assert all(p.device.type == "cuda" for model in (g, d, trainer.prior) for p in model.parameters())
    return context, trainer, task


def extend_preserving_state(trainer, total):
    before = trainer.state_dict()
    trainer.extend_execution(total)
    after = trainer.state_dict()
    assert after["max_steps"] == total
    before.pop("max_steps", None)
    after.pop("max_steps", None)
    if state_digest(before) != state_digest(after):
        raise RuntimeError("execution extension changed training state")


@reproducible_execution
def trial(index, raw, output, *, device):
    if torch.device(device).type != "cuda":
        raise ValueError("duration study requires CUDA")
    protocol = declaration()
    row = protocol["runs"][index]
    directory = raw / row["arm"] / row["task"]
    if (file_hash(directory / "state.pt") != row["parent_checkpoint_sha256"] or
            file_hash(directory / "receipt.json") != row["parent_receipt_sha256"]):
        raise ValueError("parent checkpoint/receipt changed")
    # Every originally pinned implementation file remains unchanged. The new
    # caller adds duration handling without replacing any training mechanism.
    source = json.loads((directory / "source.json").read_text())
    for name, digest in source["files"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("parent scientific implementation changed: " + name)
    output.mkdir(parents=True, exist_ok=False)
    context, trainer, task = build(row, device)
    if stable_hash(context.recipe.to_dict()) != row["parent_recipe_sha256"]:
        raise ValueError("resolved recipe differs from parent")
    state = torch.load(directory / "state.pt", weights_only=True, map_location="cpu")
    context.load_state_dict(state)
    restored = context.state_dict()
    assert state_digest(state) == state_digest(restored)
    require_optimizer_steps(restored, row["parent_updates"])
    extend_preserving_state(trainer, row["max_total_updates"])
    extended = context.state_dict()
    require_same_formulation(state, extended)
    atomic_json(output / "resume-proof.json", dict(
        parent_checkpoint_sha256=row["parent_checkpoint_sha256"],
        restored_state_sha256=state_digest(restored), parent_state_sha256=state_digest(state),
        state_restored_exactly=True, only_execution_cap_changed=True,
        original_cap=row["parent_updates"], new_cap=row["max_total_updates"]))
    atomic_json(output / "source.json", inspect_source(ROOT, extra_paths=(str(PROTOCOL.relative_to(ROOT)),)))
    observations = torch.load(directory / "observations.pt", weights_only=True)
    curve = json.loads((directory / "curve.json").read_text())
    target, score = parent.scorer(row["task"])
    spec = task["execution"]["host_definition"]
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    evaluation = context.streams.generator("eval", component="live", purpose="samples")
    block = row["parent_updates"]
    checks = {offset + math.ceil(i * block / 24)
              for offset in range(block, row["max_total_updates"], block) for i in range(1, 25)}
    new_data = hashlib.sha256()
    started = time.monotonic()
    for step in range(block + 1, row["max_total_updates"] + 1):
        real = target(spec, 128, data, step - 1)
        new_data.update(real.numpy().tobytes())
        trainer.step(real.to(device))
        if step in checks:
            before = context.streams.audit()
            samples = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
            metrics = score(samples, spec, step)
            full_failures = _bounds(metrics, task["evaluation"]["thresholds"])
            smoke_failures = _bounds(metrics, parent.declaration()["smoke_projection"][row["task"]])
            after = context.streams.audit()
            allowed = [key for key, value in context.streams.manifest()["bindings"].items() if value["family"] == "eval"]
            assert not context.streams.compare(before, after, allowed=allowed)["unintended_rng_deviations"]
            point = dict(step=step, metrics=metrics, full_pass=not full_failures,
                         smoke_pass=not smoke_failures, full_failed_bounds=full_failures,
                         smoke_failed_bounds=smoke_failures)
            curve.append(point)
            observations.append(dict(step=step, samples=samples, metrics=metrics))
            print(json.dumps(dict(event="observation", arm=row["arm"], task=row["task"], **point)), flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    final = context.state_dict()
    require_same_formulation(state, final)
    require_optimizer_steps(final, row["max_total_updates"])
    torch.save(final, output / "state.pt")
    torch.save(observations, output / "observations.pt")
    atomic_json(output / "curve.json", curve)
    result = dict(schema_version=1, scope=protocol["scope"], qualification_input=False,
                  arm=row["arm"], task=row["task"], parent_updates=block,
                  completed_updates=trainer.completed_steps, additional_updates=trainer.completed_steps-block,
                  original_budget_verdict=row["parent_verdict"], protocol_sha256=file_hash(PROTOCOL),
                  parent_recipe_sha256=row["parent_recipe_sha256"],
                  parent_receipt_sha256=row["parent_receipt_sha256"], prior=context.prior_config,
                  candidate_id=parent.declaration()["candidate_id"], recipe=context.recipe.to_dict(),
                  runtime=runtime_manifest(), device=str(device), gpu=torch.cuda.get_device_name(device),
                  continuation_loop_seconds=elapsed, new_data_sequence_sha256=new_data.hexdigest(),
                  full_terminal_suffix=parent.suffix(curve, "full_pass"),
                  smoke_terminal_suffix=parent.suffix(curve, "smoke_pass"),
                  full_verdict="PASS" if parent.suffix(curve, "full_pass") >= 5 else "FAIL",
                  smoke_verdict="PASS" if parent.suffix(curve, "smoke_pass") >= 5 else "FAIL",
                  final_metrics=curve[-1]["metrics"], terminal_observations=curve[-5:],
                  passing_checks=sum(point["full_pass"] for point in curve), total_checks=len(curve),
                  artifacts={name:file_hash(output / name) for name in ("state.pt", "observations.pt", "curve.json", "source.json", "resume-proof.json")})
    atomic_json(output / "receipt.json", result)
    print(json.dumps(dict(event="completed", arm=row["arm"], task=row["task"],
                         full=result["full_verdict"], suffix=result["full_terminal_suffix"], seconds=elapsed)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "trial"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw", type=Path, default=ROOT / "runs/api/tier1-prior-smoke-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/api/tier1-prior-duration-v1")
    args = parser.parse_args()
    if torch.device(args.device).type != "cuda":
        parser.error("GPU execution is required")
    if args.action == "trial":
        trial(args.index, args.raw, args.output, device=args.device)
        return 0
    protocol = declaration()
    args.output.mkdir(parents=True, exist_ok=False)
    errors = []
    for index, row in enumerate(protocol["runs"]):
        name = row["arm"] + "-" + row["task"]
        command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.tier1_prior_duration", "trial",
                   "--index", str(index), "--device", args.device, "--raw", str(args.raw), "--output", str(args.output / name)]
        print(json.dumps(dict(event="start", case=name, log=str(args.output / (name + ".log")))), flush=True)
        with (args.output / (name + ".log")).open("w") as stream:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=row["timeout_seconds"])
                if result.returncode:
                    errors.append(dict(case=name, returncode=result.returncode))
            except subprocess.TimeoutExpired:
                errors.append(dict(case=name, status="TIMEOUT"))
        print(json.dumps(dict(event="finish", case=name, errors=len(errors))), flush=True)
    atomic_json(args.output / "completion.json", dict(errors=errors, retries=0, continuations=3))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

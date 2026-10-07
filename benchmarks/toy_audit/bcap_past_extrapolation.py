"""Frozen CUDA investigation of joint BCAP updates and past extrapolation.

Only GANTrainer performs training. Archived alternating baselines retain their
source and zero new cost. This diagnostic does not fill ordinary Forge cells.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.sources import inspect_source, runtime_manifest
from experiments.forge.state import state_digest, require_optimizer_steps
from experiments.forge.vectorprofiles import build_vector_models
from .api_vectors import _bounds
from .reproducibility import reproducible_execution
from .tier1_prior_duration import extend_preserving_state
from .tier1_prior_smoke import scorer, suffix

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/bcap-past-extrapolation/protocol.json"


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in {**protocol["inputs"], **protocol["scientific_implementation"]}.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("frozen study input changed: " + name)
    return protocol


def checkpoints(task_id, phase, protocol):
    start = 0 if phase == "stationary" else protocol["adaptation"]["shift_at"]
    stop = protocol["stationary_steps"] if phase == "stationary" else start + protocol["adaptation"]["steps"]
    block = protocol["tasks"][task_id]["cadence_block_steps"]
    return sorted({offset + math.ceil(i * block / 24) for offset in range(start, stop, block)
                   for i in range(1, 25)})


def summarize(rows, task_id, phase, protocol):
    if [r["step"] for r in rows] != checkpoints(task_id, phase, protocol):
        raise ValueError("missing, duplicate or off-cadence checks")
    acquire = (protocol["tasks"][task_id]["acquisition_steps"] if phase == "stationary" else
               protocol["adaptation"]["shift_at"] + protocol["adaptation"]["reacquisition_steps"])
    acq, hold = [r for r in rows if r["step"] <= acquire], [r for r in rows if r["step"] > acquire]
    current = longest = 0
    first = None
    for row in rows:
        current = current + 1 if row["full_pass"] else 0
        longest = max(longest, current)
        if current >= 5 and first is None:
            first = row["step"]
    acquired, retained = suffix(acq, "full_pass") >= 5, all(r["full_pass"] for r in hold)
    return dict(acquisition_verdict="PASS" if acquired else "FAIL", acquisition_suffix=suffix(acq, "full_pass"),
                hold_verdict="PASS" if retained else "FAIL", hold_pass_checks=sum(r["full_pass"] for r in hold),
                hold_total_checks=len(hold), combined_verdict="PASS" if acquired and retained else "FAIL",
                final_terminal_suffix=current, first_five_pass_window=first, longest_pass_streak=longest,
                total_pass_checks=sum(r["full_pass"] for r in rows), total_checks=len(rows),
                acquisition_metrics=acq[-1]["metrics"], final_metrics=rows[-1]["metrics"])


def build(arm, task_id, device, *, cap=4000):
    protocol = declaration()
    task = json.loads((ROOT / protocol["tasks"][task_id]["path"]).read_text())
    task["execution"]["prior"] = deepcopy(protocol["prior"])
    task["execution"]["original_schedule_horizon"] = protocol["tasks"][task_id]["original_schedule_horizon"]
    candidate = json.loads((ROOT / protocol["candidates"][arm]).read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    g, d = build_vector_models(context, task["execution"]["host_definition"])
    trainer = context.build_trainer(g, d, max_steps=cap)
    context.streams.generator("data", component="target", purpose="training", device="cpu")
    context.streams.generator("eval", component="live", purpose="samples")
    return context, trainer, task


def initial_proof(context, task_id, protocol):
    path = ROOT / protocol["tasks"][task_id]["baseline_initial"]
    saved, current = torch.load(path, weights_only=True, map_location="cpu"), context.state_dict()
    strip = lambda recipe: {k: v for k, v in recipe.items() if k != "game_update"}
    if strip(current["recipe"]) != strip(saved["recipe"]):
        raise ValueError("recipe changed beyond game_update")
    for key in ("models", "optimizers", "initial_lrs", "streams"):
        if state_digest(current["trainer"][key]) != state_digest(saved["trainer"][key]):
            raise ValueError("initial trainer differs: " + key)
    for key in ("initialization", "prior", "streams"):
        if state_digest(current[key]) != state_digest(saved[key]):
            raise ValueError("initial context differs: " + key)
    return dict(matched=True, baseline_sha256=file_hash(path), allowed_delta="game_update only",
                model_hashes={k: state_digest(v) for k, v in current["trainer"]["models"].items()})


def reuse(task_id, raw, protocol):
    parent = ROOT / protocol["tasks"][task_id]["baseline"]
    dest = raw / ("alternating-" + task_id + "-stationary")
    dest.mkdir(exist_ok=False)
    for name in ("initial-state.pt", "state.pt", "observations.pt", "curve.json", "source.json"):
        shutil.copy2(parent / name, dest / name)
    receipt = json.loads((parent / "receipt.json").read_text())
    rows = json.loads((dest / "curve.json").read_text())
    grade = summarize(rows, task_id, "stationary", protocol)
    for key, value in grade.items():
        if receipt[key] != value:
            raise ValueError("archived grade differs: " + key)
    result = dict(arm="alternating", task=task_id, phase="stationary", mode="reuse_complete",
                  completed_updates=4000, additional_updates=0, loop_seconds=0., recipe=receipt["recipe"],
                  baseline_receipt_sha256=file_hash(parent / "receipt.json"), original_source=json.loads((parent / "source.json").read_text()),
                  original_loop_seconds=receipt["continuation_loop_seconds"], data_sha256=protocol["tasks"][task_id]["data_sha256"],
                  **grade, artifacts={name: file_hash(dest / name) for name in
                      ("initial-state.pt", "state.pt", "observations.pt", "curve.json", "source.json")})
    atomic_json(dest / "receipt.json", result)
    return result


@reproducible_execution
def trial(arm, task_id, phase, raw, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("study requires CUDA; no CPU fallback")
    protocol = declaration()
    directory = raw / f"{arm}-{task_id}-{phase}"
    directory.mkdir(exist_ok=False)
    context, trainer, task = build(arm, task_id, device)
    proof = initial_proof(context, task_id, protocol)
    target, score = scorer(task_id)
    spec, thresholds = task["execution"]["host_definition"], task["evaluation"]["thresholds"]
    frozen = None
    if phase == "shift":
        parent = raw / f"{arm}-{task_id}-stationary" / "state.pt"
        saved = torch.load(parent, weights_only=True, map_location="cpu")
        context.load_state_dict(saved)
        if state_digest(context.state_dict()) != state_digest(saved):
            raise ValueError("checkpoint restore is not exact")
        frozen_context, frozen, _ = build(arm, task_id, device)
        frozen_context.load_state_dict(saved)
        proof.update(parent_checkpoint_sha256=file_hash(parent), restored_exactly=True, cached_history_reset=False)
        extend_preserving_state(trainer, 6000)
        spec["means"] = [[protocol["adaptation"]["mean_after"]]]
    atomic_json(directory / "source.json", inspect_source(ROOT, extra_paths=(str(PROTOCOL.relative_to(ROOT)), protocol["candidates"][arm])))
    torch.save(context.state_dict(), directory / "initial-state.pt")
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    evaluation = context.streams.generator("eval", component="live", purpose="samples")
    rows, snapshots, frozen_rows = [], [], []

    def observe(step):
        before = context.streams.audit()
        points = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
        allowed = [k for k, v in context.streams.manifest()["bindings"].items() if v["family"] == "eval"]
        if context.streams.compare(before, context.streams.audit(), allowed=allowed)["unintended_rng_deviations"]:
            raise ValueError("evaluation consumed a training stream")
        metrics = score(points, spec, step)
        failed = _bounds(metrics, thresholds)
        row = dict(step=step, metrics=metrics, full_pass=not failed, full_failed_bounds=failed)
        snapshot = dict(step=step, samples=points, metrics=metrics)
        if frozen is not None:
            # Exactly the same component-index/kernel draws as the active model.
            stream = frozen_context.streams.generator("eval", component="live", purpose="samples")
            frozen_points = frozen.sample(4096, generator=stream, output_noise=False).detach().cpu()
            frozen_metrics = score(frozen_points, spec, step)
            frozen_row = dict(step=step, metrics=frozen_metrics, full_pass=not _bounds(frozen_metrics, thresholds))
            if state_digest(stream.get_state()) != state_digest(evaluation.get_state()):
                raise ValueError("active/frozen evaluation streams differ")
            frozen_rows.append(frozen_row)
            snapshot.update(frozen_samples=frozen_points, frozen_metrics=frozen_metrics)
        snapshots.append(snapshot)
        print(json.dumps(dict(event="observation", arm=arm, task=task_id, phase=phase, **row)), flush=True)
        return row

    with context.streams.preserve():
        if frozen is None:
            observe(0)
        else:
            with frozen_context.streams.preserve():
                observe(4000)
    frozen_rows.clear()
    start = trainer.completed_steps
    stop = 4000 if phase == "stationary" else 6000
    checks = set(checkpoints(task_id, phase, protocol))
    digest = hashlib.sha256()
    timeout = protocol["budget"]["per_stationary_trial_seconds" if phase == "stationary" else "per_shift_trial_seconds"]
    began = time.monotonic()
    for step in range(start + 1, stop + 1):
        real = target(spec, 128, data, step - 1)
        digest.update(real.numpy().tobytes())
        trainer.step(real.to(device))
        if step in checks:
            rows.append(observe(step))
        if time.monotonic() - began > timeout:
            raise TimeoutError("frozen trial allowance exceeded")
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - began
    if phase == "stationary" and digest.hexdigest() != protocol["tasks"][task_id]["data_sha256"]:
        raise ValueError("real batch sequence differs from baseline")
    state = context.state_dict()
    require_optimizer_steps(state, stop)
    torch.save(state, directory / "state.pt")
    if frozen is not None:
        torch.save(frozen_context.state_dict(), directory / "frozen-state.pt")
        atomic_json(directory / "frozen-curve.json", frozen_rows)
        if frozen.completed_steps != 4000:
            raise ValueError("frozen control trained")
    torch.save(snapshots, directory / "observations.pt")
    atomic_json(directory / "curve.json", rows)
    result = dict(schema_version=1, scope=protocol["scope"], qualification_input=False,
                  arm=arm, task=task_id, phase=phase, mode="fresh" if start == 0 else "resume",
                  completed_updates=stop, additional_updates=stop-start, loop_seconds=elapsed,
                  protocol_sha256=file_hash(PROTOCOL), candidate_path=protocol["candidates"][arm],
                  recipe=trainer.recipe.to_dict(), prior=protocol["prior"], initial_proof=proof,
                  device=device, gpu=torch.cuda.get_device_name(device), runtime=runtime_manifest(),
                  data_sha256=digest.hexdigest(), final_rng=context.streams.audit(), **summarize(rows, task_id, phase, protocol),
                  frozen_summary=None if frozen is None else summarize(frozen_rows, task_id, phase, protocol),
                  artifacts={name: file_hash(directory / name) for name in
                             ("initial-state.pt", "state.pt", "observations.pt", "curve.json", "source.json")})
    if frozen is not None:
        result["artifacts"].update({name: file_hash(directory / name) for name in ("frozen-state.pt", "frozen-curve.json")})
    atomic_json(directory / "receipt.json", result)
    print(json.dumps(dict(event="complete", arm=arm, task=task_id, phase=phase, grade=result["combined_verdict"], seconds=elapsed)), flush=True)
    return result


def execute(raw, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("study requires CUDA; no CPU fallback")
    protocol = declaration()
    raw.mkdir(parents=True, exist_ok=False)
    results = [reuse(task_id, raw, protocol) for task_id in protocol["tasks"]]
    plan = [(arm, task, "stationary") for arm in ("simultaneous", "extrapolation_from_past") for task in protocol["tasks"]]
    plan += [(arm, protocol["adaptation"]["task"], "shift") for arm in protocol["candidates"]]
    for arm, task, phase in plan:
        name = f"{arm}-{task}-{phase}"
        log = raw / (name + ".log")
        print(json.dumps(dict(event="start", arm=arm, task=task, phase=phase, log=str(log))), flush=True)
        timeout = protocol["budget"]["per_stationary_trial_seconds" if phase == "stationary" else "per_shift_trial_seconds"]
        command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.bcap_past_extrapolation", "trial",
                   "--arm", arm, "--task", task, "--phase", phase, "--output", str(raw), "--device", device]
        with log.open("w") as stdout:
            finished = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, timeout=timeout + 60)
        if finished.returncode:
            raise RuntimeError("trial failed; no retry: " + str(log))
        result = json.loads((raw / name / "receipt.json").read_text())
        results.append(result)
        print(json.dumps(dict(event="completed", arm=arm, task=task, phase=phase,
                             acquisition=result["acquisition_verdict"], hold=result["hold_verdict"], seconds=result["loop_seconds"])), flush=True)
    if sum(r["additional_updates"] for r in results) != protocol["budget"]["new_training_updates"]:
        raise ValueError("study update accounting differs")
    shifted = [r["data_sha256"] for r in results if r["phase"] == "shift"]
    if len(set(shifted)) != 1:
        raise ValueError("shift real batch sequences differ")
    atomic_json(raw / "results.json", dict(schema_version=1, protocol_sha256=file_hash(PROTOCOL), results=results,
        new_training_updates=sum(r["additional_updates"] for r in results), new_training_loop_seconds=sum(r["loop_seconds"] for r in results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "trial"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", choices=("alternating", "simultaneous", "extrapolation_from_past"))
    parser.add_argument("--task", choices=("gaussian1d_acquisition", "ring16_acquisition"))
    parser.add_argument("--phase", choices=("stationary", "shift"))
    args = parser.parse_args()
    if args.mode == "trial":
        trial(args.arm, args.task, args.phase, args.output, device=args.device)
    else:
        execute(args.output, device=args.device)


if __name__ == "__main__":
    main()

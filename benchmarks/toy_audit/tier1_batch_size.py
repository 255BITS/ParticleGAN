"""Bounded GPU batch comparison and stationary hold through the public API."""
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
from experiments.forge.state import state_digest, require_optimizer_steps
from experiments.forge.vectorprofiles import build_vector_models
from .api_vectors import _bounds
from .reproducibility import reproducible_execution
from .tier1_prior_duration import extend_preserving_state
from .tier1_prior_smoke import scorer, suffix

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/tier1-batch-size/protocol.json"


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in {**protocol["inputs"], **protocol["scientific_implementation"]}.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"frozen evidence/implementation changed: {path}")
    return protocol


def checkpoints(task_id, protocol):
    block = protocol["tasks"][task_id]["cadence_block_steps"]
    return sorted({offset + math.ceil(i * block / 24)
                   for offset in range(0, protocol["total_steps"], block)
                   for i in range(1, 25) if offset + math.ceil(i * block / 24) <= protocol["total_steps"]})


def build(task_id, batch, device, *, cap=None):
    protocol = declaration()
    if batch not in protocol["batches"]:
        raise ValueError("undeclared batch")
    binding = protocol["tasks"][task_id]
    task = json.loads((ROOT / binding["path"]).read_text())
    task["execution"]["prior"] = deepcopy(protocol["prior"])
    task["execution"]["host_definition"]["batch"] = batch
    task["execution"]["original_schedule_horizon"] = binding["original_schedule_horizon"]
    candidate = json.loads((ROOT / protocol["candidate_path"]).read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    g, d = build_vector_models(context, task["execution"]["host_definition"])
    trainer = context.build_trainer(g, d, max_steps=cap or protocol["total_steps"])
    context.streams.generator("data", component="target", purpose="training", device="cpu")
    context.streams.generator("eval", component="live", purpose="samples")
    return context, trainer, task


def grouped_real(target, spec, batch, rng, completed):
    """Same sequential 128-example target stream, different batch grouping."""
    if batch not in (128, 512):
        raise ValueError("undeclared batch")
    blocks = [target(spec, 128, rng, completed) for _ in range(batch // 128)]
    return torch.cat(blocks, dim=0), blocks


def summarize(rows, task_id, protocol):
    if [r["step"] for r in rows] != checkpoints(task_id, protocol):
        raise ValueError("missing, duplicate or off-cadence observations")
    acquire = protocol["tasks"][task_id]["acquisition_steps"]
    acquisition = [r for r in rows if r["step"] <= acquire]
    hold = [r for r in rows if r["step"] > acquire]
    longest = current = 0
    first_window = None
    for row in rows:
        current = current + 1 if row["full_pass"] else 0
        longest = max(longest, current)
        if current >= 5 and first_window is None:
            first_window = row["step"]
    acquired = suffix(acquisition, "full_pass") >= 5
    retained = all(r["full_pass"] for r in hold)
    cuts = sorted({acquire, protocol["total_steps"], 1000, 2000, 3000} |
                  {pair[1] for pair in protocol["secondary_equal_example_cuts"][task_id]})
    return dict(acquisition_verdict="PASS" if acquired else "FAIL",
                acquisition_suffix=suffix(acquisition, "full_pass"),
                acquisition_metrics=acquisition[-1]["metrics"],
                hold_verdict="PASS" if retained else "FAIL",
                hold_pass_checks=sum(r["full_pass"] for r in hold), hold_total_checks=len(hold),
                combined_verdict="PASS" if acquired and retained else "FAIL",
                final_terminal_suffix=suffix(rows, "full_pass"), final_metrics=rows[-1]["metrics"],
                first_five_pass_window=first_window, longest_pass_streak=longest,
                total_pass_checks=sum(r["full_pass"] for r in rows), total_checks=len(rows),
                cuts=[dict(step=step, metrics=next(r["metrics"] for r in rows if r["step"] == step),
                           terminal_suffix=suffix([r for r in rows if r["step"] <= step], "full_pass"))
                      for step in cuts])


def verify_initial(context, binding):
    saved = torch.load(ROOT / binding["baseline_prefix"] / "initial-state.pt", weights_only=True, map_location="cpu")
    current = context.state_dict()
    if {k: v for k, v in current["recipe"].items() if k != "batch_size"} != {
            k: v for k, v in saved["recipe"].items() if k != "batch_size"}:
        raise ValueError("recipe changed beyond task-owned batch")
    for key in ("models", "optimizers", "initial_lrs", "streams"):
        if state_digest(current["trainer"][key]) != state_digest(saved["trainer"][key]):
            raise ValueError("initial trainer mismatch: " + key)
    for key in ("initialization", "streams", "prior"):
        if state_digest(current[key]) != state_digest(saved[key]):
            raise ValueError("initial context mismatch: " + key)
    return dict(matched=True, baseline_initial_file_sha256=file_hash(ROOT / binding["baseline_prefix"] / "initial-state.pt"),
                model_hashes={k: state_digest(v) for k, v in current["trainer"]["models"].items()},
                allowed_recipe_delta="batch_size only; external cap declared separately",
                compared="all models, optimizer histories/rates, initialization, prior and named streams; ambient caller RNG not compared")


@reproducible_execution
def data_audit(output, *, device):
    """Reconstruct only real examples; no neural draws/updates or new control run."""
    protocol = declaration()
    result, rng_states = {}, {}
    for task_id, binding in protocol["tasks"].items():
        context, _, task = build(task_id, 128, device)
        target, _ = scorer(task_id)
        stream = context.streams.generator("data", component="target", purpose="training", device="cpu")
        digests = [hashlib.sha256() for _ in binding["data_segments"]]
        complete = hashlib.sha256()
        for index in range(4000):
            block = target(task["execution"]["host_definition"], 128, stream, index)
            data = block.numpy().tobytes()
            complete.update(data)
            for h, segment in zip(digests, binding["data_segments"]):
                if segment["start"] <= index < segment["end"]:
                    h.update(data)
            if index + 1 == binding["baseline_total_steps"]:
                baseline = torch.load(ROOT / binding["baseline_duration"] / "state.pt", weights_only=True, map_location="cpu")
                if not torch.equal(stream.get_state(), baseline["trainer"]["streams"]["data_generator"] if "data_generator" in baseline["trainer"]["streams"] else
                                   next(value for key, value in baseline["streams"]["states"].items() if json.loads(key)[:3] == ["data", "target", "training"])):
                    raise ValueError("reconstructed target stream differs from saved baseline")
        for h, segment in zip(digests, binding["data_segments"]):
            if h.hexdigest() != segment["sha256"]:
                raise ValueError("reconstructed target bytes differ from baseline")
        result[task_id] = dict(original_segments_exact=True, first_4000_microbatches_sha256=complete.hexdigest())
        rng_states[task_id] = context.streams.state_dict()
    atomic_json(output, dict(training_updates=0, model_sampling_draws=0, tasks=result))
    torch.save(rng_states, output.with_suffix(".rng.pt"))
    print(json.dumps(dict(event="data_audit", result=result)), flush=True)
    return result


@reproducible_execution
def trial(index, raw, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("batch study requires CUDA; CPU fallback forbidden")
    protocol = declaration()
    run = protocol["run_plan"][index]
    task_id, batch = run["task"], run["batch"]
    binding = protocol["tasks"][task_id]
    directory = raw / f"b{batch}-{task_id}"
    directory.mkdir(parents=True, exist_ok=False)
    cap = binding["baseline_total_steps"] if run["mode"] == "resume" else protocol["total_steps"]
    context, trainer, task = build(task_id, batch, device, cap=cap)
    initial_proof = verify_initial(context, binding)
    target, score = scorer(task_id)
    spec = task["execution"]["host_definition"]
    thresholds = task["evaluation"]["thresholds"]
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    evaluation = context.streams.generator("eval", component="live", purpose="samples")
    atomic_json(directory / "source.json", inspect_source(ROOT, extra_paths=(str(PROTOCOL.relative_to(ROOT)), protocol["candidate_path"])))
    rows, snapshots = [], []
    prefix_receipt = None
    if run["mode"] in ("resume", "reuse_complete"):
        parent = ROOT / binding["baseline_duration"]
        saved = torch.load(parent / "state.pt", weights_only=True, map_location="cpu")
        if run["mode"] == "reuse_complete":
            # Already constructed at the saved 4k execution cap.
            pass
        context.load_state_dict(saved)
        if state_digest(context.state_dict()) != state_digest(saved):
            raise ValueError("baseline restore was not exact")
        require_optimizer_steps(saved, binding["baseline_total_steps"])
        prefix_receipt = file_hash(parent / "receipt.json")
        rows = json.loads((parent / "curve.json").read_text())
        snapshots = torch.load(parent / "observations.pt", weights_only=True)
        for row in rows:
            row["full_failed_bounds"] = _bounds(row["metrics"], thresholds)
            if row["full_pass"] != (not row["full_failed_bounds"]):
                raise ValueError("original metric conditions changed")
        if run["mode"] == "resume":
            extend_preserving_state(trainer, protocol["total_steps"])
        torch.save(context.state_dict(), directory / "initial-state.pt")
    else:
        torch.save(context.state_dict(), directory / "initial-state.pt")
        with context.streams.preserve():
            points = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
        original_initial = torch.load(ROOT / binding["baseline_prefix"] / "observations.pt", weights_only=True)[0]
        if not torch.equal(points, original_initial["samples"]):
            raise ValueError("initial served samples differ from baseline")
        snapshots.append(dict(step=0, samples=points, metrics=score(points, spec, 0)))
    start = trainer.completed_steps
    checks = set(checkpoints(task_id, protocol))
    all_data = hashlib.sha256()
    common_data = hashlib.sha256()
    microcount = start * (batch // 128)
    began = time.monotonic()
    for step in range(start + 1, protocol["total_steps"] + 1):
        real, blocks = grouped_real(target, spec, batch, data, step - 1)
        for block in blocks:
            encoded = block.numpy().tobytes()
            all_data.update(encoded)
            if microcount < 4000:
                common_data.update(encoded)
            microcount += 1
        trainer.step(real.to(device))
        if step in checks:
            before = context.streams.audit()
            points = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
            metrics = score(points, spec, step)
            after = context.streams.audit()
            allowed = [k for k, v in context.streams.manifest()["bindings"].items() if v["family"] == "eval"]
            if context.streams.compare(before, after, allowed=allowed)["unintended_rng_deviations"]:
                raise ValueError("evaluation consumed training RNG")
            failed = _bounds(metrics, thresholds)
            row = dict(step=step, full_pass=not failed, full_failed_bounds=failed, metrics=metrics)
            rows.append(row)
            snapshots.append(dict(step=step, samples=points, metrics=metrics))
            print(json.dumps(dict(event="observation", task=task_id, batch=batch, **row)), flush=True)
            if time.monotonic() - began > run["timeout_seconds"]:
                raise TimeoutError("declared trial allowance exceeded")
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - began
    assert trainer.completed_steps == 4000 and trainer.completed_steps - start == run["new_updates"]
    for model in (trainer.G, trainer.D, trainer.prior):
        assert all(p.device.type == "cuda" for p in model.parameters())
    final = context.state_dict()
    require_optimizer_steps(final, 4000)
    torch.save(final, directory / "state.pt")
    torch.save(snapshots, directory / "observations.pt")
    atomic_json(directory / "curve.json", rows)
    result = dict(schema_version=1, scope=protocol["scope"], qualification_input=False, task=task_id, batch=batch,
                  mode=run["mode"], completed_updates=4000, additional_updates=4000-start,
                  protocol_sha256=file_hash(PROTOCOL), candidate_id=protocol["candidate_path"].split("/")[-1][:-5],
                  recipe=trainer.recipe.to_dict(), prior=protocol["prior"], initial_proof=initial_proof,
                  prefix_receipt_sha256=prefix_receipt, device=device, gpu=torch.cuda.get_device_name(device),
                  runtime=runtime_manifest(), continuation_loop_seconds=elapsed,
                  total_real_training_examples=4000*batch, new_real_training_examples=(4000-start)*batch,
                  new_real_sequence_sha256=all_data.hexdigest(),
                  first_4000_microbatches_sha256=common_data.hexdigest() if start == 0 else None,
                  rng=context.streams.manifest(), final_rng_states=context.streams.audit(),
                  **summarize(rows, task_id, protocol),
                  artifacts={name: file_hash(directory / name) for name in
                             ("initial-state.pt", "state.pt", "observations.pt", "curve.json", "source.json")})
    atomic_json(directory / "receipt.json", result)
    print(json.dumps(dict(event="complete", task=task_id, batch=batch, acquisition=result["acquisition_verdict"],
                          hold=result["hold_verdict"], combined=result["combined_verdict"], seconds=elapsed)), flush=True)
    return result


def execute(raw, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("batch study requires CUDA; CPU fallback forbidden")
    protocol = declaration()
    raw.mkdir(parents=True, exist_ok=False)
    data_audit(raw / "data-audit.json", device=device)
    rows = []
    for index, planned in enumerate(protocol["run_plan"]):
        path = raw / f"b{planned['batch']}-{planned['task']}"
        log = raw / f"b{planned['batch']}-{planned['task']}.log"
        print(json.dumps(dict(event="start", **planned, log=str(log))), flush=True)
        command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.tier1_batch_size", "trial",
                   "--index", str(index), "--output", str(raw), "--device", device]
        with log.open("w") as stdout:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT,
                                    timeout=max(120, planned["timeout_seconds"] + 60))
        if result.returncode:
            raise RuntimeError(f"trial failed without retry: inspect {log}")
        rows.append(json.loads((path / "receipt.json").read_text()))
        print(json.dumps(dict(event="completed", task=planned["task"], batch=planned["batch"],
                             acquisition=rows[-1]["acquisition_verdict"], hold=rows[-1]["hold_verdict"],
                             seconds=rows[-1]["continuation_loop_seconds"])), flush=True)
    audit = json.loads((raw / "data-audit.json").read_text())
    for row in rows:
        if row["batch"] == 512:
            assert row["first_4000_microbatches_sha256"] == audit["tasks"][row["task"]]["first_4000_microbatches_sha256"]
    atomic_json(raw / "results.json", dict(schema_version=1, protocol_sha256=file_hash(PROTOCOL), results=rows,
                                          new_training_updates=sum(r["additional_updates"] for r in rows),
                                          new_training_loop_seconds=sum(r["continuation_loop_seconds"] for r in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "trial"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "trial":
        trial(args.index, args.output, device=args.device)
    else:
        execute(args.output, device=args.device)


if __name__ == "__main__":
    main()

"""Bounded magnitude-sensitive prior diagnostic; shared public-API trial host.

Archived source-bound studies stay immutable. This separately frozen diagnostic
uses the same host implementation with its own protocol and trainer delta.
"""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import torch

from . import bcap_past_extrapolation as host
from .reproducibility import reproducible_execution
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest

ROOT = host.ROOT
PROTOCOL = ROOT / "reports/forge/gaussian-prior-magnitude/protocol.json"
checkpoints, summarize, scorer = host.checkpoints, host.summarize, host.scorer


@contextmanager
def bound_runner():
    previous = host.PROTOCOL, host.initial_proof
    try:
        host.PROTOCOL, host.initial_proof = PROTOCOL, initial_proof
        yield
    finally:
        host.PROTOCOL, host.initial_proof = previous


def declaration():
    with bound_runner():
        return host.declaration()


def build(*args, **kwargs):
    with bound_runner():
        return host.build(*args, **kwargs)


def logical_devices(value):
    """Only worker ordinal differs; CUDA stream seeds exclude device ordinal."""
    if isinstance(value, str):
        return value.replace("cuda:0", "cuda").replace("cuda:1", "cuda")
    if isinstance(value, dict):
        return {logical_devices(k): logical_devices(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(logical_devices(v) for v in value)
    return value


def initial_proof(context, task_id, protocol):
    path = ROOT / protocol["tasks"][task_id]["baseline_initial"]
    saved, current = torch.load(path, weights_only=True, map_location="cpu"), context.state_dict()
    strip = lambda recipe: {k: v for k, v in recipe.items() if k not in
                           ("game_update", "prior_update", "prior_gradient_scale")}
    if strip(current["recipe"]) != strip(saved["recipe"]):
        raise ValueError("recipe changed beyond declared timing and prior update")
    for key in ("models", "initial_lrs", "streams"):
        if state_digest(current["trainer"][key]) != state_digest(saved["trainer"][key]):
            raise ValueError("initial trainer differs: " + key)
    operators = deepcopy(current["trainer"]["optimizers"])
    for optimizer in operators:
        for group in optimizer["param_groups"]:
            if group["algorithm"] == "row_capped":
                group["algorithm"] = "rownorm"
                if group.pop("row_gradient_scale") != .001:
                    raise ValueError("prior scale differs from frozen hypothesis")
    if state_digest(operators) != state_digest(saved["trainer"]["optimizers"]):
        raise ValueError("optimizer changed beyond the prior direction")
    for key in ("initialization", "prior", "streams"):
        if state_digest(logical_devices(current[key])) != state_digest(logical_devices(saved[key])):
            raise ValueError("initial context differs: " + key)
    return dict(matched=True, baseline_sha256=file_hash(path),
                allowed_delta="game_update, prior_update=row_capped, prior_gradient_scale=.001; CUDA worker ordinal",
                model_hashes={k: state_digest(v) for k, v in current["trainer"]["models"].items()})


def trial(*args, **kwargs):
    with bound_runner():
        return host.trial(*args, **kwargs)


@reproducible_execution
def initialization_probe(output, *, device):
    """Separate zero-update component cohort; scale is fixed before this probe."""
    protocol = declaration()
    rows = []
    for tid in protocol["tasks"]:
        context, trainer, task = build("simultaneous", tid, device)
        initial_proof(context, tid, protocol)
        before = context.state_dict()
        torch.save(before, output / (tid + "-initial.pt"))
        target, _ = scorer(tid)
        data = context.streams.generator("data", component="target", purpose="training", device="cpu")
        real = target(task["execution"]["host_definition"], 128, data, 0).to(device)
        # Discard the critic-side draw so the generator-side draw is identical
        # to a joint first update, without applying either optimizer.
        trainer.prior.sample(128, generator=trainer.latent_generator,
                             noise_generator=trainer.prior_noise_generator)
        latent, indices = trainer.prior.sample(128, generator=trainer.latent_generator,
                                             noise_generator=trainer.prior_noise_generator)
        trainer.D.requires_grad_(False)
        loss = trainer.loss.g_loss(trainer.D(trainer.G(latent)), trainer.D(real))
        loss.backward()
        gradient = trainer.prior.z.grad[torch.unique(indices)].detach()
        norms = gradient.norm(dim=1)
        torch.save(dict(gradient=gradient, rows=torch.unique(indices), state=context.state_dict()),
                   output / (tid + "-probe.pt"))
        if state_digest(before["trainer"]["models"]) != state_digest(context.state_dict()["trainer"]["models"]):
            raise ValueError("initialization probe changed parameters")
        rows.append(dict(task=tid, cohort="initialization_gradient_no_updates", scale=.001,
            training_updates=0, selected_rows=len(norms), gradient_norm_min=float(norms.min()),
            gradient_norm_median=float(norms.median()), gradient_norm_max=float(norms.max()),
            capped_fraction=float((norms >= .001).float().mean()),
            implied_prior_step_mean=float((norms/.001).clamp_max(1).mean())*.03,
            initial_models_unchanged=True, initial_checkpoint_sha256=file_hash(output/(tid+"-initial.pt")),
            probe_checkpoint_sha256=file_hash(output/(tid+"-probe.pt"))))
    atomic_json(output / "initialization-probe.json", dict(training_updates=0, scale_predeclared=True, results=rows))
    return rows


def execute(raw, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("study requires CUDA; no CPU fallback")
    protocol = declaration()
    raw.mkdir(parents=True, exist_ok=False)
    results = []
    plan = [(arm, task, "stationary") for arm in protocol["candidates"] for task in protocol["tasks"]]
    plan += [(arm, protocol["adaptation"]["task"], "shift") for arm in protocol["candidates"]]
    for arm, task, phase in plan:
        name = f"{arm}-{task}-{phase}"
        log = raw/(name+".log")
        print(json.dumps(dict(event="start", arm=arm, task=task, phase=phase, log=str(log))), flush=True)
        timeout = protocol["budget"]["per_stationary_trial_seconds" if phase == "stationary" else "per_shift_trial_seconds"]
        command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.gaussian_prior_magnitude", "trial",
                   "--arm", arm, "--task", task, "--phase", phase, "--output", str(raw), "--device", device]
        with log.open("w") as stdout:
            finished = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, timeout=timeout+60)
        if finished.returncode:
            raise RuntimeError("trial failed; no scientific retry: " + str(log))
        result = json.loads((raw/name/"receipt.json").read_text())
        results.append(result)
        print(json.dumps(dict(event="completed", arm=arm, task=task, phase=phase,
              acquisition=result["acquisition_verdict"], hold=result["hold_verdict"], seconds=result["loop_seconds"])), flush=True)
    if sum(r["additional_updates"] for r in results) != protocol["budget"]["new_training_updates"]:
        raise ValueError("study update accounting differs")
    if len({r["data_sha256"] for r in results if r["phase"] == "shift"}) != 1:
        raise ValueError("shift data differs")
    atomic_json(raw/"results.json", dict(schema_version=1, protocol_sha256=file_hash(PROTOCOL), results=results,
        new_training_updates=sum(r["additional_updates"] for r in results),
        new_training_loop_seconds=sum(r["loop_seconds"] for r in results)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "trial", "probe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--arm", choices=("alternating", "simultaneous", "extrapolation_from_past"))
    parser.add_argument("--task", choices=("gaussian1d_acquisition", "ring16_acquisition"))
    parser.add_argument("--phase", choices=("stationary", "shift"))
    args = parser.parse_args()
    if args.mode == "trial":
        trial(args.arm, args.task, args.phase, args.output, device=args.device)
    elif args.mode == "probe":
        args.output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(initialization_probe(args.output, device=args.device)))
    else:
        execute(args.output, device=args.device)


if __name__ == "__main__":
    main()

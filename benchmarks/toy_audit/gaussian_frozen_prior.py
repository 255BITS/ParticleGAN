"""Matched frozen-initial-prior diagnostic through public GANTrainer.

The fixed-prior cohort is explicit; learned controls retain their archived identity.
Only orchestration lives here. GANTrainer owns every scientific update.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models
from . import bcap_past_extrapolation as host
from .bcap_past_extrapolation import checkpoints, summarize
from .tier1_prior_smoke import scorer

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/gaussian-frozen-prior/protocol.json"


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in {**protocol["inputs"], **protocol["scientific_implementation"]}.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("frozen input changed: " + name)
    return protocol


def build(arm, task_id, device):
    protocol = declaration()
    task = json.loads((ROOT / protocol["tasks"][task_id]["path"]).read_text())
    task["execution"]["prior"] = deepcopy(protocol["prior"])
    task["execution"]["original_schedule_horizon"] = protocol["tasks"][task_id]["original_schedule_horizon"]
    candidate = json.loads((ROOT / protocol["candidates"][arm]).read_text())
    # These are explicitly frozen-prior task variants, rather than qualified
    # learned-location tasks. Retain all other candidate/task requirements.
    for cohort in (candidate, task):
        cohort["requires_capabilities"] = [name for name in cohort.get("requires_capabilities", [])
                                           if name != "learned_locations"]
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    g, d = build_vector_models(context, task["execution"]["host_definition"])
    prior = context.build_prior()
    # Public initializers intentionally preserve buffers. Expose precisely this
    # fresh tensor as a parameter for initialization, then freeze it before an
    # optimizer exists. No fitted tensor, RNG reset or extra constructor draw.
    locations = prior.z
    delattr(prior, "z")
    prior.register_parameter("z", torch.nn.Parameter(locations))
    context.initialize(prior, component="prior")
    locations = prior.z.detach()
    delattr(prior, "z")
    prior.register_buffer("z", locations)
    context.initialization["prior"]["frozen_control"] = {
        "law": "matched public deterministic initial locations, frozen before optimizer construction",
        "initializable_view": "temporary trainable parameter; final z is buffer",
        "learnable": False, "tensor_sha256": state_digest(locations)}
    # Public trainer construction reuses the fully initialized control prior.
    # This diagnostic factory specialization prevents a second constructor draw.
    context.build_prior = lambda *, dtype=torch.float32: prior
    trainer = context.build_trainer(g, d, max_steps=4000)
    context.streams.generator("data", component="target", purpose="training", device="cpu")
    context.streams.generator("eval", component="live", purpose="samples")
    return context, trainer, task


def initial_proof(context, task_id, protocol):
    path = ROOT / protocol["tasks"][task_id]["baseline_initial"]
    saved = torch.load(path, weights_only=True, map_location="cpu")
    current = context.state_dict()
    strip = lambda value: {key: item for key, item in value.items() if key != "game_update"}
    if strip(saved["recipe"]) != strip(current["recipe"]):
        raise ValueError("recipe changed beyond game timing")
    for key in ("models", "streams"):
        if state_digest(saved["trainer"][key]) != state_digest(current["trainer"][key]):
            raise ValueError("initial trainer differs: " + key)
    if state_digest(saved["streams"]) != state_digest(current["streams"]):
        raise ValueError("initial named streams differ")
    for component in ("generator", "discriminator"):
        if saved["initialization"][component] != current["initialization"][component]:
            raise ValueError("network initialization contract differs")
    if context.prior_config["learnable"] or context._trainer.prior.z.requires_grad:
        raise ValueError("control prior is learned")
    if any(group["role"] == "prior" for group in context._trainer.opt_g.param_groups):
        raise ValueError("frozen prior has an optimizer group")
    return dict(matched=True, baseline_sha256=file_hash(path),
        allowed_delta="prior parameter becomes frozen buffer; prior optimizer group absent; game timing explicit",
        model_hashes={key: state_digest(value) for key, value in current["trainer"]["models"].items()},
        prior_initial_sha256=state_digest(current["trainer"]["models"]["prior"]))


@contextmanager
def archived_host_binding():
    """Reuse the unchanged public-trainer orchestration under this control law.

    The binding is process-local and restored even if a trial raises. Scientific
    trials each execute in their own subprocess, so no concurrent host mutation.
    """
    original = host.PROTOCOL, host.build, host.initial_proof
    host.PROTOCOL, host.build, host.initial_proof = PROTOCOL, build, initial_proof
    try:
        yield
    finally:
        host.PROTOCOL, host.build, host.initial_proof = original


def trial(arm, task_id, phase, raw, *, device):
    with archived_host_binding():
        host.trial(arm, task_id, phase, raw, device=device)
    output = raw / f"{arm}-{task_id}-{phase}"
    initial = torch.load(output / "initial-state.pt", weights_only=True, map_location="cpu")
    final = torch.load(output / "state.pt", weights_only=True, map_location="cpu")
    if state_digest(initial["trainer"]["models"]["prior"]) != state_digest(final["trainer"]["models"]["prior"]):
        raise ValueError("frozen prior moved")
    receipt = json.loads((output / "receipt.json").read_text())
    receipt.update(cohort="frozen_initial_prior", prior_unchanged=True)
    atomic_json(output / "receipt.json", receipt)


def execute(raw, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("study requires CUDA; no CPU fallback")
    protocol = declaration()
    raw.mkdir(parents=True, exist_ok=False)
    plan = [(arm, task, "stationary") for arm in protocol["candidates"] for task in protocol["tasks"]]
    plan += [(arm, protocol["adaptation"]["task"], "shift") for arm in protocol["candidates"]]
    results = []
    for arm, task, phase in plan:
        name = f"{arm}-{task}-{phase}"
        log = raw / (name + ".log")
        print(json.dumps(dict(event="start", cell=name, log=str(log))), flush=True)
        allowance = protocol["budget"]["per_stationary_trial_seconds" if phase == "stationary" else "per_shift_trial_seconds"]
        command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.gaussian_frozen_prior", "trial",
                   "--arm", arm, "--task", task, "--phase", phase, "--output", str(raw), "--device", device]
        with log.open("w") as stdout:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, timeout=allowance + 60)
        if completed.returncode:
            raise RuntimeError("trial failed; no retry: " + str(log))
        result = json.loads((raw / name / "receipt.json").read_text())
        results.append(result)
        print(json.dumps(dict(event="completed", cell=name, acquisition=result["acquisition_verdict"],
            hold=result["hold_verdict"], seconds=result["loop_seconds"])), flush=True)
    if sum(row["additional_updates"] for row in results) != protocol["budget"]["new_training_updates"]:
        raise ValueError("update budget differs")
    if len({row["data_sha256"] for row in results if row["phase"] == "shift"}) != 1:
        raise ValueError("shift batch sequences differ")
    atomic_json(raw / "results.json", dict(schema_version=1, protocol_sha256=file_hash(PROTOCOL), results=results,
        new_training_updates=sum(row["additional_updates"] for row in results),
        new_training_loop_seconds=sum(row["loop_seconds"] for row in results)))


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

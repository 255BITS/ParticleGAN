"""Prospective CUDA-only graph/gradient diagnostic at the live400 boundary."""
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

from benchmarks.transfer_suite.vector_tasks import sample_target
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.sources import inspect_source, runtime_manifest
from experiments.forge.state import state_digest, require_same_formulation, require_optimizer_steps
from experiments.forge.vectorprofiles import build_vector_models
from .api_vectors import _bounds
from .reproducibility import reproducible_execution
from .ring16_quality import score_samples

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/ring16-runtime-rounding/protocol.json"
MODULE = "benchmarks.toy_audit.ring16_runtime_rounding"


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(item) for item in value)
    return deepcopy(value)


def metadata(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device),
            "version": value._version, "storage_offset": value.storage_offset(),
            "data_pointer_mod_256": value.data_ptr() % 256}


def graph(root, parameters):
    """Read topology; no node execution hooks, derivatives or saved-tensor unpack."""
    nodes, indices, pending = [], {}, [root]
    while pending:
        node = pending.pop()
        if node is None or node in indices:
            continue
        indices[node] = len(nodes)
        nodes.append(node)
        pending.extend(child for child, _ in node.next_functions if child is not None)
    result = []
    for node in nodes:
        variable = getattr(node, "variable", None)
        result.append({"id": indices[node], "type": node.name(),
                       "sequence_number": node._sequence_nr(),
                       "parameter": parameters.get(id(variable)),
                       "next": [[None if child is None else indices[child], slot]
                                for child, slot in node.next_functions]})
    return result


class BoundaryTrace:
    """Copies actual tensors and reads actual graphs, executing originals once."""
    def __init__(self, trainer):
        from particlegan.optim import dualnorm
        self.rows = {"forwards": [], "backward_graphs": [], "optimizer_gradients": {}, "polar": []}
        parameters = {id(p): f"{role}.{name}" for role, model in
                      (("D", trainer.D), ("G", trainer.G), ("prior", trainer.prior))
                      for name, p in model.named_parameters()}
        self.old_backward, self.old_polar = torch.Tensor.backward, dualnorm.polar_factor
        original_backward, original_polar = self.old_backward, self.old_polar
        def backward(tensor, *args, **kwargs):
            self.rows["backward_graphs"].append({"value": cpu(tensor),
                "autograd_multithreading_enabled": torch.autograd.is_multithreading_enabled(),
                "nodes": graph(tensor.grad_fn, parameters)})
            return original_backward(tensor, *args, **kwargs)
        def polar(tensor):
            result = original_polar(tensor)
            self.rows["polar"].append({"input": cpu(tensor), "output": cpu(result)})
            return result
        torch.Tensor.backward, dualnorm.polar_factor = backward, polar
        self.handles = []
        for role, model in (("D", trainer.D), ("G", trainer.G)):
            def forward(model, inputs, output, role=role):
                self.rows["forwards"].append({"role": role, "input": cpu(inputs[0]), "output": cpu(output)})
            self.handles.append(model.register_forward_hook(forward))
        for role, optimizer, models in (("D", trainer.opt_d, (("D", trainer.D),)),
                                        ("G", trainer.opt_g, (("G", trainer.G), ("prior", trainer.prior)))):
            def before(optimizer, args, kwargs, role=role, models=models):
                self.rows["optimizer_gradients"][role] = {
                    f"{name}.{key}": None if p.grad is None else cpu(p.grad)
                    for name, model in models for key, p in model.named_parameters()}
            self.handles.append(optimizer.register_step_pre_hook(before))

    def close(self):
        from particlegan.optim import dualnorm
        torch.Tensor.backward, dualnorm.polar_factor = self.old_backward, self.old_polar
        for handle in self.handles:
            handle.remove()


def read_protocol():
    value = json.loads(PROTOCOL.read_text())
    if value["status"] != "ready":
        raise ValueError("protocol is not executable")
    for path, digest in value["bindings"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"frozen input changed: {path}")
    return value


def trial(arm, output, prior_root, device):
    # Check before entering reproducible_execution: it initializes CUDA RNG.
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU neural fallback")
    started = time.monotonic()
    try:
        return cuda_trial(arm, output, prior_root, device=device, arm_started=started)
    except Exception as error:
        directory = output / str(arm)
        if directory.exists() and not (directory / "receipt.json").exists():
            declared = read_protocol()["arms"][arm]
            atomic_json(directory / "interruption.json", {
                "arm": arm, "status": "INCOMPLETE", "qualification_input": False,
                "error_type": type(error).__name__, "error": str(error),
                "elapsed_seconds_including_setup": time.monotonic() - started,
                "exact_new_updates": None,
                "conservative_new_updates_debit": declared["new_updates"],
                "conservative_seconds_debit": declared["timeout_seconds"],
                "accounting": "Full arm reservation charged because exact completed update count is unavailable; no retry.",
                "artifacts": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}})
        raise


@reproducible_execution
def cuda_trial(arm, output, prior_root, *, device, arm_started):
    protocol = read_protocol()
    if not torch.autograd.is_multithreading_enabled():
        raise ValueError("ordinary prefix requires autograd multithreading enabled")
    if arm not in protocol["arms"]:
        raise ValueError("undeclared arm")
    declared = protocol["arms"][arm]
    directory = output / arm
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "frozen-protocol.json", protocol)
    task = json.loads((ROOT / protocol["task_path"]).read_text())
    candidate = json.loads((ROOT / protocol["candidate_path"]).read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    g, d = build_vector_models(context, task["execution"]["host_definition"])
    trainer = context.build_trainer(g, d, max_steps=400)
    if any(p.device.type != "cuda" for model in (g, d, trainer.prior) for p in model.parameters()):
        raise RuntimeError("all neural parameters must be CUDA")
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    evaluation = context.streams.generator("eval", component="live", purpose="samples")
    initial = cpu(context.state_dict())
    source = inspect_source(ROOT, extra_paths=(str(PROTOCOL.relative_to(ROOT)),))
    atomic_json(directory / "source.json", source)
    parent = prior_root / "live/prefix-state.pt"
    if file_hash(parent) != protocol["prefix_file_sha256"]:
        raise ValueError("retained original prefix differs")
    if not declared["fresh_prefix"]:
        context.load_state_dict(torch.load(parent, map_location="cpu", weights_only=True))
    started, digest, observations, snapshots = arm_started, hashlib.sha256(), [], []
    checks = {math.ceil(i * 400 / 24) for i in range(1, 25)}
    spec = task["execution"]["host_definition"]
    while trainer.completed_steps < 400:
        step = trainer.completed_steps + 1
        real = sample_target(spec, 128, data, step - 1)
        digest.update(real.numpy().tobytes())
        trainer.step(real.to(device), collect_stats=False)
        if step in checks:
            points = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
            metrics = score_samples(points, spec, step)
            observations.append({"step": step, "metrics": metrics,
                                 "full_pass": not _bounds(metrics, task["evaluation"]["thresholds"])})
            snapshots.append({"step": step, "samples": points, "metrics": metrics})
            print(json.dumps({"event": "prefix_observation", "arm": arm, **observations[-1]}), flush=True)
        if time.monotonic() - started > declared["timeout_seconds"]:
            raise TimeoutError("frozen arm timeout")
    prefix = cpu(context.state_dict())
    torch.save(prefix, directory / "prefix-state.pt")
    if state_digest(prefix) != protocol["prefix_state_digest"]:
        raise ValueError("prefix differs: retain failure and stop, no replacement fixture")
    boundary = {f"{role}.{name}": metadata(p) for role, model in
                (("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior))
                for name, p in model.named_parameters()}
    atomic_json(directory / "boundary-runtime.json", boundary)
    trainer.extend_execution(401)
    real = sample_target(spec, 128, data, 400)
    digest.update(real.numpy().tobytes())
    trace = BoundaryTrace(trainer)
    try:
        with torch.autograd.set_multithreading_enabled(not declared["serial_at_401"]):
            update = trainer.step(real.to(device), collect_stats=False)
        if any(not bool(torch.isfinite(v).all()) for v in update.values() if isinstance(v, torch.Tensor)):
            raise FloatingPointError("nonfinite boundary update")
    finally:
        trace.close()
        torch.save({"real": real, **trace.rows}, directory / "trace401.pt")
    final = cpu(context.state_dict())
    torch.save(final, directory / "state401.pt")
    torch.save(snapshots, directory / "observations.pt")
    atomic_json(directory / "curve.json", observations)
    require_same_formulation(initial, final)
    require_optimizer_steps(final, 401)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    if elapsed > declared["timeout_seconds"]:
        raise TimeoutError("frozen arm timeout")
    atomic_json(directory / "receipt.json", {"arm": arm, "scope": protocol["scope"],
        "qualification_input": False, "quality_verdict": "INCOMPLETE",
        "source_commit": source["origin_commit"], "source_digest": source["digest"],
        "protocol_sha256": file_hash(PROTOCOL), "new_updates": declared["new_updates"],
        "completed_updates": 401, "elapsed_seconds": elapsed, "device": device,
        "timing_scope": "whole child arm including reproducibility setup, context construction and restore; controller separately measures process startup and exit",
        "gpu": torch.cuda.get_device_name(device), "runtime": runtime_manifest(),
        "training_batch_digest": digest.hexdigest(), "named_stream_manifest": context.streams.manifest(),
        "prefix_bit_exact": True, "serial_at_401": declared["serial_at_401"],
        "artifacts": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}})
    print(json.dumps({"event": "arm_complete", "arm": arm, "new_updates": declared["new_updates"]}), flush=True)


def execute(protocol, output, prior_root, device):
    output.mkdir(parents=True, exist_ok=False)
    peers = {arm: {"status": "UNEXECUTED", "updates_debit": 0, "seconds_debit": 0}
             for arm in protocol["arms"]}
    complete = False
    try:
        for arm, declared in protocol["arms"].items():
            print(json.dumps({"event": "arm_start", "arm": arm}), flush=True)
            started = time.monotonic()
            peers[arm] = {"status": "INCOMPLETE", "updates_debit": declared["new_updates"],
                          "seconds_debit": declared["timeout_seconds"]}
            try:
                subprocess.run([sys.executable, "-u", "-m", MODULE, "trial", "--arm", arm,
                                "--output", str(output), "--prior-root", str(prior_root),
                                "--device", device], check=True, timeout=declared["timeout_seconds"])
                receipt = json.loads((output / arm / "receipt.json").read_text())
                if receipt["new_updates"] != declared["new_updates"]:
                    raise ValueError("arm receipt update count differs from declaration")
            except Exception as error:
                peers[arm] = {"status": "TIMEOUT" if isinstance(error, subprocess.TimeoutExpired) else "INCOMPLETE",
                    "updates_debit": declared["new_updates"], "seconds_debit": declared["timeout_seconds"],
                    "measured_process_wall_seconds": time.monotonic() - started,
                    "error_type": type(error).__name__, "error": str(error)}
                directory = output / arm
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / "controller-interruption.json", {"qualification_input": False,
                    "arm": arm, **peers[arm], "accounting": "Full reservation charged; remaining peers unexecuted; no retries."})
                raise
            peers[arm] = {"status": "COMPLETE", "updates_debit": receipt["new_updates"],
                          "seconds_debit": time.monotonic() - started,
                          "timing_scope": "whole subprocess including startup, setup, training, persistence and exit"}
        if (sum(p["updates_debit"] for p in peers.values()) > protocol["max_new_host_updates"] or
                sum(p["seconds_debit"] for p in peers.values()) > protocol["max_reserved_seconds"]):
            raise RuntimeError("campaign reservation exceeded")
        complete = True
    finally:
        atomic_json(output / "campaign-summary.json", {"scope": protocol["scope"], "qualification_input": False,
            "status": "COMPLETE" if complete else "INCOMPLETE", "peers": peers,
            "attempts": sum(p["status"] != "UNEXECUTED" for p in peers.values()),
            "new_updates_debit": sum(p["updates_debit"] for p in peers.values()),
            "seconds_debit": sum(p["seconds_debit"] for p in peers.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "trial"))
    parser.add_argument("--arm")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "trial":
        trial(args.arm, args.output, args.prior_root, args.device)
        return
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("CUDA is required; no CPU neural fallback")
    execute(read_protocol(), args.output, args.prior_root, args.device)


if __name__ == "__main__":
    main()

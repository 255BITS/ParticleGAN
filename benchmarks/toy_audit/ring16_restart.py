"""Bounded CUDA checkpoint-boundary diagnostic using the public trainer."""
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
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.sources import inspect_source, runtime_manifest
from experiments.forge.state import state_digest, require_same_formulation, require_optimizer_steps
from experiments.forge.vectorprofiles import build_vector_models
from .api_vectors import _bounds
from .reproducibility import reproducible_execution
from .ring16_quality import score_samples
from benchmarks.transfer_suite.vector_tasks import sample_target

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/ring16-failure/protocol.json"
MODULE = "benchmarks.toy_audit.ring16_restart"


def read(path):
    return json.loads(path.read_text())


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return deepcopy(value)


def layout(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "contiguous": tensor.is_contiguous(), "dtype": str(tensor.dtype), "device": str(tensor.device)}


def runtime_state(trainer):
    models = {"G": trainer.G, "D": trainer.D, "prior": trainer.prior}
    return {"module_modes": {role: {n: m.training for n, m in model.named_modules()} for role, model in models.items()},
            "gradients": {role: {n: None if p.grad is None else cpu(p.grad)
                                  for n, p in model.named_parameters()} for role, model in models.items()},
            "parameter_layout": {role: {n: {**layout(p), "version": p._version,
                                            "grad": None if p.grad is None else layout(p.grad)}
                                         for n, p in model.named_parameters()} for role, model in models.items()},
            "prior_noise_enabled": trainer.prior._noise_enabled,
            "penalty_collect_stats": trainer.penalty.collect_stats}


class Trace:
    """Copy actual inputs/outputs to CPU; invoke every wrapped operation once."""
    def __init__(self, trainer):
        from particlegan.optim import dualnorm
        self.trainer, self.rows, self.current, self.handles = trainer, [], None, []
        self.original_polar = dualnorm.polar_factor
        self.original_sample = trainer._sample_training_prior

        def polar(matrix):
            result = self.original_polar(matrix)
            if self.current is not None:
                self.current["polar"].append({"input": cpu(matrix), "output": cpu(result),
                                               "input_layout": layout(matrix), "output_layout": layout(result)})
            return result

        def sample(n):
            result = self.original_sample(n)
            if self.current is not None:
                self.current["latents"].append({"values": cpu(result[0]), "indices": cpu(result[1])})
            return result

        dualnorm.polar_factor = polar
        trainer._sample_training_prior = sample
        for role, module in (("G", trainer.G), ("D", trainer.D)):
            def forward(module, args, output, role=role):
                if self.current is not None:
                    self.current["forwards"].append({"role": role, "training": module.training,
                        "input": cpu(args[0]), "output": cpu(output), "input_layout": layout(args[0])})
            self.handles.append(module.register_forward_hook(forward))
        parameters = {"D": dict(trainer.D.named_parameters()),
                      "G": {**{"G." + k: v for k, v in trainer.G.named_parameters()},
                            **{"prior." + k: v for k, v in trainer.prior.named_parameters()}}}
        for role, optimizer in (("D", trainer.opt_d), ("G", trainer.opt_g)):
            def before(optimizer, args, kwargs, role=role):
                if self.current is not None:
                    self.current["optimizers"][role] = {
                        "before": {k: cpu(p) for k, p in parameters[role].items()},
                        "gradients": {k: None if p.grad is None else cpu(p.grad) for k, p in parameters[role].items()},
                        "gradient_layouts": {k: None if p.grad is None else layout(p.grad) for k, p in parameters[role].items()}}
            def after(optimizer, args, kwargs, role=role):
                if self.current is not None:
                    self.current["optimizers"][role]["after"] = {k: cpu(p) for k, p in parameters[role].items()}
            self.handles.extend([optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)])

    def begin(self, step, real):
        self.current = {"step": step, "real": cpu(real), "latents": [], "forwards": [], "polar": [], "optimizers": {}}

    def end(self, context, update):
        if self.current is not None:
            self.current["named_streams"] = cpu(context.streams.state_dict())
            self.current["global_cpu_rng"] = torch.get_rng_state().clone()
            self.current["global_cuda_rng"] = torch.cuda.get_rng_state(context.device).clone()
            self.current["losses"] = cpu(update)
            self.rows.append(self.current)
            self.current = None

    def close(self):
        from particlegan.optim import dualnorm
        dualnorm.polar_factor = self.original_polar
        self.trainer._sample_training_prior = self.original_sample
        for handle in self.handles:
            handle.remove()


def declaration():
    protocol = read(PROTOCOL)
    for path, digest in protocol["bindings"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError("frozen declaration changed: " + path)
    return protocol


def build(protocol, device):
    task = read(ROOT / protocol["task_path"])
    candidate = read(ROOT / protocol["candidate_path"])
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    g, d = build_vector_models(context, task["execution"]["host_definition"])
    trainer = context.build_trainer(g, d, max_steps=400)
    if not all(p.device.type == "cuda" for model in (g, d, trainer.prior) for p in model.parameters()):
        raise ValueError("CUDA is required")
    return context, trainer, task


def suffix(rows):
    count = 0
    for point in reversed(rows):
        if not point["full_pass"]:
            break
        count += 1
    return count


@reproducible_execution
def trial(arm, output, prior_root, *, device):
    if torch.device(device).type != "cuda":
        raise ValueError("no CPU fallback")
    protocol = declaration()
    directory = output / arm
    directory.mkdir(parents=True, exist_ok=False)
    context, trainer, task = build(protocol, device)
    parent = prior_root / "runs/api/tier1-prior-smoke-v1/mog100-n256/ring16_acquisition"
    if file_hash(parent / "state.pt") != protocol["archived_parent_checkpoint_sha256"]:
        raise ValueError("archived parent differs")
    # Freeze the complete consumed-stream registry before the initial receipt.
    context.streams.generator("data", component="target", purpose="training", device="cpu")
    context.streams.generator("eval", component="live", purpose="samples")
    initial = cpu(context.state_dict())
    torch.save(initial, directory / "initial-state.pt")
    spec = task["execution"]["host_definition"]
    collect_stats = arm == "restored_stats"
    probe = arm.startswith("restored_")
    cap, timeout = (416, 30) if probe else (1600, 300)
    observations, snapshots = [], []
    restore_proof = None
    started = time.monotonic()
    if arm != "live":
        state_path = parent / "state.pt" if arm == "archive-replay" else output / "live/prefix-state.pt"
        state = torch.load(state_path, weights_only=True, map_location="cpu")
        context.load_state_dict(state)
        if arm == "restored_twice":
            context.load_state_dict(state)
        if state_digest(context.state_dict()) != state_digest(state):
            raise ValueError("serialized state was not restored exactly")
        restore_proof = {"parent": str(state_path), "sha256": file_hash(state_path),
                         "state_sha256": state_digest(state), "exact": True}
        if arm == "restored_runtime":
            runtime = torch.load(output / "live/prefix-runtime.pt", weights_only=True, map_location="cpu")
            for role, model in (("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior)):
                for name, module in model.named_modules():
                    module.training = runtime["module_modes"][role][name]
                for name, parameter in model.named_parameters():
                    gradient = runtime["gradients"][role][name]
                    parameter.grad = None if gradient is None else gradient.to(device).clone()
        source = parent if arm == "archive-replay" else output / "live"
        if arm == "archive-replay":
            observations = read(parent / "curve.json")
            snapshots = torch.load(parent / "observations.pt", weights_only=True, map_location="cpu")
            snapshots = [p for p in snapshots if p["step"] > 0]
        else:
            observations = read(source / "prefix-curve.json")
            snapshots = torch.load(source / "prefix-observations.pt", weights_only=True, map_location="cpu")
    torch.save(cpu(context.state_dict()), directory / "before-state.pt")
    torch.save(runtime_state(trainer), directory / "before-runtime.pt")
    source = inspect_source(ROOT, extra_paths=(str(PROTOCOL.relative_to(ROOT)), protocol["candidate_path"]))
    atomic_json(directory / "source.json", source)
    checks = {math.ceil(i * 400 / 24) + block for block in range(0, 1600, 400) for i in range(1, 25)}
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    evaluation = context.streams.generator("eval", component="live", purpose="samples")
    digest = hashlib.sha256()
    trace = None if arm in ("archive-replay", "restored_untraced") else Trace(trainer)
    if trainer.completed_steps == 400:
        trainer.extend_execution(cap)
    try:
        while trainer.completed_steps < cap:
            step = trainer.completed_steps + 1
            real = sample_target(spec, 128, data, step - 1)
            digest.update(real.numpy().tobytes())
            if trace is not None and 401 <= step <= 416:
                trace.begin(step, real)
            update = trainer.step(real.to(device), collect_stats=collect_stats)
            if not all(bool(torch.isfinite(v).all()) for v in update.values() if isinstance(v, torch.Tensor)):
                raise FloatingPointError("nonfinite training update")
            if trace is not None:
                trace.end(context, update)
            if step in checks:
                points = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
                metrics = score_samples(points, spec, step)
                failed = _bounds(metrics, task["evaluation"]["thresholds"])
                observations.append({"step": step, "metrics": metrics, "full_pass": not failed, "failed_bounds": failed})
                snapshots.append({"step": step, "samples": points, "metrics": metrics})
                print(json.dumps({"event": "observation", "arm": arm, "step": step, "pass": not failed,
                                  "covariance": metrics["component_covariance_error"], "hq": metrics["hq"]}), flush=True)
            if step == 400:
                torch.save(cpu(context.state_dict()), directory / "prefix-state.pt")
                torch.save(runtime_state(trainer), directory / "prefix-runtime.pt")
                atomic_json(directory / "prefix-curve.json", observations)
                torch.save(snapshots, directory / "prefix-observations.pt")
                trainer.extend_execution(cap)
            if step == 416:
                torch.save(cpu(context.state_dict()), directory / "state416.pt")
            if time.monotonic() - started > timeout:
                raise TimeoutError("frozen arm timeout exceeded")
    finally:
        if trace is not None:
            trace.close()
            torch.save(trace.rows, directory / "trace.pt")
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    final = cpu(context.state_dict())
    require_optimizer_steps(final, cap)
    require_same_formulation(initial, final)
    torch.save(final, directory / "state.pt")
    torch.save(snapshots, directory / "observations.pt")
    atomic_json(directory / "curve.json", observations)
    atomic_json(directory / "receipt.json", {"schema_version": 1, "scope": protocol["scope"],
        "qualification_input": False, "arm": arm, "source_commit": source["origin_commit"], "source_digest": source["digest"],
        "device": str(device), "gpu": torch.cuda.get_device_name(device), "runtime": runtime_manifest(),
        "protocol_sha256": file_hash(PROTOCOL), "completed_updates": cap,
        "new_updates": cap if arm == "live" else cap - 400, "elapsed_seconds": elapsed,
        "training_batch_sequence_sha256": digest.hexdigest(), "restore": restore_proof,
        "full_verdict": "INCOMPLETE" if probe else "PASS" if suffix(observations) >= 5 else "FAIL",
        "terminal_suffix": suffix(observations), "passing_observations": sum(p["full_pass"] for p in observations),
        "final_metrics": observations[-1]["metrics"], "recipe": context.recipe.to_dict(), "named_stream_manifest": context.streams.manifest(),
        "artifacts": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}})
    print(json.dumps({"event": "arm_complete", "arm": arm, "new_updates": cap if arm == "live" else cap - 400,
                      "seconds": elapsed, "suffix": suffix(observations)}), flush=True)


def execute(output, prior_root, device):
    protocol = declaration()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "frozen-protocol.json", protocol)
    for arm in ("live", "restart", "archive-replay", "restored_untraced"):
        print(json.dumps({"event": "arm_start", "arm": arm}), flush=True)
        subprocess.run([sys.executable, "-u", "-m", MODULE, "trial", "--arm", arm,
                        "--output", str(output), "--prior-root", str(prior_root), "--device", device], check=True)
    live = torch.load(output / "live/observations.pt", map_location="cpu", weights_only=True)
    restart = torch.load(output / "restart/observations.pt", map_location="cpu", weights_only=True)
    archived = torch.load(prior_root / "runs/api/tier1-prior-duration-v1/mog100-n256-ring16_acquisition/observations.pt",
                          map_location="cpu", weights_only=True)
    archived = [p for p in archived if p["step"] > 0]
    replay = torch.load(output / "archive-replay/observations.pt", map_location="cpu", weights_only=True)
    mismatch = any(not torch.equal(a["samples"], b["samples"]) for a, b in zip(live, restart)) or any(
        not torch.equal(a["samples"], b["samples"]) for a, b in zip(archived, replay))
    if mismatch:
        for arm in ("restored_stats", "restored_twice", "restored_runtime"):
            subprocess.run([sys.executable, "-u", "-m", MODULE, "trial", "--arm", arm,
                            "--output", str(output), "--prior-root", str(prior_root), "--device", device], check=True)
    receipts = [read(p) for p in output.glob("*/receipt.json")]
    updates = sum(r["new_updates"] for r in receipts)
    seconds = sum(r["elapsed_seconds"] for r in receipts)
    if updates > protocol["max_new_host_updates"] or seconds > protocol["max_reserved_seconds"]:
        raise RuntimeError("frozen diagnostic budget exceeded")
    atomic_json(output / "completion.json", {"attempts": len(receipts), "new_updates": updates,
                                              "elapsed_seconds": seconds, "adaptive_probes_triggered": mismatch})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "trial"))
    parser.add_argument("--arm")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/api/ring16-restart-diagnostic-v1")
    parser.add_argument("--prior-root", type=Path, default=Path("/home/martyn/dev/ParticleGAN-tier1-prior-smoke"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "run":
        execute(args.output.resolve(), args.prior_root.resolve(), args.device)
    else:
        trial(args.arm, args.output.resolve(), args.prior_root.resolve(), device=args.device)


if __name__ == "__main__":
    main()

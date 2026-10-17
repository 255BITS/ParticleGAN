"""Bounded, source-frozen component diagnosis of the public DCGAN benchmark.

Plain latency is measured without component wrappers. A second deterministic
replay instruments only the final three updates; nested synchronized host spans
are inclusive and exclusive, and cannot replace plain latency. Bulk traces and
checkpoints remain local. The immutable declaration must exist before execution.
"""
from __future__ import annotations

import argparse
import importlib
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from .api import FormulationContext, task_formulation_context
from .contracts import atomic_json, canonical, file_hash, stable_hash
from .scaling_benchmark import _sync, run_benchmark


SCHEMA = "forge_dcgan_component_profile_v1"


def declare(root, output, *, candidates, tasks, source_commit, reservation_seconds=960):
    """Freeze the complete finite diagnostic before its first paid operation."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(set((root / "particlegan").rglob("*.py")) |
                          set((root / "experiments/forge").glob("*.py")) |
                          {root / "benchmarks/toy_audit/reproducibility.py"})
    value = dict(schema=SCHEMA, source_commit=source_commit,
                 source_files_sha256={str(p.relative_to(root)): file_hash(p) for p in source_paths},
                 candidates={name: json.loads((root / "configs/forge/ideas" / f"{name}.json").read_text())
                             for name in candidates},
                 tasks={name: json.loads((root / "configs/forge/tasks" / f"{name}.json").read_text())
                        for name in tasks},
                 protocol=dict(seed=0, plain_replay=True, component_replay=True,
                               instrument_final_updates=3, torch_profile_final_update=True,
                               workload="unchanged declared task; 3 warmup + 10 measured updates per replay",
                               qualification_input=False, contention="unknown unless validated real telemetry",
                               numerical_gate="both replays PASS; identical initial models, initial API prior, batch streams and final learned parameter hashes",
                               comparison="component spans diagnose only; plain latency is unwrapped; no certified paired speed ranking with unknown contention"),
                 budget=dict(reservation_seconds=reservation_seconds,
                             per_arm_seconds=reservation_seconds // (len(candidates) * len(tasks)),
                             max_arms=len(candidates) * len(tasks), paid_retries=0))
    path = output / "declaration.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("immutable diagnostic declaration differs; choose a new output")
    atomic_json(path, value)
    return value


class ComponentSpans:
    """Read-only wrappers with nested inclusive/exclusive synchronized spans."""

    def __init__(self, device, output):
        self.device, self.output = torch.device(device), Path(output)
        self.active, self.stage = False, "critic"
        self.stack, self.rows = [], []
        self.step = 0
        self.final_step = 0
        self.newton_depth = 0
        self.matrix_samples = {}

    @contextmanager
    def span(self, name, **metadata):
        if not self.active:
            yield
            return
        _sync(self.device)
        tick = time.perf_counter()
        frame = {"children_ms": 0.}
        self.stack.append(frame)
        with torch.profiler.record_function("forge::" + name):
            try:
                yield
            finally:
                _sync(self.device)
                elapsed = (time.perf_counter() - tick) * 1000
                self.stack.pop()
                if self.stack:
                    self.stack[-1]["children_ms"] += elapsed
                self.rows.append(dict(step=self.step, name=name, inclusive_ms=elapsed,
                                      exclusive_ms=max(0., elapsed-frame["children_ms"]), **metadata))

    def wrap(self, function, name, *, matrix=False):
        def measured(*args, **kwargs):
            metadata = ({"shape": list(args[0].shape), "matrix_count": math.prod(args[0].shape[:-2])}
                        if matrix and args else {})
            label = name() if callable(name) else name
            with self.span(label, **metadata):
                return function(*args, **kwargs)
        return measured

    @contextmanager
    def attach(self):
        """Install only in the explicitly labelled component replay."""
        import particlegan.bispf as bispf
        import particlegan.optim.dualnorm as dualnorm
        import particlegan.training as training
        original_build = FormulationContext.build_trainer
        with ExitStack() as stack:
            def build(context, *args, **kwargs):
                trainer = original_build(context, *args, **kwargs)
                self._bind(trainer, stack)
                return trainer
            stack.enter_context(patch.object(FormulationContext, "build_trainer", build))
            for owner, attr, label, matrix in (
                    (bispf, "sinkhorn_coupling", "critic.sinkhorn", False),
                    (dualnorm, "polar_factor", lambda: self.stage + ".optimizer." +
                     ("fallback_polar" if self.newton_depth else "polar"), True),
                    (torch.linalg, "svd", lambda: self.stage + ".optimizer.svd", True),
                    (torch.autograd, "grad", lambda: self.stage + ".autograd.grad", False),
                    (torch.autograd, "backward", lambda: self.stage + ".backward", False),
                    (training, "constraint_geometry_backward", "generator.constraint_backward", False)):
                stack.enter_context(patch.object(owner, attr, self.wrap(getattr(owner, attr), label, matrix=matrix)))
            # Frobenius is imported into dualnorm at module load; quintic is
            # imported inside step(). Patch their actual lookup locations so
            # the first lazy import and already-loaded modules both work.
            if hasattr(dualnorm, "frobenius_direction"):
                stack.enter_context(patch.object(dualnorm, "frobenius_direction",
                    self.wrap(dualnorm.frobenius_direction,
                              lambda: self.stage + ".optimizer.frobenius", matrix=True)))
            quintic_path = Path(dualnorm.__file__).with_name("convolution_quintic.py")
            if quintic_path.is_file():
                quintic = importlib.import_module("particlegan.optim.convolution_quintic")
                stack.enter_context(patch.object(quintic, "quintic_direction",
                    self.wrap(quintic.quintic_direction,
                              lambda: self.stage + ".optimizer.quintic", matrix=True)))
            if hasattr(dualnorm, "_newton_schulz_polar"):
                original_newton = dualnorm._newton_schulz_polar
                def newton(matrices, **kwargs):
                    shape = list(matrices.shape)
                    self.newton_depth += 1
                    try:
                        with self.span(self.stage + ".optimizer.newton_schulz", shape=shape,
                                       matrix_count=math.prod(shape[:-2])):
                            result = original_newton(matrices, **kwargs)
                    finally:
                        self.newton_depth -= 1
                    # Capture a representative actual-gradient matrix only on
                    # the separate torch-profile update. It never affects the
                    # two synchronized component medians or plain replay.
                    key = self.stage + ":" + str(shape[-2:])
                    if self.active and self.step == self.final_step and key not in self.matrix_samples:
                        self.matrix_samples[key] = dict(
                            matrix=matrices.detach().reshape(-1, *shape[-2:])[0].cpu(),
                            smoothing=kwargs["smoothing"], batch_shape=shape)
                    return result
                stack.enter_context(patch.object(dualnorm, "_newton_schulz_polar", newton))
            yield self

    def _bind(self, trainer, stack):
        original_step = trainer.step
        total = trainer.max_steps
        self.final_step = total
        def step(*args, **kwargs):
            self.step = trainer.completed_steps + 1
            self.active = self.step > total - 3
            self.stage = "critic"
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = (torch.profiler.profile(activities=activities, record_shapes=True)
                        if self.step == total else None)
            if profiler is not None:
                profiler.__enter__()
            try:
                with self.span("complete_update"):
                    return original_step(*args, **kwargs)
            finally:
                self.active = False
                if profiler is not None:
                    profiler.__exit__(None, None, None)
                    profiler.export_chrome_trace(str(self.output / "torch-trace.json"))
                    (self.output / "torch-table.txt").write_text(profiler.key_averages(group_by_input_shape=True).table(
                        sort_by="self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total", row_limit=60))
                    if self.matrix_samples:
                        torch.save(self.matrix_samples, self.output / "actual-gradient-matrices.pt")
                with (self.output / "components.jsonl").open("a") as stream:
                    for row in self.rows:
                        stream.write(canonical(row) + "\n")
                self.rows.clear()
        stack.enter_context(patch.object(trainer, "step", step))
        for owner, attr, label in (
                (trainer.G, "forward", "generator.forward"),
                (trainer.D, "forward", lambda: self.stage + ".critic_forward"),
                (trainer.prior_regularizer, "forward", "prior.regularizer"),
                (trainer.penalty.regularizer, "penalty", "critic.penalty"),
                (trainer, "_sample_training_prior", "prior.sample"),
                (trainer.opt_g, "step", "generator.optimizer")):
            stack.enter_context(patch.object(owner, attr, self.wrap(getattr(owner, attr), label)))
        original_d_step = trainer.opt_d.step
        def critic_step(*args, **kwargs):
            with self.span("critic.optimizer"):
                result = original_d_step(*args, **kwargs)
            self.stage = "generator"
            return result
        stack.enter_context(patch.object(trainer.opt_d, "step", critic_step))
        for name in ("begin_step", "observe_support", "observe_critic_pair", "after_critic_step",
                     "after_generator_backward", "after_generator_step", "finish_step"):
            stack.enter_context(patch.object(trainer.policy, name, self.wrap(getattr(trainer.policy, name), "policy."+name)))


def _initial_prior(receipt):
    initialization = receipt["api"]["initialization"]
    return initialization["prior"]


def run(root, output, *, candidate_id, task_id, device="cuda:0"):
    root, output = Path(root).resolve(), Path(output).resolve()
    declaration = json.loads((output / "declaration.json").read_text())
    if declaration.get("schema") != SCHEMA:
        raise ValueError("missing supported immutable declaration")
    for relative, expected in declaration["source_files_sha256"].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"frozen profiling source changed: {relative}")
    candidate, task = declaration["candidates"][candidate_id], declaration["tasks"][task_id]
    arm = output / f"{candidate_id}-{task_id}"
    arm.mkdir(exist_ok=False)
    context = task_formulation_context(candidate, task, device=device)
    spec = deepcopy(task["execution"]["performance"])
    plain = run_benchmark(context.recipe, spec, arm / "plain", device)
    component_dir = arm / "components"
    component_dir.mkdir()
    spans = ComponentSpans(device, component_dir)
    with spans.attach():
        component = run_benchmark(context.recipe, spec, component_dir, device)
    same = {key: plain["identity"][key] == component["identity"][key]
            for key in ("initial_model_sha256", "initial_parameter_sha256", "initialization_sha256",
                        "batch_sequence_sha256", "final_parameter_sha256")}
    same["public_api_initial_prior"] = _initial_prior(plain) == _initial_prior(component)
    rows = [json.loads(line) for line in (component_dir / "components.jsonl").read_text().splitlines()]
    grouped = {}
    for row in rows:
        if row["step"] == spec["warmup_updates"] + spec["measured_updates"]:
            continue  # torch profiler overhead is a separate observation
        key = row["name"]
        step = grouped.setdefault(key, {}).setdefault(row["step"], dict(inclusive_ms=0., exclusive_ms=0., calls=0))
        for field in ("inclusive_ms", "exclusive_ms"):
            step[field] += row[field]
        step["calls"] += 1
    summary = dict(schema=SCHEMA, candidate_id=candidate_id, task_id=task_id,
                   declaration_sha256=file_hash(output / "declaration.json"),
                   source_commit=declaration["source_commit"],
                   replay_parity=same, passed=plain["status"] == component["status"] == "PASS" and all(same.values()),
                   plain_metrics=plain["metrics"], hardware=plain["hardware"],
                   component_note="median of last two non-torch-profile updates; inclusive overlaps, exclusive subtracts nested spans; added synchronization perturbs timing",
                   component_medians={key: {field: statistics.median(step[field] for step in steps.values())
                                             for field in ("inclusive_ms", "exclusive_ms", "calls")}
                                      for key, steps in grouped.items()})
    # JSON has lists for shapes, not tuples.
    summary["polar_shapes"] = {str(shape): sum(row["name"].endswith(".polar") and tuple(row.get("shape", ())) == shape for row in rows)
                               for shape in {tuple(row["shape"]) for row in rows if row["name"].endswith(".polar")}}
    newton_rows = [row for row in rows if row["name"].endswith(".newton_schulz")]
    summary["newton_schulz"] = dict(
        matrix_count=sum(row["matrix_count"] for row in newton_rows),
        exact_fallback_count=sum(row["name"].endswith(".fallback_polar") for row in rows),
        observation_scope="final three instrumented updates; diagnostic counters; representative gradients saved only in final torch-profile update",
        matrix_samples_sha256=(file_hash(component_dir / "actual-gradient-matrices.pt")
                               if (component_dir / "actual-gradient-matrices.pt").exists() else None))
    summary["convolution_helpers"] = {
        name: dict(calls=sum(row["name"].endswith("." + name) for row in rows),
                   matrix_count=sum(row.get("matrix_count", 1) for row in rows
                                    if row["name"].endswith("." + name)))
        for name in ("frobenius", "quintic")}
    atomic_json(arm / "summary.json", summary)
    print(canonical(dict(event="profile_complete", candidate=candidate_id, task=task_id,
                         passed=summary["passed"], plain_ms=plain["metrics"]["update_ms_median"],
                         components=summary["component_medians"])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reservation-seconds", type=int, default=960)
    args = parser.parse_args()
    if args.action == "declare":
        if not args.source_commit:
            parser.error("declare requires --source-commit")
        value = declare(args.root, args.output, candidates=args.candidate, tasks=args.task,
                        source_commit=args.source_commit, reservation_seconds=args.reservation_seconds)
        print(canonical(dict(event="declared", output=str(args.output), budget=value["budget"])))
    else:
        if len(args.candidate) != 1 or len(args.task) != 1:
            parser.error("run selects one declared candidate/task arm")
        result = run(args.root, args.output, candidate_id=args.candidate[0], task_id=args.task[0], device=args.device)
        raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

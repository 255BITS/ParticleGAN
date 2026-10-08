"""Frozen CUDA diagnostic: smoothed DualNorm across the four prior runtime arms.

Uses Forge's unchanged public adapters, tasks, gates and named streams. Historical
truncation/scheduling factors are process-local overrides; no qualification credit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks.toy_audit.api_images import WordFixture
from benchmarks.toy_audit.reproducibility import construction_rng
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sources import inspect_source
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, task_fingerprint
from particlegan import GANTrainer
import particlegan.optim.dualnorm as dualnorm

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"


class FactorAudit:
    def __init__(self, arm, output, task, smoothing):
        self.arm, self.output, self.smoothing = arm, output, smoothing
        steps = task["execution"]["steps"]
        import math
        self.checks = {math.ceil(i * steps / 24) for i in range(1, 25)} | {1, 2}
        self.step = 0
        self.counts = Counter()
        self.rank_rows, self.states, self.initial, self.handles = [], [], {}, []
        self.data_digest = hashlib.sha256()
        self.word_data_generator = None
        self.installed = set()

    def install(self, models, optimizers):
        for name, module in models.items():
            self.initial[name] = state_digest(module.state_dict())
        def forward(module, inputs):
            if torch.is_grad_enabled():
                self.counts["graph_forward:" + str(torch.autograd.is_multithreading_enabled())] += 1
            if any(p.device.type != "cuda" for p in module.parameters()):
                raise RuntimeError("neural model left CUDA")
        def optimizer(opt, args, kwargs):
            if opt.smoothing != self.smoothing:
                raise RuntimeError("optimizer did not consume declared smoothing")
            self.counts["optimizer:" + str(torch.autograd.is_multithreading_enabled())] += 1
        for module in models.values():
            self.handles.append(module.register_forward_pre_hook(forward))
        for opt in optimizers:
            self.handles.append(opt.register_step_pre_hook(optimizer))

    def save_models(self, models):
        if self.step in self.checks:
            self.states.append({"step": self.step, "models": {
                name: state_digest(module.state_dict()) for name, module in models.items()}})

    def __enter__(self):
        audit = self
        self.old_word_step, self.old_execute = WordFixture.step, GANTrainer._execute_step
        self.old_state, self.old_property = GANTrainer.state_dict, GANTrainer.serial_backward
        self.old_polar, self.old_grad, self.old_randint = dualnorm.polar_factor, torch.autograd.grad, torch.randint
        def execute(trainer, real, *, generator_real=None, collect_stats=False):
            models = dict(G=trainer.G, D=trainer.D, prior=trainer.prior)
            if id(trainer) not in audit.installed:
                audit.install(models, (trainer.opt_g, trainer.opt_d))
                audit.installed.add(id(trainer))
            audit.step = trainer.completed_steps + 1
            try:
                with torch.autograd.set_multithreading_enabled(audit.arm["autograd_multithreading"]):
                    audit.counts["step:" + str(torch.autograd.is_multithreading_enabled())] += 1
                    result = trainer._step(real, generator_real=generator_real, collect_stats=collect_stats)
            except Exception:
                trainer.policy.abort_step()
                raise
            audit.save_models(models)
            return result
        def word_step(fixture):
            audit.step = fixture.completed_steps + 1
            models = dict(G=fixture.G, E=fixture.E, D=fixture.D, prior=fixture.prior)
            if id(fixture) not in audit.installed:
                audit.install(models, (fixture.opt_g, fixture.opt_d))
                audit.installed.add(id(fixture))
            audit.word_data_generator = fixture.data_generator
            with torch.autograd.set_multithreading_enabled(audit.arm["autograd_multithreading"]):
                audit.counts["step:" + str(torch.autograd.is_multithreading_enabled())] += 1
                result = audit.old_word_step(fixture)
            audit.save_models(models)
            return result
        def state(trainer):
            value = audit.old_state(trainer)
            value["serial_backward"] = not audit.arm["autograd_multithreading"]
            return value
        def polar(matrix, *, smoothing=0., truncate=True):
            if smoothing != audit.smoothing or matrix.device.type != "cuda":
                raise RuntimeError("polar call differs from declared CUDA smoothing")
            audit.counts["polar:" + str(torch.autograd.is_multithreading_enabled())] += 1
            value = audit.old_polar(matrix, smoothing=smoothing, truncate=audit.arm["truncation"])
            if audit.step in audit.checks:
                singular = torch.linalg.svdvals(matrix)
                threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular[0]
                keep = singular > threshold
                weights = singular / torch.hypot(singular, torch.full_like(singular, smoothing))
                if audit.arm["truncation"]:
                    weights *= keep
                audit.rank_rows.append(dict(step=audit.step, shape=list(matrix.shape),
                    retained_rank=int(keep.sum()), available_rank=len(singular),
                    below_threshold=int((~keep).sum()),
                    actual_removed=int((~keep).sum()) if audit.arm["truncation"] else 0,
                    singular_max=float(singular[0]), singular_min=float(singular[-1]),
                    update_spectral_weights_mean=float(weights.mean()),
                    update_sha256=state_digest(value)))
            return value
        def grad(*args, **kwargs):
            audit.counts["autograd_grad:" + str(torch.autograd.is_multithreading_enabled())] += 1
            return audit.old_grad(*args, **kwargs)
        def randint(*args, **kwargs):
            value = audit.old_randint(*args, **kwargs)
            if audit.word_data_generator is not None and kwargs.get("generator") is audit.word_data_generator:
                audit.data_digest.update(value.detach().cpu().numpy().tobytes())
            return value
        WordFixture.step, GANTrainer._execute_step = word_step, execute
        GANTrainer.state_dict, GANTrainer.serial_backward = state, property(lambda _: not audit.arm["autograd_multithreading"])
        dualnorm.polar_factor, torch.autograd.grad, torch.randint = polar, grad, randint
        return self

    def __exit__(self, *exc):
        WordFixture.step, GANTrainer._execute_step = self.old_word_step, self.old_execute
        GANTrainer.state_dict, GANTrainer.serial_backward = self.old_state, self.old_property
        dualnorm.polar_factor, torch.autograd.grad, torch.randint = self.old_polar, self.old_grad, self.old_randint
        for handle in self.handles:
            handle.remove()
        atomic_json(self.output / "factor-audit.json", dict(mode_counts=dict(self.counts),
            initial_models=self.initial, states=self.states, ranks=self.rank_rows,
            word_data_sequence_sha256=self.data_digest.hexdigest(), additional_rng_draws=0))


def prepare(task_id):
    protocol = read_json(PROTOCOL)
    task = load_tasks(ROOT)[task_id]
    if task_fingerprint(task) != protocol["tasks"][task_id]["fingerprint"]:
        raise ValueError("task identity changed")
    candidate = read_json(ROOT / protocol["candidate_path"])
    if file_hash(ROOT / protocol["candidate_path"]) != protocol["candidate_sha256"]:
        raise ValueError("incumbent recipe changed")
    candidate = deepcopy(candidate)
    candidate["id"] = protocol["id"]
    candidate["recipe_overrides"]["optimizer_smoothing"] = protocol["smoothing_lambda"]
    for key in ("resolved_recipe", "resolved_configuration_recipe"):
        if key in candidate:
            candidate[key]["optimizer_smoothing"] = protocol["smoothing_lambda"]
    return protocol, task, dict(candidate=candidate, candidate_revision=stable_hash(candidate),
        protocol=read_json(HERE / "screening-protocol.json"), tasks={task_id: task})


def execute(task_id, arm_id, output):
    from experiments.forge.gaussian_tasks import run_gaussian
    from experiments.forge.word_adapter import run_word
    from experiments.forge.adapters import _vector
    protocol, task, request = prepare(task_id)
    arm = next(row for row in protocol["arms"] if row["id"] == arm_id)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("commit scientific sources before execution")
    output.mkdir(parents=True, exist_ok=False)
    source = inspect_source(ROOT, extra_paths=[str(PROTOCOL.relative_to(ROOT)),
        str(Path(__file__).relative_to(ROOT)), str((HERE / "screening-protocol.json").relative_to(ROOT)), protocol["candidate_path"]])
    atomic_json(output / "source-manifest.json", source)
    atomic_json(output / "request.json", dict(request=request, protocol=protocol, arm=arm))
    allowance = task["resources"]["timeout_seconds"]
    def timeout(*_):
        raise TimeoutError("declared task allowance exhausted")
    previous_signal = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, allowance)
    started = time.monotonic()
    try:
        with construction_rng(0, "cuda:0"), torch.autograd.set_multithreading_enabled(arm["autograd_multithreading"]), FactorAudit(arm, output, task, protocol["smoothing_lambda"]) as audit:
            if task_id == "gaussian1d_smoke":
                raw = run_gaussian(request, task, output, "cuda:0", diagnostic=True)
            elif task_id == "five_word_joint_acquisition":
                raw = run_word(request, task, output, "cuda:0", capture_media=False, retain_scored_outputs=True)
            else:
                raw = _vector(request, task, output, "cuda:0", retain_scored_outputs=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_signal)
    elapsed = time.monotonic() - started
    raw["scope"] = "task_only_smooth_polar_factorial_diagnostic"
    atomic_json(output / "raw-result.json", raw)
    grade = raw.get("gaussian_grade") or grade_result(task, raw)
    factors = read_json(output / "factor-audit.json")
    flag = str(arm["autograd_multithreading"])
    if not factors["mode_counts"] or any(key.split(":")[-1] != flag for key in factors["mode_counts"]):
        raise RuntimeError("observed graph scheduling differs from arm")
    if factors["mode_counts"].get("step:" + flag) != task["execution"]["steps"]:
        raise RuntimeError("actual update count differs from task")
    compact = dict(schema_version=1, task=task_id, arm=arm, qualification_input=False,
        source_commit=source["origin_commit"], source_digest=source["digest"],
        protocol_sha256=file_hash(PROTOCOL), task_fingerprint=task_fingerprint(task),
        grade=grade, recipe=raw["recipe"], prior=raw["prior"],
        observations=raw["evidence"]["observations"],
        confirmations=raw["evidence"].get("confirmations", []),
        guards=raw["evidence"]["guards"], host=raw["evidence"]["host"],
        initialization=raw.get("initialization"), sampling_law=raw["evidence"]["sampling_law"],
        data_sha256=raw["evidence"].get("data_sha256", factors["word_data_sequence_sha256"]),
        initial_models=factors["initial_models"], mode_counts=factors["mode_counts"],
        cost=dict(wall_seconds=elapsed, reservation_seconds=allowance, **raw["cost"]),
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            deterministic=torch.are_deterministic_algorithms_enabled(), tf32=torch.backends.cuda.matmul.allow_tf32),
        artifacts={str(path.relative_to(output)): dict(sha256=file_hash(path), bytes=path.stat().st_size)
            for path in sorted(output.rglob("*")) if path.is_file()})
    atomic_json(output / "receipt.json", compact)
    print(json.dumps(dict(event="concluded", task=task_id, arm=arm_id,
        grade=grade.get("gate_status", grade.get("status")), wall_seconds=elapsed)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    execute(args.task, args.arm, args.output.resolve())


if __name__ == "__main__":
    main()

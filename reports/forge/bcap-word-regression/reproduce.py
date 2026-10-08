"""Bounded nonqualifying CUDA factorial of polar truncation and autograd scheduling.

Uses the existing Forge WordFixture/run_word and grader without changing package
defaults. The enabled scheduling arm bypasses only run_task's policy decorator;
the actual forward and optimizer calls are audited inside the requested scope.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from benchmarks.toy_audit.api_images import WordFixture
from benchmarks.toy_audit.api_run import render_gif
from benchmarks.toy_audit.reproducibility import construction_rng
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sources import inspect_source
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks, task_fingerprint
from experiments.forge.word_adapter import run_word, word_context
import particlegan.optim.dualnorm as dualnorm

PROTOCOL = Path(__file__).with_name("protocol.json")


class FactorAudit:
    def __init__(self, arm, output, checks):
        self.arm, self.output, self.checks = arm, output, set(checks) | {1, 2}
        self.step, self.calls, self.mode_counts = 0, 0, Counter()
        self.rank_rows, self.state_rows, self.handles = [], [], []
        self.data_digest = hashlib.sha256()
        self.original_step = WordFixture.step
        self.original_polar = dualnorm.polar_factor
        self.original_randint = torch.randint
        self.data_generator = None

    def __enter__(self):
        audit = self
        def randint(*args, **kwargs):
            result = audit.original_randint(*args, **kwargs)
            if kwargs.get("generator") is audit.data_generator:
                audit.data_digest.update(result.detach().cpu().numpy().tobytes())
            return result
        def step(fixture):
            audit.step = fixture.completed_steps + 1
            audit.calls = 0
            audit.data_generator = fixture.data_generator
            audit.mode_counts["step:" + str(torch.autograd.is_multithreading_enabled())] += 1
            if not audit.handles:
                def forward(module, inputs):
                    audit.mode_counts["forward:" + str(torch.autograd.is_multithreading_enabled())] += 1
                def optimizer(opt, args, kwargs):
                    audit.mode_counts["optimizer:" + str(torch.autograd.is_multithreading_enabled())] += 1
                for module in (fixture.G, fixture.E, fixture.D):
                    audit.handles.append(module.register_forward_pre_hook(forward))
                for opt in (fixture.opt_g, fixture.opt_d):
                    audit.handles.append(opt.register_step_pre_hook(optimizer))
            result = audit.original_step(fixture)
            if audit.step in audit.checks:
                audit.state_rows.append({"step": audit.step, "model_state_sha256": {
                    name: state_digest(module.state_dict()) for name, module in
                    (("G", fixture.G), ("E", fixture.E), ("D", fixture.D), ("prior", fixture.prior))},
                    "data_prefix_sha256": audit.data_digest.hexdigest()})
            return result
        def polar(matrix):
            audit.calls += 1
            audit.mode_counts["polar:" + str(torch.autograd.is_multithreading_enabled())] += 1
            original_dtype = matrix.dtype
            value = matrix if original_dtype in (torch.float32, torch.float64) else matrix.float()
            left, singular, right = torch.linalg.svd(value, full_matrices=False)
            threshold = max(value.shape) * torch.finfo(value.dtype).eps * singular[0]
            keep = singular > threshold
            update = ((left * keep) @ right if audit.arm["truncation"] else left @ right).to(original_dtype)
            if audit.step in audit.checks:
                audit.rank_rows.append({"step": audit.step, "call": audit.calls,
                    "shape": list(value.shape), "rank": int(keep.sum()), "available_rank": len(singular),
                    "removed": int((~keep).sum()), "threshold": float(threshold),
                    "singular_max": float(singular[0]), "singular_min": float(singular[-1]),
                    "gradient_sha256": state_digest(matrix), "update_sha256": state_digest(update)})
            return update
        WordFixture.step, dualnorm.polar_factor, torch.randint = step, polar, randint
        return self

    def __exit__(self, *exc):
        WordFixture.step, dualnorm.polar_factor, torch.randint = self.original_step, self.original_polar, self.original_randint
        for handle in self.handles:
            handle.remove()
        atomic_json(self.output / "factor-audit.json", {"mode_counts": dict(self.mode_counts),
            "rank_rows": self.rank_rows, "state_rows": self.state_rows,
            "data_sequence_sha256": self.data_digest.hexdigest(), "additional_rng_draws": 0})


def prepare(protocol):
    candidate = read_json(ROOT / protocol["candidate_path"])
    task = load_tasks(ROOT)[protocol["task"]]
    if file_hash(ROOT / protocol["candidate_path"]) != protocol["candidate_sha256"]:
        raise ValueError("frozen candidate changed")
    if task_fingerprint(task) != protocol["task_fingerprint"]:
        raise ValueError("frozen task changed")
    context = word_context({"candidate": candidate, "protocol": {"seed": 0}}, task, "cuda:0", root=ROOT)
    if stable_hash(context.recipe.to_dict()) != protocol["resolved_recipe_sha256"]:
        raise ValueError("global recipe changed")
    if (task["execution"]["steps"], task["evaluation"]["observations"],
        task["evaluation"]["minimum_stable_checks"], task["resources"]["timeout_seconds"]) != (20001, 24, 5, 900):
        raise ValueError("task budget or scoring law changed")
    return candidate, task


def execute(protocol, arm, output):
    output.mkdir(parents=True, exist_ok=False)
    candidate, task = prepare(protocol)
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("commit the execution sources before running")
    source = inspect_source(ROOT, extra_paths=[str(PROTOCOL.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)), protocol["candidate_path"]])
    atomic_json(output / "source-manifest.json", source)
    request = {"candidate": candidate, "protocol": {"seed": 0},
               "candidate_revision": stable_hash(candidate), "tasks": {task["id"]: task}}
    atomic_json(output / "request.json", {"request": request, "arm": arm, "protocol": protocol})
    def timeout(signum, frame):
        raise TimeoutError("900-second reserved diagnostic exhausted")
    old_signal = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 900)
    checks = [int(np.ceil(i * 20001 / 24)) for i in range(1, 25)]
    started = time.monotonic()
    try:
        with construction_rng(0, "cuda:0"), torch.autograd.set_multithreading_enabled(arm["autograd_multithreading"]), FactorAudit(arm, output, checks) as audit:
            # A step-zero media observation would consume an additional eval
            # draw and shift the original task's 24 scheduled sample batches.
            raw = run_word(request, task, output, "cuda:0", capture_media=False,
                           retain_scored_outputs=True)
            actual_mode = torch.autograd.is_multithreading_enabled()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_signal)
    elapsed = time.monotonic() - started
    # Explicit diagnostic scope; this adapter call never enters Forge qualification.
    raw["scope"] = "bcap_word_runtime_factorial_diagnostic_only"
    atomic_json(output / "raw-result.json", raw)
    grade = grade_result(task, raw)
    factor = read_json(output / "factor-audit.json")
    expected = str(arm["autograd_multithreading"])
    if any(key.split(":")[-1] != expected for key in factor["mode_counts"]):
        raise RuntimeError("autograd mode differed inside actual execution")
    compact = {"schema_version": 1, "id": arm["id"], "arm": arm,
        "scope": "task_only_nonqualifying_causal_diagnostic", "qualification_input": False,
        "source_commit": source["origin_commit"], "source_digest": source["digest"],
        "protocol_sha256": file_hash(PROTOCOL), "task_fingerprint": task_fingerprint(task),
        "resolved_recipe_sha256": stable_hash(raw["recipe"]), "grade": grade,
        "final_metrics": raw["evidence"]["live"], "observations": raw["evidence"]["observations"],
        "initialization": raw["initialization"], "host": raw["evidence"]["host"],
        "prior": raw["prior"], "sampling_law": raw["evidence"]["sampling_law"],
        "guards": raw["evidence"]["guards"], "data_sequence_sha256": factor["data_sequence_sha256"],
        "mode_counts": factor["mode_counts"], "actual_autograd_mode": actual_mode,
        "rank_summary": {"measured_matrix_updates": len(factor["rank_rows"]),
            "updates_with_removed_directions": sum(row["removed"] > 0 for row in factor["rank_rows"]),
            "removed_directions": sum(row["removed"] for row in factor["rank_rows"]),
            "available_directions": sum(row["available_rank"] for row in factor["rank_rows"])},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0), "device": "cuda:0", "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32": torch.backends.cuda.matmul.allow_tf32},
        "cost": {**raw["cost"], "wall_seconds": elapsed, "reservation_seconds": 900},
        "artifacts": {name: {"sha256": file_hash(output / name), "bytes": (output / name).stat().st_size}
            for name in ("source-manifest.json", "request.json", "raw-result.json", "state.pt", "observed-records.pt", "factor-audit.json")}}
    atomic_json(output / "compact-receipt.json", compact)
    print(json.dumps({"event": "concluded", "arm": arm, "grade": grade["gate_status"], "seconds": elapsed,
                      "rank_summary": compact["rank_summary"], "final": compact["final_metrics"]}), flush=True)


def render(arm, output):
    compact = read_json(output / "compact-receipt.json")
    records = torch.load(output / "observed-records.pt", map_location="cpu", weights_only=False)
    selected = [records[i] for i in np.rint(np.linspace(0, len(records) - 1, 9)).astype(int)]
    case = {"id": arm["id"], "goal": "Acquire five canonical words and reconstruct each correctly paired input",
            "scope": "Unchanged word gate; runtime factorial diagnostic only", "default_recipe": "BCAP/DualNorm",
            "default_steps": 20001, "eval_samples": 1024, "sampling": {"evaluation": compact["sampling_law"]}}
    render_gif(case, selected, output / "goal.gif", full_budget=True, requested_steps=20001, final_verdict=compact["grade"]["gate_status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--render")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/forge/bcap-word-regression")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    if len(protocol["arms"]) != 4 or protocol["maximum_runs"] != 4 or protocol["reserved_seconds"] != 3600:
        raise ValueError("factorial budget changed")
    if not args.run and not args.render:
        prepare(protocol)
        print(json.dumps({"event": "validated", "protocol": protocol["id"], "arms": protocol["arms"]}), flush=True)
        return
    arm = next(row for row in protocol["arms"] if row["id"] == (args.run or args.render))
    if args.run:
        execute(protocol, arm, args.output / arm["id"])
    else:
        render(arm, args.output / arm["id"])


if __name__ == "__main__":
    main()

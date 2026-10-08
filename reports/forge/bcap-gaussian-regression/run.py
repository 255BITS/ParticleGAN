"""Bounded, non-qualifying 2x2 diagnostic using the original public Forge task.

Production files are never changed. Historical semantics are installed only in
this process, explicitly recorded, and restored on exit. Saved diagnostic states
must not be admitted as ordinary Forge qualification or checkpoint parents.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest
from particlegan import GANTrainer
import particlegan.optim.dualnorm as dualnorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = {
    "current": (True, False),
    "old_polar_serial": (False, False),
    "truncated_parallel": (True, True),
    "historical": (False, True),
}


def historical_polar():
    binding = json.loads((HERE / "protocol.json").read_text())["historical_polar"]
    source = subprocess.check_output(
        ["git", "show", binding["revision"] + ":" + binding["path"]], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != binding["sha256"]:
        raise ValueError("historical polar source differs from declared bytes")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "polar_factor")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), binding["path"], "exec"), namespace)
    return namespace["polar_factor"]


@contextmanager
def factors(truncate, parallel, output):
    """Own both historical factors and audit the actual graph execution mode."""
    old_execute, old_state = GANTrainer._execute_step, GANTrainer.state_dict
    old_property, old_polar = GANTrainer.serial_backward, dualnorm.polar_factor
    old_svd, old_grad = torch.linalg.svd, torch.autograd.grad
    modes = {name: Counter() for name in ("update", "forward", "autograd_grad")}
    ranks, trajectories = Counter(), []
    installed = set()

    def execute(self, real, *, generator_real=None, collect_stats=False):
        if id(self) not in installed:
            for model in (self.G, self.D):
                model.register_forward_pre_hook(
                    lambda *_: modes["forward"].update([torch.autograd.is_multithreading_enabled()]))
            installed.add(id(self))
        try:
            with torch.autograd.set_multithreading_enabled(parallel):
                modes["update"].update([torch.autograd.is_multithreading_enabled()])
                result = self._step(real, generator_real=generator_real, collect_stats=collect_stats)
        except Exception:
            self.policy.abort_step()
            raise
        trajectories.append({"step": self.completed_steps,
            "models_sha256": state_digest({name: model.state_dict()
                for name, model in (("G", self.G), ("D", self.D), ("prior", self.prior))}),
            "gradients_sha256": state_digest({name: [p.grad for p in model.parameters()]
                for name, model in (("G", self.G), ("D", self.D), ("prior", self.prior))})})
        return result

    def state(self):
        result = old_state(self)
        # The checkpoint must record the actual diagnostic runtime constraint.
        result["serial_backward"] = not parallel
        return result

    def svd(matrix, *args, **kwargs):
        result = old_svd(matrix, *args, **kwargs)
        values = result[1].detach().cpu().tolist()
        threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * values[0]
        rank = sum(value > threshold for value in values)
        ranks[(tuple(matrix.shape), rank, len(values))] += 1
        return result

    def grad(*args, **kwargs):
        modes["autograd_grad"].update([torch.autograd.is_multithreading_enabled()])
        return old_grad(*args, **kwargs)

    GANTrainer._execute_step = execute
    GANTrainer.state_dict = state
    GANTrainer.serial_backward = property(lambda _: not parallel)
    dualnorm.polar_factor = old_polar if truncate else historical_polar()
    torch.linalg.svd, torch.autograd.grad = svd, grad
    try:
        with torch.autograd.set_multithreading_enabled(parallel):
            yield modes, ranks, trajectories
    finally:
        GANTrainer._execute_step, GANTrainer.state_dict = old_execute, old_state
        GANTrainer.serial_backward, dualnorm.polar_factor = old_property, old_polar
        torch.linalg.svd, torch.autograd.grad = old_svd, old_grad
        atomic_json(output / "trajectory.json", trajectories)


def run_arm(name, output):
    from experiments.forge.gaussian_tasks import run_gaussian
    declared = json.loads((HERE / "inputs.json").read_text())
    protocol = json.loads((HERE / "protocol.json").read_text())
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != protocol["gpu_model"]:
        raise ValueError("exactly one declared CUDA device must be visible")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    output.mkdir(parents=True, exist_ok=False)
    truncate, parallel = ARMS[name]
    began = time.monotonic()
    with factors(truncate, parallel, output) as (modes, ranks, trajectories):
        raw = run_gaussian(declared, declared["tasks"]["gaussian1d_smoke"], output,
                           "cuda:0", diagnostic=True)
    elapsed = time.monotonic() - began
    if elapsed > protocol["per_arm_seconds"]:
        raise TimeoutError("diagnostic arm exceeded its complete allowance")
    grade, evidence = raw["gaussian_grade"], raw["evidence"]
    if any(set(counter) != {parallel} for counter in modes.values()):
        raise ValueError("actual forward/higher-order/update scheduling did not match arm")
    summary = {
        "arm": name, "scope": "non_qualifying_causal_diagnostic", "qualification": False,
        "truncation": truncate, "autograd_multithreading_enabled": parallel,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_sha256": file_hash(Path(__file__)), "protocol_sha256": file_hash(HERE / "protocol.json"),
        "inputs_sha256": file_hash(HERE / "inputs.json"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "wall_seconds": elapsed,
        "grade": grade, "guards": evidence["guards"], "data_sha256": evidence["data_sha256"],
        "host": evidence["host"], "recipe": raw["recipe"], "prior": raw["prior"],
        "observations": evidence["observations"], "confirmations": evidence["confirmations"],
        "mode_audit": {key: {str(flag): count for flag, count in value.items()} for key, value in modes.items()},
        "singular_ranks": [{"shape": list(shape), "rank": rank, "directions": directions,
                            "calls": count} for (shape, rank, directions), count in sorted(ranks.items())],
        "trajectory_sha256": file_hash(output / "trajectory.json"),
        "initial_state_sha256": file_hash(output / "evaluator/initial-state.pt"),
        "final_state_sha256": file_hash(output / "evaluator/state.pt"),
        "observer_sha256": file_hash(output / "evaluator/observed-samples.pt"),
        "model_endpoint_sha256": trajectories[-1]["models_sha256"],
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"arm": name, "grade": grade["status"], "wall_seconds": elapsed,
                      "endpoint": grade["metrics"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_arm(args.arm, args.output.resolve())

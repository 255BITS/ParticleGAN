"""Per-tensor adaptive gradient descent, without momentum or Adam moments."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from . import sgd_study
from .study import TRAIN_TASKS, write_json

FRACTIONS = (.001, .003, .01, .03)
PARAMETER_FLOOR = .01
GRADIENT_EPSILON = 1e-12
MIN_BASE_RATIO = 1e-4
MAX_BASE_RATIO = 100.


def grouped_sgd(parameters, lr):
    return torch.optim.SGD([{"params": [p], "lr": lr} for p in parameters],
                           lr=lr, momentum=0., dampening=0., weight_decay=0.,
                           nesterov=False, foreach=False)


class RelativeStepController:
    """Choose a scalar step for each parameter tensor using its current RMS."""
    def __init__(self, weights, total_steps, *, g_fraction, d_fraction, ablation="none", interval=20):
        if ablation != "none":
            raise ValueError("this fixed rule has no learned-feature ablation")
        self.fractions = {"g": g_fraction, "d": d_fraction}
        if any(not math.isfinite(v) or v <= 0 for v in self.fractions.values()):
            raise ValueError("target fractions must be finite and positive")
        self.total_steps, self.interval = total_steps, interval
        self.state, self.trace, self.seconds = {}, [], 0.

    @torch.no_grad()
    def step(self, optimizer, completed_updates, *, role, schedule="constant"):
        start = time.perf_counter()
        if not isinstance(optimizer, torch.optim.SGD):
            raise ValueError("per-tensor raw descent requires SGD")
        if role not in self.fractions or not 0 <= completed_updates < self.total_steps:
            raise ValueError("invalid role or completed-update count")
        if schedule != "constant":
            raise ValueError("the initial per-tensor study uses fixed target fractions")
        if optimizer not in self.state:
            self.state[optimizer] = dict(rates=[g["lr"] for g in optimizer.param_groups], role=role, last=-1)
        state = self.state[optimizer]
        if state["role"] != role or completed_updates <= state["last"]:
            raise ValueError("role changed or step repeated")
        state["last"] = completed_updates
        values = []
        for group, base_rate in zip(optimizer.param_groups, state["rates"]):
            if len(group["params"]) != 1 or any(group.get(k, 0) for k in ("momentum", "dampening", "weight_decay", "nesterov", "maximize")):
                raise ValueError("requires one tensor per group and no momentum or weight decay")
            parameter = group["params"][0]
            if parameter.grad is None:
                continue
            # Float64 accumulation avoids interpreting a finite, large float32
            # gradient as nonfinite merely because its square overflows.
            rms = lambda value: float(torch.linalg.vector_norm(value.detach().double()) / math.sqrt(value.numel()))
            grad_rms, parameter_rms = rms(parameter.grad), rms(parameter)
            if not math.isfinite(grad_rms) or not math.isfinite(parameter_rms):
                raise FloatingPointError("nonfinite per-tensor descent attributes")
            proposed = self.fractions[role] * max(parameter_rms, PARAMETER_FLOOR) / max(grad_rms, GRADIENT_EPSILON)
            alpha = max(base_rate * MIN_BASE_RATIO, min(base_rate * MAX_BASE_RATIO, proposed))
            group["lr"] = alpha
            if completed_updates % self.interval == 0:
                values.append(dict(shape=list(parameter.shape), gradient_rms=grad_rms, parameter_rms=parameter_rms,
                                   proposed_alpha=proposed, alpha=alpha, base_lr=base_rate,
                                   raw_update_rms=alpha * grad_rms,
                                   clipped_low=proposed < base_rate * MIN_BASE_RATIO,
                                   clipped_high=proposed > base_rate * MAX_BASE_RATIO))
        if completed_updates % self.interval == 0:
            self.trace.append(dict(step=completed_updates, role=role, target_fraction=self.fractions[role], tensors=values))
        self.seconds += time.perf_counter() - start


def run_episode(task, g_lr, d_lr, g_fraction, d_fraction):
    def controller(weights, total_steps, **kwargs):
        return RelativeStepController(weights, total_steps, g_fraction=g_fraction, d_fraction=d_fraction, **kwargs)
    with patch.object(sgd_study, "make_sgd", grouped_sgd), patch.object(sgd_study, "RawGradientLRAdapter", controller):
        row = sgd_study.run_episode(task, g_lr, d_lr)
    row.update(optimizer="per_tensor_adaptive_gradient_descent", g_fraction=g_fraction, d_fraction=d_fraction)
    return row


def run(output, sgd_output):
    output, sgd_output = Path(output), Path(sgd_output)
    if (output / "sweep.json").exists():
        raise FileExistsError("refusing to overwrite per-tensor sweep")
    torch.set_num_threads(1)
    prior_sweep = json.loads((sgd_output / "sweep.json").read_text())
    base = prior_sweep["best_constant"]
    protocol = sgd_study.fingerprint()
    protocol.update(version="per-tensor-relative-descent-v1", optimizer="one SGD group per tensor; no momentum, decay, or Adam state",
                    action="alpha=clip(target_role*max(parameter_RMS,.01)/max(gradient_RMS,1e-12),base_LR*1e-4,base_LR*100)",
                    fractions=FRACTIONS, base_rates={"g": base["g_lr"], "d": base["d_lr"]},
                    parameter_floor=PARAMETER_FLOOR, gradient_epsilon=GRADIENT_EPSILON,
                    base_ratio_bounds=[MIN_BASE_RATIO, MAX_BASE_RATIO], observation_interval=1, trace_interval=20,
                    preceding_sweep_sha256=hashlib.sha256((sgd_output / "sweep.json").read_bytes()).hexdigest())
    root = Path(__file__).resolve().parents[2]
    protocol["source_sha256"][str(Path(__file__).relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(protocol=protocol, rows=[], controls={"sgd_constant":base, "sgd_cosine":prior_sweep["best_cosine"],
                                                       "adam_cosine":prior_sweep["controls"]})
    write_json(output / "sweep.json", report)
    for index, (g_fraction, d_fraction) in enumerate(itertools.product(FRACTIONS, FRACTIONS)):
        name = f"relative_{index:02d}"
        episodes = []
        for task in TRAIN_TASKS:
            row = run_episode(task, base["g_lr"], base["d_lr"], g_fraction, d_fraction)
            sgd_study.save_episode(output, name, row)
            episodes.append(row)
        report["rows"].append(dict(name=name, g_fraction=g_fraction, d_fraction=d_fraction,
                                   objective=sum(r["objective"] for r in episodes) / 2,
                                   episodes=[sgd_study.summary(r) for r in episodes]))
        report["best"] = min(report["rows"], key=lambda r:r["objective"])
        write_json(output / "sweep.json", report)
    print("PER-TENSOR SWEEP COMPLETE", report["best"]["name"], report["best"]["objective"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sgd-output", type=Path, required=True)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(args.output, args.sgd_output)

"""Causal RMS memory and target-fraction decay at fixed selected fractions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch

from particlegan import learning_rate_scale
from . import sgd_study
from .relative_sgd_study import (RelativeStepController, grouped_sgd, PARAMETER_FLOOR,
                                 GRADIENT_EPSILON, MIN_BASE_RATIO, MAX_BASE_RATIO)
from .study import TRAIN_TASKS, write_json

CONDITIONS = (("current_constant", None, None), ("current_cosine0", None, 0.),
              ("current_cosine03", None, .3), ("current_cosine06", None, .6),
              ("rms09_constant", .9, None), ("rms09_cosine06", .9, .6),
              ("rms099_constant", .99, None), ("rms099_cosine06", .99, .6))


class SmoothedRelativeController(RelativeStepController):
    def __init__(self, *args, rms_beta=None, cosine_start=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rms_beta, self.cosine_start = rms_beta, cosine_start
        self.square_memory = {}

    @torch.no_grad()
    def step(self, optimizer, completed_updates, *, role, schedule="constant"):
        started, previous_seconds = time.perf_counter(), self.seconds
        original_fractions = self.fractions
        scale = (1. if self.cosine_start is None else
                 learning_rate_scale(completed_updates, self.total_steps, self.cosine_start, .05))
        self.fractions = {key: value * scale for key, value in original_fractions.items()}
        try:
            super().step(optimizer, completed_updates, role=role, schedule=schedule)
            if self.rms_beta is not None:
                memory = self.square_memory.setdefault(optimizer, {})
                traced_index = 0
                for index, (group, base_rate) in enumerate(zip(optimizer.param_groups, self.state[optimizer]["rates"])):
                    parameter = group["params"][0]
                    if parameter.grad is None:
                        continue
                    rms = lambda value: float(torch.linalg.vector_norm(value.detach().double()) / math.sqrt(value.numel()))
                    current, param = rms(parameter.grad), rms(parameter)
                    mean_square = (current * current if index not in memory else
                                   self.rms_beta * memory[index] + (1 - self.rms_beta) * current * current)
                    memory[index] = mean_square
                    denominator = math.sqrt(mean_square)
                    proposed = self.fractions[role] * max(param, PARAMETER_FLOOR) / max(denominator, GRADIENT_EPSILON)
                    alpha = max(base_rate * MIN_BASE_RATIO, min(base_rate * MAX_BASE_RATIO, proposed))
                    group["lr"] = alpha
                    if completed_updates % self.interval == 0:
                        self.trace[-1]["tensors"][traced_index].update(
                            running_gradient_rms=denominator, proposed_alpha=proposed, alpha=alpha,
                            raw_update_rms=alpha * current, clipped_low=proposed < base_rate * MIN_BASE_RATIO,
                            clipped_high=proposed > base_rate * MAX_BASE_RATIO)
                        traced_index += 1
            if completed_updates % self.interval == 0:
                self.trace[-1].update(rms_beta=self.rms_beta, cosine_start=self.cosine_start, fraction_multiplier=scale)
        finally:
            self.fractions = original_fractions
            self.seconds = previous_seconds + time.perf_counter() - started


def run_episode(task, base_rates, fractions, rms_beta, cosine_start):
    def controller(weights, total_steps, **kwargs):
        return SmoothedRelativeController(weights, total_steps, g_fraction=fractions["g"], d_fraction=fractions["d"],
                                          rms_beta=rms_beta, cosine_start=cosine_start, **kwargs)
    with patch.object(sgd_study, "make_sgd", grouped_sgd), patch.object(sgd_study, "RawGradientLRAdapter", controller):
        row = sgd_study.run_episode(task, base_rates["g"], base_rates["d"])
    row.update(optimizer="per_tensor_adaptive_gradient_descent", g_fraction=fractions["g"], d_fraction=fractions["d"],
               rms_beta=rms_beta, cosine_start=cosine_start)
    return row


def run(output, preceding):
    output, preceding = Path(output), Path(preceding)
    if (output / "followup.json").exists():
        raise FileExistsError("refusing to overwrite per-tensor follow-up")
    torch.set_num_threads(1)
    initial = json.loads((preceding / "sweep.json").read_text())
    chosen = initial["best"]
    fractions = {"g": chosen["g_fraction"], "d": chosen["d_fraction"]}
    base_rates = initial["protocol"]["base_rates"]
    protocol = dict(initial["protocol"])
    protocol.update(version="per-tensor-rms-decay-v1", selected_pair=chosen["name"], fractions=fractions,
                    conditions=CONDITIONS,
                    running_rms="One scalar EMA of squared gradient RMS per tensor, initialized to first observation; includes current gradient; no coordinate moments or update momentum.",
                    preceding_sha256=hashlib.sha256((preceding / "sweep.json").read_bytes()).hexdigest())
    root = Path(__file__).resolve().parents[2]
    protocol["source_sha256"][str(Path(__file__).relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(protocol=protocol, rows=[], parity={})
    write_json(output / "followup.json", report)
    for name, beta, start in CONDITIONS:
        episodes = []
        for task in TRAIN_TASKS:
            row = run_episode(task, base_rates, fractions, beta, start)
            sgd_study.save_episode(output, name, row)
            episodes.append(row)
            if name == "current_constant":
                reference = json.loads((preceding / "episodes" / f"{chosen['name']}_{task}.json").read_text())
                differences = [abs(a[k] - b[k]) for a, b in zip(row["curve"], reference["curve"])
                               for k in a if k != "seconds"]
                report["parity"][task] = dict(max_metric_difference=max(differences, default=None),
                                               same_curve_length=len(row["curve"]) == len(reference["curve"]))
        report["rows"].append(dict(name=name, rms_beta=beta, cosine_start=start,
                                   objective=sum(r["objective"] for r in episodes) / 2,
                                   episodes=[sgd_study.summary(r) for r in episodes]))
        report["best"] = min(report["rows"], key=lambda r:r["objective"])
        write_json(output / "followup.json", report)
    print("RMS/DECAY COMPLETE", report["best"]["name"], report["best"]["objective"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preceding", type=Path, required=True)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(args.output, args.preceding)

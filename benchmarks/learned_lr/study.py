"""Fit an optimizer-feedback LR policy on separate synthetic distributions.

Run: python -m benchmarks.learned_lr.study --output /tmp/learned-lr
Every attempted policy and episode curve is retained. Seed 0 is fixed; policy
perturbations change controller parameters, never the GAN initialization seed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from unittest.mock import patch

import torch

from particlegan import learning_rate_scale
from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.baseline import Candidate
from benchmarks.locked_shared.observation import recording, sustained
from .controller import FEATURES, OptimizerLRAdapter

STEPS = 1200
TRAIN_TASKS = ("ring4", "grid9")
HOLDOUT_TASKS = ("ring8", "ring8_scale_half", "ring8_r1r2")
BASE = Candidate("learned_lr_base", reg_coeff=3., reg_kappa=1.25,
                 particle_l2=0., lr_multiplier=.85)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def policy(weights):
    return {"schema": 1, "features": list(FEATURES), "row_roles": ["g", "d"],
            "weights": weights.tolist(), "interval": 20, "smoothing": .5,
            "multiplier_bounds": [.05, 2.]}


def task_means(name):
    if name == "grid9":
        return torch.cartesian_prod(torch.tensor([-2., 0., 2.]), torch.tensor([-2., 0., 2.]))
    count = 4 if name == "ring4" else 8
    radius = 1.5 if name == "ring8_scale_half" else 3.
    return mode_hold.ring_means(count, radius)


def run_episode(task, *, policy_dict=None, scheduler="learned", ablation="none"):
    """Run the unchanged ring-host update mechanics with a scoped LR hook."""
    started = time.monotonic()
    means = task_means(task)
    sigma = .035 if task == "ring8_scale_half" else .07
    candidate = BASE if task != "ring8_r1r2" else BASE.__class__(
        **{**asdict(BASE), "reg_arm": "a_r1r2", "reg_coeff": .1})
    adapter = OptimizerLRAdapter(policy_dict, STEPS, ablation=ablation) if scheduler == "learned" else None
    optimizer_order, rates, traces = [], {}, []
    def schedule(optimizer, completed_updates):
        if optimizer not in optimizer_order:
            optimizer_order.append(optimizer)
            rates[optimizer] = [group["lr"] for group in optimizer.param_groups]
        role = "d" if optimizer_order.index(optimizer) == 0 else "g"
        if adapter is not None:
            adapter.step(optimizer, completed_updates, role=role)
        else:
            scale = learning_rate_scale(completed_updates, STEPS, .6, .05) if scheduler == "cosine" else 1.
            for group, rate in zip(optimizer.param_groups, rates[optimizer]):
                group["lr"] = rate * scale
            if completed_updates % 20 == 0:
                traces.append(dict(step=completed_updates, role=role, multiplier=scale))
    original_diversity = mode_hold.diversity
    reference = mode_hold.sample_ring(means, mode_hold.EVAL_N, sigma, torch.Generator().manual_seed(101))
    angles = torch.arange(16) * math.pi / 16
    projections = torch.stack((angles.cos(), angles.sin()))
    projected_real = (reference @ projections).sort(dim=0).values
    data_scale = float(reference.square().mean().sqrt())
    def measure(samples, task_centers, *args, **kwargs):
        metrics = original_diversity(samples, task_centers, sigma=sigma, **kwargs)
        metrics["sw1"] = float(((samples @ projections).sort(dim=0).values - projected_real).abs().mean()) / data_scale
        return metrics
    with ExitStack() as stack:
        stack.enter_context(patch.object(mode_hold, "ring_means", lambda: means))
        stack.enter_context(patch.object(mode_hold, "SIGMA", sigma))
        stack.enter_context(patch.object(mode_hold, "LR", mode_hold.LR * candidate.lr_multiplier))
        stack.enter_context(patch.object(mode_hold, "diversity", measure))
        stack.enter_context(patch.object(mode_hold, "schedule_optimizer", schedule))
        observer = stack.enter_context(recording(STEPS))
        result = mode_hold.train_mode_hold(
            mode_hold.ModeHoldRecipe(particle_l2=0., steps=STEPS), seed=0,
            gan_factory=candidate.make_loss, cap_factory=candidate.make_penalty,
            diagnostics=False)
    curve = observer.curve
    # Distributional objective: average learning trajectory plus final-quarter
    # distance. No mode/HQ gate or EMA enters training-policy selection.
    objective = .5 * sum(p["sw1"] for p in curve) / len(curve) + .5 * sum(p["sw1"] for p in curve[-6:]) / 6
    return dict(task=task, scheduler=scheduler, ablation=ablation, objective=objective,
                seconds=time.monotonic() - started, live=curve[-1], ema=result,
                convergence=sustained(curve, [("modes", ">=", len(means)), ("hq", ">=", .9)],
                                      expected_steps=sorted(observer.steps)), curve=curve,
                controller_seconds=adapter.controller_seconds if adapter else 0.,
                actions=adapter.trace if adapter else traces)


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "particlegan").glob("*.py")),
             *sorted((root / "benchmarks" / "locked_shared").glob("*.py")),
             *sorted(Path(__file__).parent.glob("*.py"))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def train(output, generations=4, population=8):
    output = Path(output)
    if (output / "training.json").exists():
        raise FileExistsError("refusing to overwrite existing study")
    torch.set_num_threads(1)
    report = {"protocol": {"version": "generic-lr-v1", "seed": 0, "torch": torch.__version__,
               "python": platform.python_version(), "steps": STEPS, "training_tasks": TRAIN_TASKS,
               "held_out_tasks": HOLDOUT_TASKS, "candidate": asdict(BASE),
               "generations": generations, "population": population, "elite_count": 3,
               "model": {"z_dim": 4, "hidden": 96, "hidden_layers": 3, "critic_fourier": 3,
                         "particles": 12, "batch": 128, "lr": .002 * .85,
                         "betas": [0., .99], "data_sigma": .07},
               "objective": "mean over tasks of 0.5 mean trajectory normalized SW1 + 0.5 final-quarter normalized SW1",
               "selection": "training distributions only; no held-out evaluation until policy freeze",
               "source_sha256": source_fingerprint()}, "attempts": []}
    write_json(output / "training.json", report)
    stream = torch.Generator().manual_seed(0)
    mean = torch.zeros(2, 6, dtype=torch.float64)
    std = torch.tensor([.45, 1.1, .2, .25, .2, .2], dtype=torch.float64).repeat(2, 1)
    # First population includes flat LR and a time-only exponential schedule.
    # Neither imitates cosine targets; all policies optimize actual GAN rollouts.
    best = None
    for generation in range(generations):
        proposals = [mean.clone()]
        if generation == 0:
            initial_decay = mean.clone()
            initial_decay[:, 0], initial_decay[:, 1] = .3, -1.4
            proposals.append(initial_decay)
        while len(proposals) < population:
            proposals.append(mean + torch.randn(2, 6, generator=stream, dtype=torch.float64) * std)
        rows = []
        for index, weights in enumerate(proposals):
            label = f"generation{generation:02d}_candidate{index:02d}"
            episode_rows = []
            print(f"START {label}", flush=True)
            for task in TRAIN_TASKS:
                row = run_episode(task, policy_dict=policy(weights))
                episode_rows.append(row)
                write_json(output / "episodes" / f"{label}_{task}.json", row)
                print(f"{label} {task} objective={row['objective']:.6f} modes={row['live']['modes']} hq={row['live']['hq']:.4f} seconds={row['seconds']:.1f}", flush=True)
            objective = sum(row["objective"] for row in episode_rows) / len(episode_rows)
            attempt = dict(name=label, generation=generation, objective=objective,
                           policy=policy(weights), episodes=[{k: row[k] for k in ("task", "objective", "seconds", "live")} for row in episode_rows])
            report["attempts"].append(attempt)
            rows.append(attempt)
            if best is None or objective < best["objective"]:
                best = attempt
            report["best"] = best["name"]
            write_json(output / "training.json", report)
        elites = sorted(rows, key=lambda r: r["objective"])[:3]
        elite_weights = torch.tensor([row["policy"]["weights"] for row in elites], dtype=torch.float64)
        mean = .25 * mean + .75 * elite_weights.mean(0)
        std = (.25 * std + .75 * elite_weights.std(0, unbiased=False)).clamp_min(.04)
        print(f"GENERATION {generation} best={best['name']} objective={best['objective']:.6f}", flush=True)
    frozen = {**best["policy"], "selected_attempt": best["name"], "training_objective": best["objective"],
              "training_tasks": list(TRAIN_TASKS), "source_sha256": report["protocol"]["source_sha256"]}
    write_json(output / "policy.json", frozen)
    print(f"FROZEN {best['name']} objective={best['objective']:.6f}", flush=True)
    return frozen


def evaluate(output, policy_dict):
    output = Path(output)
    report = dict(protocol="generic-lr-v1 frozen held-out evaluation", policy=policy_dict, rows=[])
    for task in HOLDOUT_TASKS:
        for scheduler, ablation in (("constant", "none"), ("cosine", "none"), ("learned", "none"), ("learned", "time_only")):
            row = run_episode(task, policy_dict=policy_dict, scheduler=scheduler, ablation=ablation)
            report["rows"].append(row)
            write_json(output / "heldout.json", report)
            print(f"HOLDOUT {task} {scheduler}/{ablation} objective={row['objective']:.6f} modes={row['live']['modes']} hq={row['live']['hq']:.4f} stable={row['convergence']['stable_from_step']} seconds={row['seconds']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--train-only", action="store_true")
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    learned = train(args.output, args.generations, args.population)
    if not args.train_only:
        evaluate(args.output, learned)


if __name__ == "__main__":
    main()

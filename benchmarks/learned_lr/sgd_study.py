"""Research: tune plain SGD, then optionally learn causal scalar LR feedback.

No momentum, Adam moments, clipping, preconditioning, or weight decay enters
the inner update. All experiments use seed 0 and the original training tasks.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import traceback
from unittest.mock import patch

import torch

from particlegan import learning_rate_scale
from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.observation import recording, sustained
from .study import BASE, STEPS, TRAIN_TASKS, task_means, write_json, run_episode as adam_episode

FEATURES = ("bias", "progress", "log_gradient_ratio", "gradient_alignment", "log_parameter_ratio")
G_RATES = (.01, .05, .25, 1.25)
D_RATES = (.001, .005, .025, .125)


def make_sgd(parameters, lr):
    return torch.optim.SGD(parameters, lr=lr, momentum=0., dampening=0.,
                           weight_decay=0., nesterov=False, foreach=False)


class RawGradientLRAdapter:
    """Scalar LR feedback around a genuine raw-gradient SGD step."""
    def __init__(self, weights, total_steps, *, ablation="none", interval=20):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        if self.weights.shape != (2, 5) or not torch.isfinite(self.weights).all():
            raise ValueError("weights must be finite, shape [2, 5]")
        if ablation not in ("none", "time_only"):
            raise ValueError("invalid ablation")
        self.total_steps, self.interval, self.ablation = total_steps, interval, ablation
        self.state, self.trace, self.seconds = {}, [], 0.

    @torch.no_grad()
    def observe(self, optimizer, completed_updates, role):
        if not isinstance(optimizer, torch.optim.SGD):
            raise ValueError("raw-gradient controller requires SGD")
        for group in optimizer.param_groups:
            if any(group.get(k, 0) for k in ("momentum", "dampening", "weight_decay", "nesterov", "maximize")):
                raise ValueError("only unmodified, momentum-free SGD is supported")
        if role not in ("g", "d") or not 0 <= completed_updates < self.total_steps:
            raise ValueError("invalid optimizer role or update count")
        if optimizer not in self.state:
            self.state[optimizer] = dict(rates=[g["lr"] for g in optimizer.param_groups],
                                         log_scale=0., role=role, last=-1)
        state = self.state[optimizer]
        if state["role"] != role or completed_updates <= state["last"]:
            raise ValueError("role changed or update repeated")
        state["last"] = completed_updates
        params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        if not params:
            raise ValueError("backward must precede controller observation")
        gradient = torch.cat([p.grad.detach().flatten() for p in params])
        parameter = torch.cat([p.detach().flatten() for p in params])
        grad_rms, param_rms = [float(x.square().mean().sqrt()) for x in (gradient, parameter)]
        if not math.isfinite(grad_rms) or not math.isfinite(param_rms):
            raise FloatingPointError("nonfinite raw SGD attributes")
        if "baseline" not in state:
            state["baseline"] = max(grad_rms, 1e-12), max(param_rms, 1e-12)
        old = state.get("previous_gradient")
        alignment = 0. if old is None else float(torch.nn.functional.cosine_similarity(
            gradient.unsqueeze(0), old.unsqueeze(0), eps=1e-12))
        state["previous_gradient"] = gradient.clone()
        ratio = lambda value, base: max(-3., min(3., math.log(max(value, 1e-12) / base)))
        features = [1., completed_updates / self.total_steps,
                    ratio(grad_rms, state["baseline"][0]), alignment,
                    ratio(param_rms, state["baseline"][1])]
        if self.ablation == "time_only":
            features[2:] = [0., 0., 0.]
        return state, features, dict(gradient_rms=grad_rms, parameter_rms=param_rms)

    @torch.no_grad()
    def step(self, optimizer, completed_updates, *, role, schedule="learned"):
        started = time.perf_counter()
        if completed_updates % self.interval == 0:
            state, features, raw = self.observe(optimizer, completed_updates, role)
            if schedule == "learned":
                target = float(self.weights[0 if role == "g" else 1] @ torch.tensor(features, dtype=torch.float64))
                target = max(math.log(.05), min(math.log(4.), target))
                state["log_scale"] = .5 * (state["log_scale"] + target)
            self.trace.append(dict(step=completed_updates, role=role, features=features, **raw))
        state = self.state[optimizer]
        scale = (math.exp(state["log_scale"]) if schedule == "learned" else
                 learning_rate_scale(completed_updates, self.total_steps, .6, .05) if schedule == "cosine" else 1.)
        for group, rate in zip(optimizer.param_groups, state["rates"]):
            group["lr"] = rate * scale
        if completed_updates % self.interval == 0:
            self.trace[-1].update(multiplier=scale, learning_rates=[g["lr"] for g in optimizer.param_groups])
        self.seconds += time.perf_counter() - started
        return scale


def objective(row):
    """Sustained coverage/HQ first; late deficits then distributional distance."""
    if row.get("error") or not row.get("convergence", {}).get("complete"):
        return 1000.
    curve = row["curve"]
    def deficit(p):
        return (1 - p["modes"] / p["n_modes"]) + max(0., .9 - p["hq"]) / .9
    stable = row["convergence"]["stable_from_step"]
    distance = .5 * sum(p["sw1"] for p in curve) / len(curve) + .5 * sum(p["sw1"] for p in curve[-6:]) / 6
    return (20. * (stable is None) + 3. * sum(deficit(p) for p in curve[-5:]) / 5
            + 2. * deficit(curve[-1]) + .2 * distance + .1 * (stable / STEPS if stable else 1.))


def run_episode(task, g_lr, d_lr, *, schedule="constant", weights=None, ablation="none"):
    start = time.monotonic()
    means = task_means(task)
    sigma = .035 if task == "ring8_scale_half" else .07
    adapter = RawGradientLRAdapter(torch.zeros(2, 5) if weights is None else weights, STEPS, ablation=ablation)
    created, curve = [], []
    observer = None
    row = dict(task=task, optimizer="plain_sgd", schedule=schedule, ablation=ablation,
               g_lr=g_lr, d_lr=d_lr, status="RUNNING")
    def factory(parameters, *, lr, betas):
        role = "g" if not created else "d"
        opt = make_sgd(parameters, g_lr if role == "g" else d_lr)
        created.append((opt, role))
        return opt
    def schedule_optimizer(optimizer, step):
        role = next(role for opt, role in created if opt is optimizer)
        adapter.step(optimizer, step, role=role, schedule=schedule)
    reference = mode_hold.sample_ring(means, mode_hold.EVAL_N, sigma, torch.Generator().manual_seed(101))
    angles = torch.arange(16) * math.pi / 16
    projections = torch.stack((angles.cos(), angles.sin()))
    projected_real = (reference @ projections).sort(dim=0).values
    scale = float(reference.square().mean().sqrt())
    original_diversity = mode_hold.diversity
    def measure(samples, centers, *args, **kwargs):
        if not torch.isfinite(samples).all():
            raise FloatingPointError("nonfinite generated samples")
        metrics = original_diversity(samples, centers, sigma=sigma, **kwargs)
        metrics["sw1"] = float(((samples @ projections).sort(dim=0).values - projected_real).abs().mean()) / scale
        if not math.isfinite(metrics["sw1"]):
            raise FloatingPointError("nonfinite distributional metric")
        return metrics
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(mode_hold.torch.optim, "Adam", factory))
            stack.enter_context(patch.object(mode_hold, "ring_means", lambda: means))
            stack.enter_context(patch.object(mode_hold, "SIGMA", sigma))
            stack.enter_context(patch.object(mode_hold, "diversity", measure))
            stack.enter_context(patch.object(mode_hold, "schedule_optimizer", schedule_optimizer))
            observer = stack.enter_context(recording(STEPS))
            row["ema"] = mode_hold.train_mode_hold(
                mode_hold.ModeHoldRecipe(particle_l2=0., steps=STEPS), seed=0,
                gan_factory=BASE.make_loss, cap_factory=BASE.make_penalty, diagnostics=False)
        row["status"] = "COMPLETE"
    except Exception:
        row.update(status="ERROR", error=traceback.format_exc())
    curve = observer.curve if observer else []
    row.update(curve=curve, live=curve[-1] if curve else {}, actions=adapter.trace,
               controller_seconds=adapter.seconds, seconds=time.monotonic() - start,
               convergence=sustained(curve, [("modes", ">=", len(means)), ("hq", ">=", .9)],
                                     expected_steps=list(range(50, STEPS + 1, 50))))
    row["objective"] = objective(row)
    return row


def fingerprint():
    root = Path(__file__).resolve().parents[2]
    sources = [*sorted((root / "particlegan").glob("*.py")),
               *sorted((root / "benchmarks/locked_shared").glob("*.py")),
               Path(__file__), Path(__file__).with_name("study.py")]
    return dict(version="raw-sgd-v1", seed=0, threads=torch.get_num_threads(),
                torch=str(torch.__version__), python=platform.python_version(), platform=platform.platform(),
                steps=STEPS, train_tasks=TRAIN_TASKS, validation_only_tasks=["ring8", "full_nine_toy_suite"],
                formulation=asdict(BASE),
                model={"z_dim": 4, "hidden": 96, "hidden_layers": 3, "critic_fourier": 3,
                       "particles": 12, "batch_size": 128, "dtype": "float32", "device": "cpu",
                       "training_sigma": .07, "ema": .995, "prior_regularization": .05},
                controller={"features": FEATURES, "interval": 20, "smoothing": .5,
                            "multiplier_bounds": [.05, 4.], "log_ratio_bounds": [-3., 3.]},
                optimizer="SGD(momentum=0,dampening=0,weight_decay=0,nesterov=False,foreach=False)",
                objective="20*no_sustained_pass + 3*mean_last5_deficit + 2*final_deficit + .2*SW1_rollout + .1*stable_progress; deficit=missing_mode_fraction + max(0,.9-HQ)/.9",
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


def save_episode(output, label, row):
    write_json(output / "episodes" / f"{label}_{row['task']}.json", row)
    live = row["live"]
    print(f"{label} {row['task']} {row['status']} objective={row['objective']:.5f} modes={live.get('modes')} hq={live.get('hq')} stable={row['convergence']['stable_from_step']} sec={row['seconds']:.2f}", flush=True)


def summary(row):
    return {k: row[k] for k in ("task", "status", "objective", "live", "convergence", "seconds", "controller_seconds")}


def sweep(output):
    output = Path(output)
    if (output / "sweep.json").exists():
        raise FileExistsError("refusing to overwrite SGD sweep")
    torch.set_num_threads(1)
    report = dict(protocol=fingerprint(), rate_pairs=list(itertools.product(G_RATES, D_RATES)), rows=[], controls=[])
    write_json(output / "sweep.json", report)
    for schedule in ("constant", "cosine"):
        for index, (g_lr, d_lr) in enumerate(report["rate_pairs"]):
            label = f"sgd_{schedule}_{index:02d}"
            rows = []
            for task in TRAIN_TASKS:
                row = run_episode(task, g_lr, d_lr, schedule=schedule)
                save_episode(output, label, row)
                rows.append(row)
            report["rows"].append(dict(name=label, schedule=schedule, g_lr=g_lr, d_lr=d_lr,
                                       objective=sum(r["objective"] for r in rows) / 2,
                                       episodes=[summary(r) for r in rows]))
            write_json(output / "sweep.json", report)
    for task in TRAIN_TASKS:
        row = adam_episode(task, scheduler="cosine")
        row.update(optimizer="adam", status="COMPLETE")
        row["objective"] = objective(row)
        save_episode(output, "adam_cosine", row)
        report["controls"].append(summary(row))
        write_json(output / "sweep.json", report)
    report["best_constant"] = min((r for r in report["rows"] if r["schedule"] == "constant"), key=lambda r: r["objective"])
    report["best_cosine"] = min((r for r in report["rows"] if r["schedule"] == "cosine"), key=lambda r: r["objective"])
    write_json(output / "sweep.json", report)
    print("SWEEP COMPLETE", report["best_constant"]["name"], report["best_constant"]["objective"], report["best_cosine"]["name"], report["best_cosine"]["objective"], flush=True)


def fit(output, generations=4, population=8):
    output = Path(output)
    if (output / "fit.json").exists():
        raise FileExistsError("refusing to overwrite SGD fitting run")
    torch.set_num_threads(1)
    sweep_report = json.loads((output / "sweep.json").read_text())
    base = sweep_report["best_constant"]
    report = dict(protocol=fingerprint(), base_rates={"g": base["g_lr"], "d": base["d_lr"]},
                  generations=generations, population=population, proposals=[])
    write_json(output / "fit.json", report)
    mean = torch.zeros(2, 5, dtype=torch.float64)
    std = torch.tensor([.3, .7, .3, .2, .2], dtype=torch.float64).repeat(2, 1)
    rng = torch.Generator().manual_seed(0)
    best = None
    for generation in range(generations):
        proposals = [mean.clone()]
        if generation == 0:
            decay = mean.clone()
            decay[:, 0], decay[:, 1] = .3, -2.5
            proposals.append(decay)
        while len(proposals) < population:
            proposals.append(mean + torch.randn(2, 5, generator=rng, dtype=torch.float64) * std)
        rows = []
        for index, weights in enumerate(proposals):
            label = f"policy_{generation:02d}_{index:02d}"
            episodes = []
            for task in TRAIN_TASKS:
                row = run_episode(task, base["g_lr"], base["d_lr"], schedule="learned", weights=weights)
                save_episode(output, label, row)
                episodes.append(row)
            proposal = dict(name=label, weights=weights.tolist(),
                            objective=sum(r["objective"] for r in episodes) / 2,
                            episodes=[summary(r) for r in episodes])
            rows.append(proposal)
            report["proposals"].append(proposal)
            if best is None or proposal["objective"] < best["objective"]:
                best = proposal
            report["best"] = best
            write_json(output / "fit.json", report)
        elites = sorted(rows, key=lambda r: r["objective"])[:3]
        values = torch.tensor([r["weights"] for r in elites], dtype=torch.float64)
        mean = .25 * mean + .75 * values.mean(0)
        std = (.25 * std + .75 * values.std(0, unbiased=False)).clamp_min(.04)
    policy = dict(schema=1, optimizer="plain_sgd", features=list(FEATURES), row_roles=["g", "d"],
                  weights=best["weights"], interval=20, multiplier_bounds=[.05, 4.], smoothing=.5,
                  base_rates=report["base_rates"], selected_attempt=best["name"], objective=best["objective"],
                  source_sha256=report["protocol"]["source_sha256"])
    write_json(output / "sgd_policy.json", policy)
    print("FROZEN", best["name"], best["objective"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sweep", "fit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (sweep if args.command == "sweep" else fit)(args.output)


if __name__ == "__main__":
    main()

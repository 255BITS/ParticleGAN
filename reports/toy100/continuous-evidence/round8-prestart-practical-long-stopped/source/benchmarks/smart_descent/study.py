"""Development search and predeclared fresh transfer for gradient feedback.

python -u -m benchmarks.smart_descent.study --output /tmp/smart-descent-v2
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch

from benchmarks.locked_shared import baseline, mode_hold
from benchmarks.locked_shared.observation import recording, sustained
from benchmarks.learned_lr_evaluation import control_host_schedules
from .controller import FEATURES, GradientFeedback, control_regularization

BASE = baseline.Candidate("smart_descent", reg_coeff=3., reg_kappa=1.25,
                          particle_l2=0., lr_multiplier=.85)
# Declared before the first run; evaluated only after the development winner is frozen.
TRANSFER = [
    dict(name="ring6_width64", kind="ring", modes=6, radius=2.5, sigma=.07,
         hidden=64, layers=2, fourier=2, steps=1600, particles=12, arm="b_cap", coeff=3.),
    dict(name="grid16_width128", kind="grid", side=4, spacing=1.25, sigma=.07,
         hidden=128, layers=3, fourier=2, steps=1600, particles=32, arm="b_cap", coeff=3.),
    dict(name="ellipse8_r1r2", kind="ellipse", modes=8, radius=3., sigma=.07,
         hidden=64, layers=3, fourier=3, steps=1600, particles=16, arm="a_r1r2", coeff=.1),
]


def policy(weights, schedule="cosine"):
    return dict(version=2, features=list(FEATURES), roles=["g", "d"],
                actions=["lr_log_multiplier", "next_regularization_log_multiplier"],
                weights=weights.tolist(), interval=5, schedule=schedule)


def fingerprint():
    result = baseline.protocol()
    root = Path(__file__).resolve().parents[2]
    for path in [*Path(__file__).parent.glob("*.py"), root / "benchmarks/learned_lr_evaluation.py"]:
        result["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result.update(torch_git_revision=torch.version.git_version, torch_build=torch.__config__.show(),
                  cpu_capability=torch.backends.cpu.get_cpu_capability())
    return result


def run_toy(name, card, *, ablation="none"):
    controller = GradientFeedback(card, baseline.BUDGETS[name], ablation=ablation)
    started = time.perf_counter()
    try:
        with control_host_schedules(controller), control_regularization(controller):
            result = baseline.run_toy(name, BASE)
        json.dumps(result, allow_nan=False)
    except Exception:
        result = dict(error=traceback.format_exc())
    result.update(seconds=time.perf_counter() - started, controller_seconds=controller.controller_seconds,
                  actions=controller.trace)
    return result


def ring_objective(result):
    if "error" in result or not result.get("convergence", {}).get("complete"):
        return 1000.
    curve = result["observations"]
    deficit = lambda row: (1 - row["modes"] / 8) + max(0., .9 - row["hq"]) / .9
    cost = sum(deficit(p) for p in curve) / len(curve)
    cost += 2 * sum(deficit(p) for p in curve[-5:]) / 5 + 3 * deficit(curve[-1])
    confirmed = result["convergence"]["confirmed_step"]
    return cost + (confirmed / 1200 if confirmed is not None else 2.)


def row_summary(row):
    score = baseline.score_row(row, {})
    stable = sum(t.get("convergence", {}).get("stable_from_step") is not None for t in row["toys"].values())
    confirms = [t.get("convergence", {}).get("confirmed_step", None) for t in row["toys"].values()]
    total_fraction = sum((t.get("convergence", {}).get("confirmed_step") or baseline.BUDGETS[n] * 2) / baseline.BUDGETS[n]
                         for n, t in row["toys"].items()) / max(len(confirms), 1)
    return dict(bounds=score["passed_metrics"], stable=stable, complete=len(row["toys"]) == 9,
                mean_confirmation_fraction=total_fraction,
                seconds=sum(t["seconds"] for t in row["toys"].values()))


def render(report, output):
    lines = ["# Smart descent v2 — development search", "",
             "Previously inspected toys are development data. Ring failures are screened before the remaining eight toys; "
             "screened rows are incomplete and cannot receive an overall PASS. Live weights determine every score; EMA stays separate.", "",
             "| Candidate | Evaluated toys | Live bounds | Sustained toys | Ring modes / HQ | Ring confirmation | Seconds |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in report["rows"]:
        summary = row_summary(row)
        ring = row["toys"]["mode_hold"]
        live = ring.get("live", {})
        confirmed = ring.get("convergence", {}).get("confirmed_step")
        lines.append(f"| {row['name']} | {len(row['toys'])}/9 | {summary['bounds']}/29 | {summary['stable']}/9 | "
                     f"{live.get('modes', '—')}/8 / {live.get('hq', 0):.2%} | {confirmed or '—'} | {summary['seconds']:.2f} |")
    lines += ["", "All attempted policies, full curves, actions, errors and source hashes: [search.json](search.json).",
              "Timing is one CPU observation including measurement and controller overhead.", ""]
    (output / "README.md").write_text("\n".join(lines))


def search(output, generations=3, population=12):
    if output.exists():
        raise FileExistsError("use a new output directory")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    report = dict(protocol=fingerprint(), base=asdict(BASE), fresh_transfer=TRANSFER,
                  split="All previously inspected tasks are development data; transfer specs frozen before this search.",
                  search=dict(generations=generations, population=population, seed=0, elite_count=3,
                              selection="CEM ring coverage/HQ/stability objective; final winner requires all nine toys sustained and 29/29 bounds, then lowest mean normalized confirmation step."),
                  rows=[])
    def save():
        baseline.write_json(output / "search.json", report)
        render(report, output)
    save()
    stream = torch.Generator().manual_seed(0)
    mean = torch.zeros(2, 2, len(FEATURES), dtype=torch.float64)
    std = torch.full_like(mean, .06)
    std[:, 1] = .035
    best_ring = None
    for generation in range(generations):
        proposals = [mean.clone()]
        if best_ring is not None:
            proposals.append(torch.tensor(best_ring["policy"]["weights"], dtype=torch.float64))
        while len(proposals) < population:
            proposals.append(mean + torch.randn(mean.shape, generator=stream, dtype=mean.dtype) * std)
        generation_rows = []
        for index, weights in enumerate(proposals):
            name = "cosine_control" if generation == index == 0 else f"g{generation:02d}_p{index:02d}"
            card = policy(weights)
            # Identical policies are reused explicitly, without rerunning training or timing.
            duplicate = next((r for r in report["rows"] if r["policy"] == card), None)
            if duplicate:
                generation_rows.append(duplicate)
                continue
            print(f"START {name}", flush=True)
            row = dict(name=name, policy=card, config=asdict(BASE), toys={})
            row["toys"]["mode_hold"] = run_toy("mode_hold", card)
            row["ring_objective"] = ring_objective(row["toys"]["mode_hold"])
            report["rows"].append(row)
            save()
            ring = row["toys"]["mode_hold"]
            if ring.get("convergence", {}).get("stable_from_step") is not None:
                for toy in baseline.BUDGETS:
                    if toy == "mode_hold":
                        continue
                    row["toys"][toy] = run_toy(toy, card)
                    save()
            generation_rows.append(row)
            if best_ring is None or row["ring_objective"] < best_ring["ring_objective"]:
                best_ring = row
            print(json.dumps(dict(event="DONE", name=name, ring=ring.get("live"),
                                  objective=row["ring_objective"], **row_summary(row))), flush=True)
        elites = sorted(generation_rows, key=lambda r: r["ring_objective"])[:3]
        values = torch.tensor([r["policy"]["weights"] for r in elites], dtype=torch.float64)
        mean = .3 * mean + .7 * values.mean(0)
        std = (.3 * std + .7 * values.std(0, unbiased=False)).clamp_min(.015)
    valid = [r for r in report["rows"] if row_summary(r)["bounds"] == 29 and row_summary(r)["stable"] == 9]
    best = min(valid, key=lambda r: row_summary(r)["mean_confirmation_fraction"]) if valid else None
    if best is not None:
        frozen = dict(policy=best["policy"], selected=best["name"], summary=row_summary(best),
                      fresh_transfer=TRANSFER, selection="development only", source_sha256=report["protocol"]["source_sha256"])
        baseline.write_json(output / "frozen.json", frozen)
        report["selected"] = best["name"]
        print(f"FROZEN {best['name']} {row_summary(best)}", flush=True)
    save()
    return report


def transfer_episode(spec, card, *, ablation="none"):
    if spec["kind"] == "grid":
        axis = (torch.arange(spec["side"]) - (spec["side"] - 1) / 2) * spec["spacing"]
        means = torch.cartesian_prod(axis, axis)
    else:
        means = mode_hold.ring_means(spec["modes"], spec["radius"])
        if spec["kind"] == "ellipse":
            means[:, 1] *= .5
    cfg = baseline.Candidate(**{**asdict(BASE), "reg_arm": spec["arm"], "reg_coeff": spec["coeff"]})
    controller = GradientFeedback(card, spec["steps"], ablation=ablation)
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            for key, value in {"ring_means": lambda: means, "SIGMA": spec["sigma"], "HIDDEN": spec["hidden"],
                               "N_HIDDEN": spec["layers"], "FOURIER": spec["fourier"], "LR": .002 * .85}.items():
                stack.enter_context(patch.object(mode_hold, key, value))
            original_diversity = mode_hold.diversity
            def measure(samples, centers, **kwargs):
                return original_diversity(samples, centers, sigma=spec["sigma"], **kwargs)
            stack.enter_context(patch.object(mode_hold, "diversity", measure))
            stack.enter_context(control_host_schedules(controller))
            stack.enter_context(control_regularization(controller))
            observer = stack.enter_context(recording(spec["steps"]))
            raw = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(n_particles=spec["particles"], particle_l2=0., steps=spec["steps"]),
                                           seed=0, gan_factory=cfg.make_loss, cap_factory=cfg.make_penalty, diagnostics=True)
        result = dict(live=raw["live"], ema={k: raw[k] for k in ("modes", "hq", "effective_modes")},
                      observations=observer.curve,
                      convergence=sustained(observer.curve, [("modes", ">=", len(means)), ("hq", ">=", .9)], expected_steps=observer.steps))
        json.dumps(result, allow_nan=False)
    except Exception:
        result = dict(error=traceback.format_exc())
    result.update(seconds=time.perf_counter() - started, controller_seconds=controller.controller_seconds,
                  actions=controller.trace)
    return result


def evaluate(output):
    torch.set_num_threads(1)
    frozen = json.loads((output / "frozen.json").read_text())
    if (output / "evaluation.json").exists():
        raise FileExistsError("evaluation already exists")
    cards = [("cosine", policy(torch.zeros(2, 2, 5)), "none"),
             ("feedback", frozen["policy"], "none"), ("bias_only", frozen["policy"], "bias_only"),
             ("lr_only", frozen["policy"], "lr_only"), ("reg_only", frozen["policy"], "reg_only")]
    report = dict(protocol=fingerprint(), frozen=frozen, rows=[])
    for name, card, ablation in cards:
        row = dict(name=name, policy=card, ablation=ablation, config=asdict(BASE), toys={}, transfer={})
        report["rows"].append(row)
        for toy in baseline.BUDGETS:
            row["toys"][toy] = run_toy(toy, card, ablation=ablation)
            baseline.write_json(output / "evaluation.json", report)
        for spec in frozen["fresh_transfer"]:
            row["transfer"][spec["name"]] = transfer_episode(spec, card, ablation=ablation)
            baseline.write_json(output / "evaluation.json", report)
            print(json.dumps(dict(event="TRANSFER", controller=name, task=spec["name"],
                                  live=row["transfer"][spec["name"]].get("live"),
                                  convergence=row["transfer"][spec["name"]].get("convergence"))), flush=True)
        print(json.dumps(dict(event="EVALUATED", name=name, **row_summary(row))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    evaluate(args.output) if args.evaluate else search(args.output, args.generations, args.population)


if __name__ == "__main__":
    main()

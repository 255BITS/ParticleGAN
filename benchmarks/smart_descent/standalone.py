"""Fit feedback directly without a clock or an external LR schedule."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import baseline
from . import study
from .refine import objective

# Prior transfer cases have now been inspected; these two are newly reserved.
TRANSFER = [
    dict(name="ring5_width80", kind="ring", modes=5, radius=2., sigma=.06,
         hidden=80, layers=2, fourier=2, steps=1400, particles=10, arm="b_cap", coeff=3.),
    dict(name="ellipse6_width128", kind="ellipse", modes=6, radius=3., sigma=.08,
         hidden=128, layers=2, fourier=2, steps=1800, particles=18, arm="a_r1r2", coeff=.1),
]


def run(output, warm_start, generations=3, population=8):
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError("use a new output directory")
    output.mkdir(parents=True)
    warm = json.loads(warm_start.read_text())["policy"]
    if any(value != 0 for role in warm["weights"] for value in role[1]):
        raise ValueError("LR-only search requires zero regularization weights in its warm start")
    report = dict(protocol=study.fingerprint(), base=study.asdict(study.BASE), fresh_transfer=TRANSFER,
                  split="Full nine-host development set; original three transfer cases already inspected, not reused as fresh. New two-case transfer fixed before this search.",
                  search=dict(generations=generations, population=population, perturbation_rng=1731, gan_seed=0,
                              objective="50*failed bounds + 20*non-sustained toys + mean normalized confirmation step",
                              scope="Adam with constant initial rates and learned LR feedback; no time input, cosine, regularization actions or momentum changes"), rows=[])
    def save():
        baseline.write_json(output / "search.json", report)
        study.render(report, output)
    save()
    stream = torch.Generator().manual_seed(1731)
    mean = torch.zeros(2, 2, 5, dtype=torch.float64)
    std = torch.zeros_like(mean)
    std[:, 0] = .045
    for generation in range(generations):
        proposals = [mean.clone()]
        if generation == 0:
            proposals.append(torch.tensor(warm["weights"], dtype=mean.dtype))
        elif report["rows"]:
            best = min(report["rows"], key=objective)
            proposals.append(torch.tensor(best["policy"]["weights"], dtype=mean.dtype))
        while len(proposals) < population:
            proposals.append(mean + torch.randn(mean.shape, generator=stream, dtype=mean.dtype) * std)
        generation_rows = []
        for index, weights in enumerate(proposals):
            card = study.policy(weights, "constant")
            duplicate = next((r for r in report["rows"] if r["policy"] == card), None)
            if duplicate:
                generation_rows.append(duplicate)
                continue
            name = "constant_control" if generation == index == 0 else f"standalone_g{generation:02d}_p{index:02d}"
            row = dict(name=name, policy=card, config=study.asdict(study.BASE), toys={}, ring_objective=1000.)
            print(f"START {name}", flush=True)
            for toy in ["mode_hold", *(n for n in baseline.BUDGETS if n != "mode_hold")]:
                row["toys"][toy] = study.run_toy(toy, card)
                baseline.write_json(output / "in_progress.json", row)
            row["ring_objective"] = study.ring_objective(row["toys"]["mode_hold"])
            row["development_objective"] = objective(row)
            report["rows"].append(row)
            generation_rows.append(row)
            save()
            print(json.dumps(dict(event="DONE", name=name, objective=objective(row), **study.row_summary(row))), flush=True)
        elites = sorted(generation_rows, key=objective)[:3]
        values = torch.tensor([r["policy"]["weights"] for r in elites], dtype=mean.dtype)
        mean = .3 * mean + .7 * values.mean(0)
        std[:, 0] = (.3 * std[:, 0] + .7 * values[:, :, 0].std(0, unbiased=False)).clamp_min(.009)
    best = min(report["rows"], key=objective)
    report["selected"] = best["name"]
    baseline.write_json(output / "frozen.json", dict(policy=best["policy"], selected=best["name"], summary=study.row_summary(best),
                        fresh_transfer=TRANSFER, selection="development only; new two-case transfer unseen",
                        source_sha256=report["protocol"]["source_sha256"]))
    save()
    print(f"FROZEN {best['name']} {study.row_summary(best)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=8)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(args.output, args.warm_start, args.generations, args.population)


if __name__ == "__main__":
    main()

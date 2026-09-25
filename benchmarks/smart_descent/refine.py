"""Second development stage: optimize full-suite convergence with LR feedback.

All previously observed toys are development data. Fresh transfer remains
unseen until the final policy freeze. No regularization weights are perturbed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import baseline
from . import study


def objective(row):
    summary = study.row_summary(row)
    return (50 * (29 - summary["bounds"]) + 20 * (9 - summary["stable"])
            + summary["mean_confirmation_fraction"])


def run(source, output, generations=2, population=12):
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError("use a new output directory")
    parent_bytes = (source / "search.json").read_bytes()
    original = json.loads(parent_bytes)
    # The control is re-used as data, with exact source artifact provenance.
    control = deepcopy(next(r for r in original["rows"] if r["name"] == "cosine_control"))
    if len(control["toys"]) != 9:
        raise ValueError("source baseline must be complete")
    output.mkdir(parents=True)
    (output / "parent_snapshot.json").write_bytes(parent_bytes)
    report = dict(protocol=study.fingerprint(), base=study.asdict(study.BASE), fresh_transfer=study.TRANSFER,
                  split="Previously inspected full nine-host suite is development; declared fresh transfer remains unseen.",
                  parent_snapshot_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                  search=dict(generations=generations, population=population, perturbation_rng=731,
                              gan_seed=0, objective="50*failed bounds + 20*non-sustained toys + mean normalized confirmation step",
                              scope="LR-only feedback, regularization fixed; every new candidate runs all nine toys"),
                  rows=[control])
    def save():
        baseline.write_json(output / "search.json", report)
        study.render(report, output)
    save()
    stream = torch.Generator().manual_seed(731)
    mean = torch.zeros(2, 2, 5, dtype=torch.float64)
    std = torch.zeros_like(mean)
    std[:, 0] = .025
    for generation in range(generations):
        best = min(report["rows"], key=objective)
        proposals = [mean.clone(), torch.tensor(best["policy"]["weights"], dtype=torch.float64)]
        while len(proposals) < population:
            proposals.append(mean + torch.randn(mean.shape, generator=stream, dtype=mean.dtype) * std)
        generation_rows = []
        for index, weights in enumerate(proposals):
            card = study.policy(weights)
            duplicate = next((r for r in report["rows"] if r["policy"] == card), None)
            if duplicate:
                generation_rows.append(duplicate)
                continue
            name = f"lr_g{generation:02d}_p{index:02d}"
            row = dict(name=name, policy=card, config=study.asdict(study.BASE), toys={}, ring_objective=1000.)
            print(f"START {name}", flush=True)
            # Ring first makes live progress readable, but no screening is used.
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
        std[:, 0] = (.3 * std[:, 0] + .7 * values[:, :, 0].std(0, unbiased=False)).clamp_min(.006)
    best = min(report["rows"], key=objective)
    summary = study.row_summary(best)
    frozen = dict(policy=best["policy"], selected=best["name"], summary=summary,
                  fresh_transfer=study.TRANSFER, selection="full development suite only; transfer unseen",
                  source_sha256=report["protocol"]["source_sha256"])
    baseline.write_json(output / "frozen.json", frozen)
    report["selected"] = best["name"]
    save()
    print(f"FROZEN {best['name']} {summary}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=12)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(args.source, args.output, args.generations, args.population)


if __name__ == "__main__":
    main()

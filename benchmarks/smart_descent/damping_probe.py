"""Stronger clock-free responses beyond the local standalone search scale."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import baseline
from . import study
from .refine import objective
from .standalone import TRANSFER


def run(source, output):
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError("use a new output directory")
    payload = (source / "search.json").read_bytes()
    parent = json.loads(payload)
    control = deepcopy(next(r for r in parent["rows"] if r["name"] == "constant_control"))
    output.mkdir(parents=True)
    (output / "parent_snapshot.json").write_bytes(payload)
    report = dict(protocol=study.fingerprint(), base=study.asdict(study.BASE), fresh_transfer=TRANSFER,
                  parent_snapshot_sha256=hashlib.sha256(payload).hexdigest(),
                  split="Full nine-host development set; standalone's two declared transfer cases remain unseen.",
                  search=dict(scope="12 stronger LR-only policies with no clock or external schedule; ring screening before other eight hosts",
                              features=["log_gradient_ratio", "log_gradient_innovation"], gains=[.3, .7],
                              roles=["g", "d", "both"], gan_seed=0), rows=[control])
    def save():
        baseline.write_json(output / "search.json", report)
        study.render(report, output)
    save()
    for feature, index in (("growth", 1), ("innovation", 3)):
        for gain in (.3, .7):
            for role in ("g", "d", "both"):
                weights = torch.zeros(2, 2, 5)
                for role_index in ([0, 1] if role == "both" else [0 if role == "g" else 1]):
                    weights[role_index, 0, index] = -gain
                name = f"{feature}_{str(gain).replace('.', 'p')}_{role}"
                card = study.policy(weights, "constant")
                row = dict(name=name, policy=card, config=study.asdict(study.BASE), toys={})
                print(f"START {name}", flush=True)
                row["toys"]["mode_hold"] = study.run_toy("mode_hold", card)
                row["ring_objective"] = study.ring_objective(row["toys"]["mode_hold"])
                report["rows"].append(row)
                save()
                if row["toys"]["mode_hold"].get("convergence", {}).get("stable_from_step") is not None:
                    for toy in baseline.BUDGETS:
                        if toy == "mode_hold":
                            continue
                        row["toys"][toy] = study.run_toy(toy, card)
                        save()
                row["development_objective"] = objective(row)
                save()
                print(json.dumps(dict(event="DONE", name=name, **study.row_summary(row),
                                      ring=row["toys"]["mode_hold"].get("live"))), flush=True)
    report["selected"] = min(report["rows"], key=objective)["name"]
    save()
    print(f"FROZEN {report['selected']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)

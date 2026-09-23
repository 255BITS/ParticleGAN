"""Read-only layer-scale measurements for the already-selected SGD rates."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import sgd_study
from .study import TRAIN_TASKS, write_json


def run(output):
    torch.set_num_threads(1)
    output = Path(output)
    sweep = json.loads((output / "sweep.json").read_text())
    base = sweep["best_constant"]
    report = {"selection": base["name"], "purpose": "read-only layer-scale diagnosis; no fitting decisions",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": []}
    for task in TRAIN_TASKS:
        layers, created = [], []
        original = sgd_study.make_sgd
        def factory(parameters, lr):
            optimizer = original(parameters, lr)
            role = "g" if not created else "d"
            created.append(optimizer)
            steps = [0]
            def before_update(opt, args, kwargs):
                step = steps[0]
                steps[0] += 1
                if step not in (0, 100, 300, 600, 1199):
                    return
                values = []
                with torch.no_grad():
                    for group in opt.param_groups:
                        for p in group["params"]:
                            grad_rms = float(p.grad.square().mean().sqrt()) if p.grad is not None else 0.
                            param_rms = float(p.square().mean().sqrt())
                            step_rms = grad_rms * group["lr"]
                            values.append(dict(shape=list(p.shape), gradient_rms=grad_rms,
                                               parameter_rms=param_rms, update_rms=step_rms,
                                               relative_update_rms=step_rms / param_rms if param_rms > 0 else None))
                layers.append(dict(step=step, role=role, parameters=values))
            optimizer.register_step_pre_hook(before_update)
            return optimizer
        with patch.object(sgd_study, "make_sgd", factory):
            row = sgd_study.run_episode(task, base["g_lr"], base["d_lr"])
        previous = json.loads((output / "episodes" / f"{base['name']}_{task}.json").read_text())
        differences = []
        for before, after in zip(previous["curve"], row["curve"]):
            differences.extend(abs(before[k] - after[k]) for k in before if k != "seconds")
        report["rows"].append(dict(task=task, layers=layers, live=row["live"],
                                   max_metric_difference=max(differences, default=None),
                                   same_curve_length=len(row["curve"]) == len(previous["curve"])))
        write_json(output / "layer_diagnostics.json", report)
        print(task, "max_metric_difference", report["rows"][-1]["max_metric_difference"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

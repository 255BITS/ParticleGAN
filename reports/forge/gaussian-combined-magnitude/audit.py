"""Read-only explanation of saved full gates and cached GPU directions.

This posthoc diagnostic never trains, backpropagates, samples a model or changes
the protocol. Saved-output scoring is the inherited CPU numerical exception.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import ndtr
import torch
from experiments.forge.contracts import atomic_json, file_hash


def inspect(raw, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("cached-field analysis requires CUDA")
    run = json.loads((raw / "results.json").read_text())
    rows = []
    for receipt in run["results"]:
        name = f"{receipt['arm']}-{receipt['task']}-{receipt['phase']}"
        directory = raw / name
        curve = json.loads((directory / "curve.json").read_text())
        acquisition = 5000 if receipt["phase"] == "shift" else (1000 if receipt["task"].startswith("gaussian") else 1600)
        hold = [r for r in curve if r["step"] > acquisition]
        row = dict(cell=name, checkpoint_sha256=file_hash(directory / "state.pt"),
                   hold_checks=len(hold), hold_failed_bounds=dict(Counter(
                       b for r in hold for b in r["full_failed_bounds"])))
        if receipt["task"].startswith("gaussian"):
            samples = torch.load(directory / "observations.pt", map_location="cpu", weights_only=True)
            values = np.sort(samples[-1]["samples"][:, 0].double().numpy())
            standard = (values - values.mean()) / values.std()
            cdf, ranks = ndtr(standard), np.arange(len(values)) / len(values)
            row.update(final_fitted_normal_ks=max(float(np.max(cdf-ranks)), float(np.max(ranks+1/len(values)-cdf))),
                       final_skewness=float(np.mean(standard**3)), final_excess_kurtosis=float(np.mean(standard**4)-3),
                       hold_mean_error_sigma_range=[min(r["metrics"]["mean_error_sigma"] for r in hold), max(r["metrics"]["mean_error_sigma"] for r in hold)],
                       hold_std_ratio_range=[min(r["metrics"]["std_ratio"] for r in hold), max(r["metrics"]["std_ratio"] for r in hold)])
        if receipt["arm"] == "extrapolation_from_past":
            saved = torch.load(directory / "state.pt", map_location=device, weights_only=True)
            cache = saved["trainer"]["extrapolation"]["previous"]
            parameters = []
            for name, field in cache.items():
                if name == "prior.z":
                    continue
                factor = math.sqrt(max(1., field.shape[0]/field.shape[1])) if field.ndim == 2 else 1.
                cap = factor*math.sqrt(min(field.shape)) if field.ndim == 2 else 1.
                parameters.append(dict(parameter=name, cached_direction_norm=float(field.norm()),
                                       dualnorm_motion_cap_norm=cap, cap_norm_utilization=float(field.norm())/cap))
            norms = cache["prior.z"].norm(dim=1)
            active = norms[norms > 0]
            if bool((norms > 1.000001).any()):
                raise ValueError("saved prior field exceeds its motion cap")
            row.update(parameters=parameters, prior_nonzero_rows=len(active),
                       prior_direction_min_norm=float(active.min()), prior_direction_mean_norm=float(active.mean()),
                       prior_direction_max_norm=float(active.max()), implied_mean_prior_row_motion=.03*float(active.mean()),
                       prior_fraction_at_cap=float((active >= .99999).float().mean()))
        rows.append(row)
    return dict(scope="posthoc_saved_state_explanation_not_gate", training_updates=0,
                new_gradient_evaluations=0, model_sampling_draws=0, field_device=device,
                source_sha256=file_hash(Path(__file__)), results=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = inspect(args.raw, args.device)
    atomic_json(args.output, result)
    print(json.dumps(dict(audited=len(result["results"]), training_updates=0, new_gradient_evaluations=0)))

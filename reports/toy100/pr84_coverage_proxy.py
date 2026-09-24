"""One fixed-cloud check of one-sided sampled-real quantization drift.

This has no model, critic, optimizer, or training updates. It reads the host's
one noisy support draw from the frozen PR84 audit and applies Lloyd centroid
updates against one independently drawn native 128-example real minibatch.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.locked_shared.mode_hold import diversity, ring_means, sample_ring


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith(".gz") else data)


def simulate(points, real, means, *, missing=None):
    x = torch.tensor(points, dtype=torch.float32)
    rows = []
    for step in range(21):
        distances = torch.cdist(real, x)
        assign = distances.argmin(1)
        counts = torch.bincount(assign, minlength=len(x))
        centroid = torch.stack([real[assign == i].mean(0) if counts[i] else x[i]
                                for i in range(len(x))])
        force = centroid - x
        score = diversity(x, means)
        row = {"step": step, "modes": score["modes"], "hq": score["hq"],
               "coverage_loss": float(distances.min(1).values.square().mean()),
               "empty_cells": int((counts == 0).sum())}
        if step == 0:
            row["assignment_counts"] = counts.tolist()
            if missing is not None:
                direction = means[missing] - x
                direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
                row["centroid_force_toward_missing"] = (force * direction).sum(1).tolist()
        rows.append(row)
        x = centroid
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cold", type=Path, required=True)
    p.add_argument("--warm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    cold, warm = read_json(args.cold), read_json(args.warm)
    cold_observation = cold["result"]["observations"][-1]
    warm_observation = warm["observations"][-1]
    means = ring_means()
    real = sample_ring(means, 128, .07, torch.Generator().manual_seed(0))
    output = {
        "scope": "fixed cloud from one noisy support draw per particle, not model training",
        "real_batch": 128,
        "real_batch_sha256": hashlib.sha256(real.numpy().tobytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cold_input_sha256": hashlib.sha256(args.cold.read_bytes()).hexdigest(),
        "warm_input_sha256": hashlib.sha256(args.warm.read_bytes()).hexdigest(),
        "cold_missing_mode_for_reporting_only": cold_observation["missing_modes"],
        "cold": simulate(cold_observation["support"]["points"], real, means,
                         missing=cold_observation["missing_modes"][0]),
        "warm": simulate(warm_observation["support"]["points"], real, means),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: [rows[i] for i in (0, 1, 2, 20)]
                      for name, rows in (("cold", output["cold"]), ("warm", output["warm"]))}))


if __name__ == "__main__":
    main()

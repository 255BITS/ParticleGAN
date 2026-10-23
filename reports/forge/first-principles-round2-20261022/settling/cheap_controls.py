"""Fixed local-exchangeability thermostat screen; no training or new draws.

The algorithm and admission criteria are frozen before this screen runs.
Numerics use float64 hypergeometric tails, not simulated label permutations.
All controller inputs are archived samples or separate analytical fixtures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time

import numpy as np
from scipy.special import gammaln
from scipy.spatial.distance import cdist
import torch

ALPHA = .05
WINDOW = 2048
REQUEST = "728323efa37b4c2345764a9d"
SOURCE = "8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_pass(metrics):
    return metrics["finite_fraction"] == 1 and metrics["cdf_ks"] <= .05 and metrics["mean_error_sigma"] <= .2 and .8 <= metrics["std_ratio"] <= 1.2


def overlap_pass(metrics):
    return metrics["sw1_normalized"] <= .18 and metrics["mean_error"] <= .15 and metrics["covariance_error"] <= .45


@lru_cache(maxsize=4096)
def tails(total, real_count, neighbors):
    lo, hi = max(0, neighbors - (total - real_count)), min(real_count, neighbors)
    count = np.arange(lo, hi + 1, dtype=np.float64)
    def logchoose(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    logp = logchoose(real_count, count) + logchoose(total - real_count, neighbors - count) - logchoose(total, neighbors)
    probability = np.exp(logp)
    probability /= probability.sum()
    both = np.minimum(1., 2 * np.minimum(np.cumsum(probability), np.cumsum(probability[::-1])[::-1]))
    return lo, both


def scan(real, fake):
    if real.shape != fake.shape or len(real) < 2:
        raise ValueError("balanced nonempty panels required")
    count = len(real)
    points = np.concatenate((real, fake)).astype(np.float64, copy=False)
    size = len(points)
    scale_count_bound = math.ceil(math.log2(size)) + 1
    first = math.ceil(math.log2(2 * size * scale_count_bound / ALPHA ** 2))
    ranks = []
    rank = min(first, size // 2)
    while rank < size // 2:
        ranks.append(rank)
        rank *= 2
    ranks.append(size // 2)
    distance = cdist(points, points)
    np.fill_diagonal(distance, np.inf)
    partition = np.partition(distance, [rank - 1 for rank in ranks], axis=1)
    minimum = 1.
    witness = None
    labels = np.arange(size) < count
    for rank in ranks:
        radius = partition[:, rank - 1]
        closed = distance <= radius[:, None]
        neighborhood = closed.sum(1)
        real_neighbors = closed[:, :count].sum(1)
        for actual in np.unique(neighborhood):
            for is_real in (True, False):
                selected = (neighborhood == actual) & (labels == is_real)
                if not selected.any():
                    continue
                lo, probabilities = tails(size - 1, count - int(is_real), int(actual))
                p = probabilities[real_neighbors[selected] - lo]
                local = int(p.argmin())
                value = float(p[local])
                if value < minimum:
                    index = int(np.flatnonzero(selected)[local])
                    minimum = value
                    witness = {"center_index": index, "center_sample": "real" if is_real else "fake",
                        "rank": rank, "closed_neighbors": int(actual), "real_neighbors": int(real_neighbors[index])}
    adjusted = min(1., size * len(ranks) * minimum)
    mobility = 1. if adjusted == 0 else min(1., max(0., math.log(1 / adjusted) / math.log(1 / ALPHA)))
    return {"mobility": mobility, "p_global": adjusted, "minimum_local_p": minimum,
        "panel_size": count, "ranks": ranks, "witness": witness}


def summary(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values),
        "full_heat": sum(value >= 1 for value in values), "zero": sum(value == 0 for value in values)} if values else None


def auc(good, bad):
    return sum(float(b > a) + .5 * (b == a) for a in good for b in bad) / (len(good) * len(bad)) if good and bad else None


def atom_counts(points, weights, size=WINDOW):
    weights = np.asarray(weights)
    counts = np.floor(weights * size).astype(int)
    for index in np.argsort(-(weights * size - counts), kind="stable")[:size - counts.sum()]:
        counts[index] += 1
    return np.repeat(np.asarray(points, dtype=np.float64), counts, axis=0), counts


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    start = time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/queue"))
    args = parser.parse_args()
    state = json.loads((args.queue / "queue/state.json").read_text())
    if state["submissions"][REQUEST]["request"]["source"]["digest"] != SOURCE:
        raise ValueError("source mismatch")
    saved = {}
    receipts = {}
    for job in state["jobs"].values():
        task = job["definition"]["execution_group"]
        if REQUEST not in job["subscribers"] or task not in ("gaussian1d_smoke", "gaussian1d_stability", "vector_overlap"):
            continue
        row, = job["result"]["task_results"]
        evidence = row["evidence"]
        receipt = evidence["saved_observer_outputs"]
        path = Path(evidence.get("artifact_root") or job["attempts"][-1]["path"]) / receipt["path"]
        if sha(path) != receipt["sha256"]:
            raise ValueError("sample hash mismatch")
        metrics = {item["step"]: item for item in evidence["observations"]}
        records = torch.load(path, map_location="cpu", weights_only=False)
        saved[task] = {item["step"]: {"samples": item["samples"].double().numpy(), "metrics": metrics[item["step"]]}
                       for item in records if item["step"] in metrics}
        receipts[task] = {"path": str(path), "sha256": receipt["sha256"]}
    passing = {task: [step for step, item in rows.items() if (overlap_pass if task == "vector_overlap" else gaussian_pass)(item["metrics"])]
               for task, rows in saved.items()}
    references = {"stationary": saved["gaussian1d_smoke"][min(passing["gaussian1d_smoke"])]["samples"],
                  "shifted": saved["gaussian1d_stability"][min(step for step in passing["gaussian1d_stability"] if step > 4000)]["samples"],
                  "overlap": saved["vector_overlap"][min(passing["vector_overlap"])]["samples"]}
    result = {"schema_version": 1, "algorithm": "closed-ball multiscale hypergeometric label scan",
        "alpha": ALPHA, "window": WINDOW, "source_digest": SOURCE,
        "sampling_draws_added": 0, "optimizer_updates_added": 0, "sample_receipts": receipts,
        "same_law_full": [], "same_law_small": [], "saved_states": [], "fixtures": {}}

    def run(label, real, fake):
        if time.process_time() - start > 112:
            raise RuntimeError("cheap control CPU budget exhausted")
        out = scan(real, fake)
        print(json.dumps({"case": label, "mobility": out["mobility"], "p_global": out["p_global"],
                          "cpu_seconds": time.process_time() - start}), flush=True)
        return out

    for task in ("gaussian1d_stability", "vector_overlap"):
        for step in passing[task]:
            values = saved[task][step]["samples"]
            result["same_law_full"].append({"task": task, "step": step,
                **run(f"same_law_{task}_{step}_2048", values[:WINDOW], values[WINDOW:2 * WINDOW])})
            for offset in range(0, 4096, 256):
                result["same_law_small"].append({"task": task, "step": step, "offset": offset,
                    **run(f"same_law_{task}_{step}_{offset}_128", values[offset:offset + 128], values[offset + 128:offset + 256])})
    selected = {"gaussian1d_smoke": [42, 84, 125, 209, 500, 1000],
                "gaussian1d_stability": [1042, 1125, 2000, 4000, 4042, 4709, 5000, 5042, 6000],
                "vector_overlap": [50, 100, 750, 800, 1000, 1050, 1100, 1150, 1200]}
    for task, steps in selected.items():
        for step in steps:
            item = saved[task][step]
            metrics = item["metrics"]
            if task == "vector_overlap":
                reference = references["overlap"]
                passed = overlap_pass(metrics)
                severity = max(metrics["mean_error"] / .15, metrics["covariance_error"] / .45, metrics["sw1_normalized"] / .18)
            else:
                reference = references["stationary" if step <= 4000 else "shifted"]
                passed = gaussian_pass(metrics)
                severity = max(metrics["mean_error_sigma"] / .2, metrics["cdf_ks"] / .05, abs(metrics["std_ratio"] - 1) / .2)
            result["saved_states"].append({"task": task, "step": step, "receipt_pass": passed, "severity": severity,
                **run(f"reference_{task}_{step}", reference[:WINDOW], item["samples"][WINDOW:2 * WINDOW])})
    constant = np.full((WINDOW, 1), 3.)
    controls = {"matched_constant": (constant, constant.copy()), "shifted_constant": (constant, constant + 1)}
    for name in ("stationary", "overlap"):
        values = references[name][:WINDOW]
        controls[f"collapsed_{name}"] = values, np.broadcast_to(values.mean(0), values.shape).copy()
    corners = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]
    balanced, _ = atom_counts(corners, [.25] * 4)
    controls["matched_four_atoms"] = balanced, balanced.copy()
    controls["missing_one_four_atoms"] = balanced, atom_counts(corners, [1/3] * 3 + [0])[0]
    rare, rare_counts = atom_counts(corners, [.55, .30, .13, .02])
    missing_counts = rare_counts.copy()
    missing_counts[0] += missing_counts[-1]
    missing_counts[-1] = 0
    controls["missing_rare2percent"] = rare, np.repeat(np.asarray(corners), missing_counts, axis=0)
    for modes in (8, 16):
        angle = np.arange(modes) * (2 * np.pi / modes)
        points = np.stack((np.cos(angle), np.sin(angle)), axis=1)
        controls[f"missing_one_ring{modes}"] = atom_counts(points, [1/modes] * modes)[0], atom_counts(points, [1/(modes - 1)] * (modes - 1) + [0])[0]
    for name, (real, fake) in controls.items():
        result["fixtures"][name] = run(name, real, fake)
    good = [item["mobility"] for item in result["saved_states"] if item["receipt_pass"]]
    severe = [item["mobility"] for item in result["saved_states"] if item["severity"] >= 2]
    full = [item["mobility"] for item in result["same_law_full"]]
    small = [item["mobility"] for item in result["same_law_small"]]
    fixture_ok = all(row["mobility"] == (0. if name.startswith("matched") else 1.) for name, row in result["fixtures"].items())
    score = auc(good, severe)
    result["summary"] = {"same_law_full": summary(full), "same_law_small": summary(small),
        "saved_pass": summary(good), "saved_severe": summary(severe), "auroc": score,
        "all_support_fixtures_full_and_matched_fixtures_zero": fixture_ok,
        "screen_pass": fixture_ok and sum(value == 1 for value in full) <= 1
            and sum(value == 1 for value in small) <= .05 * len(small)
            and score is not None and score >= .8 and statistics.median(good) <= .5 and statistics.median(severe) >= .95}
    result["cpu_seconds"] = time.process_time() - start
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

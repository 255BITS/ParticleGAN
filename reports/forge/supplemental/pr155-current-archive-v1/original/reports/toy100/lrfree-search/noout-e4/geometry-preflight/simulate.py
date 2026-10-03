#!/usr/bin/env python3
"""Run the synthetic latent-geometry fixture fixed in SPEC.md."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


KS = (5, 10, 20, 40)
OUT = Path(__file__).with_name("results.json")


def isolated(legitimate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = legitimate.shape[1]
    directions = rng.normal(size=(8, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radius = 6.0 * float(np.median(np.linalg.norm(legitimate, axis=1)))
    return directions * radius + rng.normal(scale=0.02, size=(8, d))


def tables(d: int, rng: np.random.Generator):
    normal = rng.normal(size=(1024, d))
    normal_isolates = isolated(normal, rng)
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    stretch = np.ones(d)
    stretch[0] = 10.0
    transform = lambda x: (x * stretch) @ rotation

    heavy = rng.standard_t(df=5, size=(1024, d)) * math.sqrt(3.0 / 5.0)
    rare_bulk = rng.normal(size=(1008, d))
    rare_cluster = rng.normal(scale=0.25, size=(16, d))
    rare_cluster[:, 0] += 6.0 * math.sqrt(d)
    rare = np.concatenate((rare_bulk, rare_cluster))

    return rotation, {
        "normal": (np.concatenate((normal, normal_isolates)), None),
        "anisotropic": (np.concatenate((transform(normal), transform(normal_isolates))), None),
        "heavy_tail": (np.concatenate((heavy, isolated(heavy, rng))), None),
        "rare_group": (np.concatenate((rare, isolated(rare, rng))), slice(1008, 1024)),
    }


def distances(x: np.ndarray) -> np.ndarray:
    squared_norm = np.einsum("ij,ij->i", x, x)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (x @ x.T)
    np.maximum(squared, 0.0, out=squared)
    np.sqrt(squared, out=squared)
    np.fill_diagonal(squared, np.inf)
    return squared


def gains(distance: np.ndarray, k: int) -> np.ndarray:
    neighbors = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(len(distance))[:, None]
    neighbor_distances = distance[rows, neighbors]
    k_distance = neighbor_distances.max(axis=1)
    reach = np.maximum(neighbor_distances, k_distance[neighbors])
    local_density = 1.0 / reach.mean(axis=1)
    lof = local_density[neighbors].mean(axis=1) / local_density
    return np.maximum(0.0, 1.0 - 1.0 / lof)


def score(stage: str, d: int, family: str, k: int, gain: np.ndarray,
          rare_slice: Optional[slice]) -> dict:
    legitimate = gain[:1024]
    planted = gain[1024:]
    result = {
        "stage": stage, "dimension": d, "family": family, "k": k,
        "legitimate_mean": float(np.mean(legitimate)),
        "legitimate_p95": float(np.quantile(legitimate, 0.95)),
        "isolate_mean": float(np.mean(planted)),
        "isolate_fraction_ge_03": float(np.mean(planted >= 0.30)),
        "rare_group_mean": (float(np.mean(legitimate[rare_slice]))
                            if rare_slice is not None else None),
    }
    result["pass"] = (
        result["legitimate_mean"] <= 0.07
        and result["legitimate_p95"] <= 0.30
        and result["isolate_mean"] >= 0.40
        and result["isolate_fraction_ge_03"] >= 0.75
        and (result["rare_group_mean"] is None or result["rare_group_mean"] <= 0.10)
    )
    print(
        f"{stage:11} d={d:2d} {family:11} k={k:2d} "
        f"legit_mean={result['legitimate_mean']:.4f} "
        f"legit_p95={result['legitimate_p95']:.4f} "
        f"rare={result['rare_group_mean']} "
        f"isolate_mean={result['isolate_mean']:.4f} "
        f"isolate_ge_03={result['isolate_fraction_ge_03']:.2f} "
        f"{'PASS' if result['pass'] else 'FAIL'}",
        flush=True,
    )
    return result


def evaluate(stage: str, seed: int, dimensions: tuple[int, ...],
             ks: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    rows, invariances = [], []
    for d in dimensions:
        rotation, fixtures = tables(d, rng)
        for family, (positions, rare_slice) in fixtures.items():
            base_distance = distances(positions)
            for k in ks:
                gain = gains(base_distance, k)
                rows.append(score(stage, d, family, k, gain, rare_slice))
                if family == "normal":
                    errors = {
                        "scale_1e-3": float(np.max(np.abs(gain - gains(distances(positions * 1e-3), k)))),
                        "scale_100": float(np.max(np.abs(gain - gains(distances(positions * 100.0), k)))),
                        "rotation": float(np.max(np.abs(gain - gains(distances(positions @ rotation), k)))),
                    }
                    item = {"stage": stage, "dimension": d, "k": k,
                            "max_error": errors, "pass": all(v <= 1e-6 for v in errors.values())}
                    invariances.append(item)
                    print(f"{stage:11} d={d:2d} normal      k={k:2d} invariance={errors} "
                          f"{'PASS' if item['pass'] else 'FAIL'}", flush=True)
    return rows, invariances


def run() -> None:
    calibration, calibration_invariance = evaluate("calibration", 20261005, (2, 8), KS)
    eligible = [k for k in KS
                if all(r["pass"] for r in calibration if r["k"] == k)
                and all(r["pass"] for r in calibration_invariance if r["k"] == k)]
    chosen = min(eligible) if eligible else None
    if chosen is None:
        validation, validation_invariance = [], []
        passed = False
    else:
        validation, validation_invariance = evaluate("validation", 20261006,
                                                     (3, 16), (chosen,))
        passed = all(r["pass"] for r in validation + validation_invariance)
    result = {
        "spec": "SPEC.md", "calibration_seed": 20261005,
        "validation_seed": 20261006, "candidate_k": KS,
        "eligible_k": eligible, "chosen_k": chosen, "passed": passed,
        "calibration": calibration, "calibration_invariance": calibration_invariance,
        "validation": validation, "validation_invariance": validation_invariance,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}; eligible={eligible}, "
          f"chosen={chosen}; {OUT}")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Read-only mechanism summary of matched Forge readouts; launches no training.

The parent campaign owns the single leaderboard and scientific selection.
This reducer checks pairing metadata and summarizes allocation/shape witnesses.
It never changes gates or regrades original outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASKS = (
    "gaussian1d_smoke", "gaussian1d_stability", "vector_unequal_mass",
    "vector_unequal_width", "vector_anisotropic", "vector_overlap", "img_bars4",
    "img_blobs4", "img_intensity2", "grid100", "rotated100", "staggered100",
    "ring16_acquisition", "mode_hold", "img_stripes2",
)
METRICS = (
    "mass_tv", "min_mass_ratio", "component_counts", "component_mass",
    "component_covariance_error", "component_core_covariance_error",
    "component_covariance_errors", "component_core_covariance_errors",
    "component_min_eigen_ratio", "max_component_spill", "hq", "modes",
    "distribution_tv", "mode_fractions", "quality_mode_fractions", "mean_rmse",
    "sw1_normalized", "precision", "center_rms_sigma", "cov_frob_rms",
    "radial_ks", "cdf_ks", "std_ratio", "mean_error_sigma",
)


def read(path, expected_id):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["candidate_id"] != expected_id:
        raise ValueError(f"expected {expected_id}, got {data['candidate_id']}")
    rows = {}
    for row in data["task_results"]:
        task = row["task_id"]
        if task in rows:
            raise ValueError(f"duplicate result for {task}; choose one certified final attempt explicitly")
        rows[task] = row
    if set(rows) != set(TASKS):
        raise ValueError(f"unexpected task panel: {sorted(set(rows).symmetric_difference(TASKS))}")
    return rows, dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                      candidate_id=expected_id, candidate_revision=data["candidate_revision"],
                      attempt_ids=data["attempt_ids"])


def training_streams(row):
    """Normalize only CUDA device ordinal; retain every consumed non-eval stream."""
    packet = row["evidence"]["provenance_checkpoint"]
    normalized = {}
    for raw_key, digest in packet["named_stream_state_sha256"].items():
        key = json.loads(raw_key)
        if key[0] == "eval":
            continue
        if key[-1].startswith("cuda:"):
            key[-1] = "cuda"
        normalized[json.dumps(key, separators=(",", ":"))] = digest
    return normalized


def summary(row):
    evidence = row["evidence"]
    grade = row.get("evaluator_result", {})
    convergence = grade.get("convergence", {})
    result = dict(attempt_id=row["_attempt_id"], status=row["status"],
                  gate_status=row["gate_status"],
                  endpoint={key: row["metrics"][key] for key in METRICS if key in row["metrics"]},
                  convergence=convergence,
                  failing_endpoint_metrics=[entry for entry in grade.get("metrics", [])
                                            if entry.get("status") != "PASS"],
                  guards=evidence.get("guards", {}))
    if row["task_id"] == "gaussian1d_stability":
        result["phase_counts"] = grade
    observations = evidence.get("observations", [])
    if row["task_id"] == "vector_unequal_mass" and observations:
        rare_index = min(range(len(observations[-1]["target_mass"])),
                         key=observations[-1]["target_mass"].__getitem__)
        occupied = [obs["step"] for obs in observations if obs["component_counts"][rare_index] > 0]
        empty = [obs["step"] for obs in observations if obs["component_counts"][rare_index] == 0]
        result["rare_observed_support"] = dict(index=rare_index, scheduled_checks=len(observations),
            occupied_checks=len(occupied), empty_checks=len(empty),
            first_occupied_step=min(occupied, default=None), last_occupied_step=max(occupied, default=None),
            first_empty_step=min(empty, default=None), last_empty_step=max(empty, default=None),
            endpoint_count=observations[-1]["component_counts"][rare_index])
    return result


def compare(control, candidate):
    output = {}
    for task in TASKS:
        left, right = control[task], candidate[task]
        checks = {key: left.get(key) == right.get(key)
                  for key in ("initializer", "initialization", "initial_prior", "prior_contract")}
        checks["sampling_law"] = left["evidence"].get("sampling_law") == right["evidence"].get("sampling_law")
        checks["optimizer_updates"] = (left["evidence"]["guards"].get("optimizer_updates")
                                      == right["evidence"]["guards"].get("optimizer_updates"))
        checks["complete_consumed_non_eval_streams"] = training_streams(left) == training_streams(right)
        output[task] = dict(pairing=checks, control=summary(left), candidate=summary(right))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control, source_control = read(args.control, "bispf-optimized-quintic-v1")
    candidate, source_candidate = read(args.candidate, "bispf-output-energy-v1")
    result = dict(schema_version=1, scope="read_only_mechanism_analysis", qualification_input=False,
                  additional_updates=0, additional_draws=0, sources=[source_control, source_candidate],
                  tasks=compare(control, candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    errors = [(task, key) for task, row in result["tasks"].items()
              for key, value in row["pairing"].items() if not value]
    print(json.dumps(dict(event="allocation_analysis", tasks=len(result["tasks"]), pairing_errors=errors,
                          output=str(args.output)), sort_keys=True), flush=True)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

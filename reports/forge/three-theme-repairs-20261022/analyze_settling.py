"""Read-only settling analysis of frozen three-theme receipts and saved tensors.

No model construction, optimizer update, sample or random draw occurs. The
parent publication owns the single comparison leaderboard; this writes compact
mechanism diagnostics, never a second ranking.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch


REQUESTS = {"control": "728323efa37b4c2345764a9d", "settling": "7a58ff2ed6de1ece217a77ce"}
SOURCE_COMMIT = "2a2b662bdfa6249076b27dbea4c136e23d3e7a80"
SOURCE_DIGEST = "8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_pass(row):
    return (row["finite_fraction"] == 1 and row["mean_error_sigma"] <= .2
            and .8 <= row["std_ratio"] <= 1.2 and row["cdf_ks"] <= .05)


def overlap_pass(row):
    return row["mean_error"] <= .15 and row["covariance_error"] <= .45 and row["sw1_normalized"] <= .18


def runs(values):
    longest = suffix = 0
    for value in values:
        suffix = suffix + 1 if value else 0
        longest = max(longest, suffix)
    return {"passing": sum(values), "observations": len(values), "longest": longest, "terminal_suffix": suffix}


def scalar_summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else None


def load_saved(path, expected=None):
    sha = digest(path)
    if expected is not None and sha != expected:
        raise ValueError(f"artifact hash differs: {path}")
    return torch.load(path, map_location="cpu", weights_only=False), {"path": str(path), "sha256": sha}


def observer_checkpoint(path, expected=None):
    state, receipt = load_saved(path, expected)
    trainer = state["trainer"]
    record = trainer["optimizers"][1]["regularizer"]["record"]
    return {**receipt, "step": trainer["completed_steps"], "observer": record.get("bispf", record)}


def final_state_diagnostic(row):
    info = row["evidence"].get("provenance_checkpoint")
    if not info:
        return None
    path = Path(info["artifact_root"]) / info["path"]
    state, receipt = load_saved(path, info["sha256"])
    trainer = state["trainer"]
    hashes = {}

    def append(value, item):
        if isinstance(item, torch.Tensor):
            value.update(f"tensor:{item.dtype}:{tuple(item.shape)}".encode())
            value.update(item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key, child in sorted(item.items()):
                value.update(json.dumps(key).encode())
                append(value, child)
        elif isinstance(item, (list, tuple)):
            value.update(f"sequence:{len(item)}".encode())
            for child in item:
                append(value, child)
        else:
            value.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())

    for model, tensors in trainer["models"].items():
        value = hashlib.sha256()
        append(value, tensors)
        hashes[model] = value.hexdigest()
    optimizer_hashes = []
    for optimizer in trainer["optimizers"]:
        value = hashlib.sha256()
        append(value, optimizer)
        optimizer_hashes.append(value.hexdigest())
    record = trainer["optimizers"][1]["regularizer"]["record"]
    return {**receipt, "step": trainer["completed_steps"], "observer": record.get("bispf", record),
        "model_tensor_sha256": hashes, "optimizer_state_sha256": optimizer_hashes}


def samples(job, row):
    receipt = row["evidence"]["saved_observer_outputs"]
    root = Path(row["evidence"].get("artifact_root") or job["attempts"][-1]["path"])
    return load_saved(root / receipt["path"], receipt["sha256"])


def observations(row):
    evidence = row["evidence"]
    if "observations" in evidence:
        return evidence["observations"]
    manifest = evidence.get("artifact_manifest", {}).get("files", {})
    relative = f"{row['task_id']}/events.jsonl"
    if relative in manifest:
        path = Path(evidence["artifact_root"]) / relative
        if digest(path) != manifest[relative]["sha256"]:
            raise ValueError(f"native event artifact differs: {path}")
        return [{key: value for key, value in item.items() if key != "elapsed"}
                for item in map(json.loads, path.read_text().splitlines())
                if item.get("event") == "eval" and item.get("model") == "live"]
    return None


def native_snapshot_comparison(baseline, candidate):
    name = baseline["task_id"]
    left = baseline["evidence"].get("artifact_manifest", {}).get("files", {})
    right = candidate["evidence"].get("artifact_manifest", {}).get("files", {})
    files = sorted(key for key in left.keys() & right.keys() if key.startswith(f"{name}/snapshots/") and key.endswith(".npz"))
    if not files:
        return None
    different = []
    same_steps = []
    target_equal = True
    for relative in files:
        a = Path(baseline["evidence"]["artifact_root"]) / relative
        b = Path(candidate["evidence"]["artifact_root"]) / relative
        if digest(a) != left[relative]["sha256"] or digest(b) != right[relative]["sha256"]:
            raise ValueError(f"native snapshot differs: {relative}")
        with np.load(a) as x, np.load(b) as y:
            step = int(Path(relative).stem.split("_")[-1])
            (same_steps if np.array_equal(x["live"], y["live"]) else different).append(step)
            target_equal &= np.array_equal(x["target"], y["target"])
    return {"snapshots": len(files), "target_arrays_exact": target_equal,
        "live_sample_arrays_exact": not different, "first_different_step": min(different) if different else None,
        "last_exact_before_first_difference": max((step for step in same_steps if step < min(different)), default=None) if different else None}


def same_law_sampling_floor(observed, passing_steps):
    values = []
    for item in observed:
        if item["step"] not in passing_steps:
            continue
        existing = item["samples"].float().flatten(1)
        # Two disjoint 128-row panels from the same already-scored generated
        # law. There is no resampling, new data draw, random split or training.
        for offset in range(0, len(existing) - 255, 256):
            sq = torch.cdist(existing[offset:offset + 128], existing[offset + 128:offset + 256]).square()
            residual = torch.sqrt(.5 * (sq.min(1).values.mean() + sq.min(0).values.mean())
                                  / sq.median().clamp_min(1e-12))
            values.append(float(residual))
    return {"scope": "two disjoint iid panels of the same generated law at scored passing states; not historical real/fake training residual",
        "panel_size": 128, "panel_pairs": len(values), "raw_symmetric_nn_residual": scalar_summary(values),
        "cooling_tau": .020, "above_cooling_tau": sum(value > .020 for value in values),
        "sampling_draws_added": 0}


def gaussian_diagnostic(job, row):
    observed, receipt = samples(job, row)
    by_step = {item["step"]: item for item in observed}
    if not all(item["step"] in by_step for item in row["evidence"]["observations"]):
        raise ValueError("missing scored Gaussian observation")
    passing = {item["step"] for item in row["evidence"]["observations"] if gaussian_pass(item)}
    output = {"sample_artifact": receipt, "phases": {}, "checkpoints": [],
        "same_law_sampling_floor": same_law_sampling_floor(observed, passing)}
    phases = {"stationary": (1000, 4000), "recovery": (4000, 5000), "shifted_hold": (5000, 6000)}
    for name, (low, high) in phases.items():
        metrics = [item for item in row["evidence"]["observations"] if low < item["step"] <= high]
        motion = []
        previous = None
        for item in observed:
            if item["step"] > high:
                break
            current = item["samples"].double().flatten().sort().values
            if previous is not None and low < item["step"]:
                motion.append(float((current - previous).abs().mean()) / .5)
            previous = current
        output["phases"][name] = {
            **runs([gaussian_pass(item) for item in metrics]),
            "cdf_ks": scalar_summary([item["cdf_ks"] for item in metrics]),
            "std_ratio": scalar_summary([item["std_ratio"] for item in metrics]),
            "adjacent_empirical_w1_sigma": scalar_summary(motion),
            "w1_scope": "distribution movement between independent scored batches, including finite-sample variation; not paired output displacement",
        }
    final = observed[-1]["samples"].double().flatten()
    probs = torch.tensor([.01, .05, .5, .95, .99], dtype=torch.float64)
    quantiles = torch.quantile(final, probs)
    target = 3 + .5 * torch.special.ndtri(probs)
    output["final_signed_quantile_errors_sigma"] = dict(zip(map(str, probs.tolist()), ((quantiles - target) / .5).tolist()))
    root = Path(row["evidence"]["artifact_root"])
    files = row["evidence"]["artifact_manifest"]["files"]
    for name in ("initial-state.pt", "pre-shift-state.pt", "state.pt"):
        output["checkpoints"].append(observer_checkpoint(root / name, files[name]["sha256"]))
    return output, observed


def overlap_diagnostic(job, row):
    observed, receipt = samples(job, row)
    # The frozen target is .5*N([-.35,0], .55^2 I) + .5*N([.35,0], .55^2 I).
    # These analytic moments are additional signed diagnostics; the original
    # evaluator's empirical-reference metrics and gates remain authoritative.
    target = torch.diag(torch.tensor([.55 ** 2 + .35 ** 2, .55 ** 2], dtype=torch.float64))
    selected = []
    for item in observed:
        if item["step"] not in (750, 800, 1000, 1050, 1100, 1150, 1200):
            continue
        values = item["samples"].double()
        covariance = torch.cov(values.T)
        bias = covariance - target
        selected.append({"step": item["step"], "mean": values.mean(0).tolist(),
            "covariance": covariance.tolist(), "signed_analytic_covariance_bias": bias.tolist(),
            "bias_eigenvalues": torch.linalg.eigvalsh(bias).tolist()})
    checkpoint = Path(job["attempts"][-1]["path"]) / "provenance/provenance-state.pt"
    passing = {item["step"] for item in row["evidence"]["observations"] if overlap_pass(item)}
    return {"sample_artifact": receipt, "passing_runs": runs([overlap_pass(item) for item in row["evidence"]["observations"]]),
        "same_law_sampling_floor": same_law_sampling_floor(observed, passing),
        "analytic_target_covariance": target.tolist(), "signed_covariance_windows": selected,
        "final_checkpoint": observer_checkpoint(checkpoint)}, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/queue"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    state = json.loads((args.queue_root / "queue/state.json").read_text())
    result = {"schema_version": 1, "source_commit": SOURCE_COMMIT, "source_digest": SOURCE_DIGEST,
        "sampling_draws_added": 0, "optimizer_updates_added": 0,
        "thermal_gradient_applied_move_history": "unavailable: trainer return diagnostics were not serialized by the frozen adapters",
        "requests": {}, "comparison": {}, "saved_tensor_diagnostics": {}}
    rows = {}
    panels = {}
    for arm, request in REQUESTS.items():
        source = state["submissions"][request]["request"]["source"]
        if source["origin_commit"] != SOURCE_COMMIT or source["digest"] != SOURCE_DIGEST:
            raise ValueError(f"request source binding differs: {request}")
        jobs = [job for job in state["jobs"].values() if request in job["subscribers"]]
        rows[arm] = {}
        result["requests"][arm] = {"request_id": request, "tasks": {}}
        for job in jobs:
            task = job["definition"]["execution_group"]
            entry = {"job_status": job["status"]}
            if job.get("result"):
                row, = job["result"]["task_results"]
                rows[arm][task] = row
                entry.update(status=row["status"], gate_status=row["gate_status"],
                    attempt=job["attempts"][-1]["attempt_id"], metrics=row["metrics"],
                    evaluator_result=row["evaluator_result"], charged_seconds=job["charged_seconds"])
                entry["final_state"] = final_state_diagnostic(row)
                if task in ("gaussian1d_stability", "vector_overlap") and row["evidence"].get("saved_observer_outputs"):
                    diagnostic, panel = (gaussian_diagnostic if task == "gaussian1d_stability" else overlap_diagnostic)(job, row)
                    result["saved_tensor_diagnostics"].setdefault(task, {})[arm] = diagnostic
                    panels[arm, task] = panel
            result["requests"][arm]["tasks"][task] = entry
    complete = all(len(items) == 15 for items in rows.values())
    result["complete"] = complete
    if not complete and not args.allow_incomplete:
        raise SystemExit("Both 15-task requests must be terminal; use --allow-incomplete for a provisional readout")
    for task in sorted(rows["control"].keys() & rows["settling"].keys()):
        baseline, candidate = rows["control"][task], rows["settling"][task]
        observed_a, observed_b = observations(baseline), observations(candidate)
        result["comparison"][task] = {"control": baseline["status"], "settling": candidate["status"],
            "endpoint_metrics_exact": baseline["metrics"] == candidate["metrics"],
            "observations_exact": observed_a == observed_b if observed_a is not None and observed_b is not None else None,
            "observation_count": len(observed_a) if observed_a is not None else None,
            "initialization_exact": baseline["initialization"] == candidate["initialization"],
            "rng_receipts_exact": baseline["rng"] == candidate["rng"]}
        stream_a = baseline["evidence"].get("provenance_checkpoint", {}).get("named_stream_state_sha256")
        stream_b = candidate["evidence"].get("provenance_checkpoint", {}).get("named_stream_state_sha256")
        result["comparison"][task]["final_named_stream_states_exact"] = stream_a == stream_b if stream_a is not None and stream_b is not None else None
        if observed_a is not None and observed_b is not None:
            different = [a["step"] for a, b in zip(observed_a, observed_b) if a != b]
            result["comparison"][task]["first_changed_observation_step"] = min(different) if different else None
        snapshots = native_snapshot_comparison(baseline, candidate)
        if snapshots is not None:
            result["comparison"][task]["native_snapshots"] = snapshots
        final_a = result["requests"]["control"]["tasks"][task]["final_state"]
        final_b = result["requests"]["settling"]["tasks"][task]["final_state"]
        if final_a is not None and final_b is not None:
            result["comparison"][task]["final_model_tensors_exact"] = final_a["model_tensor_sha256"] == final_b["model_tensor_sha256"]
            result["comparison"][task]["final_optimizer_states_exact"] = final_a["optimizer_state_sha256"] == final_b["optimizer_state_sha256"]
        if ("control", task) in panels and ("settling", task) in panels:
            a, b = panels["control", task], panels["settling", task]
            result["comparison"][task]["scored_sample_tensors_exact"] = len(a) == len(b) and all(
                left["step"] == right["step"] and torch.equal(left["samples"], right["samples"])
                for left, right in zip(a, b))
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    if args.summary_output:
        if args.output is None:
            raise ValueError("--summary-output requires --output for the external full-analysis receipt")
        summary = {key: value for key, value in result.items() if key not in ("requests", "saved_tensor_diagnostics")}
        summary["arms"] = {}
        for arm, data in result["requests"].items():
            final = {task: row for task, row in data["tasks"].items() if row.get("status")}
            summary["arms"][arm] = {"request_id": data["request_id"],
                "status_counts": dict(Counter(row["status"] for row in final.values())),
                "charged_seconds": sum(row["charged_seconds"] for row in final.values()),
                "tasks": {task: {"status": row["status"], "metrics": row["metrics"],
                    "attempt": row["attempt"], "final_checkpoint": {key: row["final_state"][key]
                        for key in ("path", "sha256", "step", "observer")} if row["final_state"] else None}
                    for task, row in final.items()}}
        summary["control_saved_tensor_diagnostics"] = {task: arms["control"]
            for task, arms in result["saved_tensor_diagnostics"].items() if "control" in arms}
        summary["raw_analysis"] = {"path": str(args.output), "sha256": digest(args.output)}
        summary["cost_scope"] = "charged attempt seconds; hardware contention and instrumentation preclude a speed ranking"
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

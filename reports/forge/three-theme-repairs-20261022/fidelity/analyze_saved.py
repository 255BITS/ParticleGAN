"""Read completed campaign outputs; never train or call an output sampler.

Run after both fifteen-task requests finish. Output is a compact diagnostic
receipt, not another generated goal leaderboard. Bulk observations remain in
the local queue archive. Imports use no network and all tensor reads map CPU.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


SOURCE = "8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb"
REQUESTS = {"control": "728323efa37b4c2345764a9d", "fidelity": "a0749bea42cc192e60afed81"}
METRICS = {"hq", "modes", "mean_rmse", "mass_tv", "sw1_normalized", "mean_error", "covariance_error",
           "component_covariance_error", "component_core_covariance_error", "component_min_eigen_ratio",
           "component_core_min_eigen_ratio", "max_component_spill", "min_mass_ratio", "sample_count", "step",
           "cdf_ks", "mean_error_sigma", "std_ratio", "finite_fraction", "precision", "center_rms_sigma",
           "cov_trace_bias", "abs_cov_trace_bias", "radial_ks", "all_finite", "n", "valid_n"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve_summary(row, attempt):
    """Reduce complete curves to ranges/counts; retain bulk streams locally."""
    observations = row["evidence"].get("observations", [])
    selected = ("hq", "modes", "mean_rmse", "mass_tv", "sw1_normalized", "cdf_ks",
                "component_covariance_error", "component_core_covariance_error",
                "component_min_eigen_ratio", "max_component_spill", "mean_error")
    ranges = {}
    for metric in selected:
        values = [record[metric] for record in observations if isinstance(record.get(metric), (int, float))
                  and math.isfinite(record[metric])]
        if values:
            ranges[metric] = {"min": min(values), "max": max(values), "last": values[-1]}
    result = {"observations": len(observations), "metric_ranges": ranges,
              "declared_convergence": row["evaluator_result"].get("convergence")}
    metrics = row["evaluator_result"].get("metrics", [])
    if isinstance(metrics, list):
        bounds = []
        joint = [True] * len(observations)
        for bound in metrics:
            metric, threshold, op = bound["metric"], bound["threshold"], bound["op"]
            assert op in (">=", "<=", "==")
            values = [record.get(metric) for record in observations]
            passes = [isinstance(value, (int, float)) and math.isfinite(value) and
                      ((value >= threshold) if op == ">=" else (value <= threshold) if op == "<="
                       else value == threshold) for value in values]
            joint = [a and b for a, b in zip(joint, passes)]
            bounds.append({"metric": metric, "op": op, "threshold": threshold,
                           "passing_observations": sum(passes), "observations": len(values)})
        result["per_gate_observation_counts"] = bounds
        if bounds:
            longest, run = 0, 0
            for passed in joint:
                run = run + 1 if passed else 0
                longest = max(longest, run)
            result["joint_passing_checks"] = sum(joint)
            result["longest_joint_run"] = longest
            if result["declared_convergence"] is not None:
                assert sum(joint) == result["declared_convergence"]["passing_observations"]
    if row["task_id"] in ("grid100", "rotated100", "staggered100"):
        path = attempt / f"native100/{row['task_id']}/events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
        live = [event for event in events if event.get("event") == "eval" and event.get("model") == "live"]
        declaration = json.loads(path.with_name("summary.json").read_text())["accuracy"]
        terminal = [event for event in live if event["step"] in declaration["check_steps"]]
        assert [event["step"] for event in terminal] == declaration["check_steps"]
        precision = [event["metrics"]["precision"] for event in live]
        result["native_live_curve"] = {
            "artifact": str(path), "sha256": sha(path), "coverage_checks": len(live),
            "passing_coverage_checks": sum(bool(event["metrics"].get("passed")) for event in live),
            "precision_range": [min(precision), max(precision)],
            "terminal_accuracy_checks": len(terminal),
            "passing_terminal_accuracy_checks": sum(bool(event["accuracy"].get("passed")) for event in terminal),
            "terminal_precision_range": [min(event["accuracy"]["precision"] for event in terminal),
                                         max(event["accuracy"]["precision"] for event in terminal)],
            "undefined_terminal_shape_checks": sum(event["accuracy"].get("center_rms_sigma") is None
                                                   for event in terminal)}
    return result


def image_decomposition(path):
    records = torch.load(path, map_location="cpu", weights_only=False)
    record = records[-1]
    samples, targets = record["samples"].double(), record["targets"].double()
    distances = (samples[:, None] - targets[None]).square().mean((2, 3, 4))
    nearest = distances.argmin(1)
    error = samples - targets[nearest]
    support = targets[1, 0] > 0
    mean_error = error[:, 0, support].mean(1)
    mean_sse = support.sum() * mean_error.square()
    spatial_sse = error[:, 0, support].square().sum(1) - mean_sse
    background_sse = error[:, 0, ~support].square().sum(1)
    total = mean_sse + spatial_sse + background_sse
    misses = distances.min(1).values > .06**2
    spatial = error[:, 0, support] - mean_error[:, None]
    patterns = []
    for k in range(len(targets)):
        rows = spatial[nearest == k]
        energy = rows.square().sum()
        patterns.append({"level": k, "rows": len(rows),
                         "systematic_spatial_sse_fraction": float(len(rows) * rows.mean(0).square().sum() / energy)
                         if len(rows) and energy else None,
                         "first_spatial_pc_sse_fraction": float(torch.linalg.svdvals(rows)[0].square() / energy)
                         if len(rows) and energy else None})
    return {
        "artifact": str(path), "sha256": sha(path), "step": record["step"],
        "nearest_counts": nearest.bincount(minlength=len(targets)).tolist(),
        "clean_counts": nearest[~misses].bincount(minlength=len(targets)).tolist(),
        "miss_count": int(misses.sum()),
        "patch_mean_dominant_misses": int(((mean_sse > .5 * total) & misses).sum()),
        "per_level_spatial_pattern": patterns,
        "squared_error_fraction": {
            name: float(part.sum() / total.sum()) if total.sum() else 0.
            for name, part in (("patch_mean_offset", mean_sse),
                               ("within_patch_spatial_variation", spatial_sse),
                               ("background", background_sse))},
        "original_pixel_observation": record["metrics"],
    }


def image_saturation(path):
    records = torch.load(path, map_location="cpu", weights_only=False)
    selected_steps = {records[0]["step"], 100, 200, records[-1]["step"]}
    return {"artifact": str(path), "sha256": sha(path),
            "near_zero_threshold": 1e-6, "scope": "pixel diagnostics; original RMSE/mode gates unchanged",
            "milestones": [{"step": record["step"], "min": float(record["samples"].min()),
                            "max": float(record["samples"].max()), "mean": float(record["samples"].mean()),
                            "near_zero_pixel_fraction": float((record["samples"] < 1e-6).float().mean())}
                           for record in records if record["step"] in selected_steps]}


def width_decomposition(path, host):
    records = torch.load(path, map_location="cpu", weights_only=False)
    record = records[-1]
    samples = record["samples"].double()
    means = torch.tensor(host["means"], dtype=torch.float64)
    covariances = torch.tensor(host["covariances"], dtype=torch.float64)
    nearest = torch.cdist(samples, means).argmin(1)
    rows = []
    for k in range(len(means)):
        points = samples[nearest == k]
        if not len(points):
            rows.append({"component": k, "assigned": 0, "four_sigma_tail_count": 0,
                         "mean_displacement": None, "full_covariance_relative_error": None,
                         "full_trace_over_target": None, "tail_fraction_of_full_scatter": None,
                         "core_fraction_of_full_scatter": None, "max_mahalanobis_radius": None})
            continue
        delta = points - means[k]
        mahalanobis = torch.einsum("ni,ij,nj->n", delta, torch.linalg.inv(covariances[k]), delta)
        core = mahalanobis <= 16
        centered = points - points.mean(0)
        full_covariance = centered.T @ centered / len(points)
        trace = full_covariance.trace()
        # This exact additive trace split keeps every assigned point, including
        # the between-group mean separation: it never replaces the full gate.
        core_trace_contribution = centered[core].square().sum() / len(points)
        tail_trace_contribution = centered[~core].square().sum() / len(points)
        rows.append({
            "component": k, "assigned": len(points), "four_sigma_tail_count": int((~core).sum()),
            "mean_displacement": float((points.mean(0) - means[k]).norm()),
            "full_covariance_relative_error": float((full_covariance - covariances[k]).norm()
                                                     / covariances[k].norm()),
            "full_trace_over_target": float(trace / covariances[k].trace()),
            "tail_fraction_of_full_scatter": float(tail_trace_contribution / trace) if trace else 0.,
            "core_fraction_of_full_scatter": float(core_trace_contribution / trace) if trace else 0.,
            "max_mahalanobis_radius": float(mahalanobis.max().sqrt()) if len(points) else None,
        })
    return {"artifact": str(path), "sha256": sha(path), "step": record["step"], "components": rows}


@torch.no_grad()
def width_center_census(path, host):
    """One clean G(z) forward; no continuation, kernel jitter or resampling.

    This exact learned-center census is distinct from the served MoG score.
    Isolate and restore dummy constructor randomness before loading weights.
    """
    from lib.toy_models import SimpleMLPGenerator
    packet = torch.load(path, map_location="cpu", weights_only=False)
    trainer = packet["trainer"]
    assert trainer["completed_steps"] == host["steps"]
    assert packet["prior"]["standardize"] is False
    with torch.random.fork_rng(devices=[]):
        generator = SimpleMLPGenerator(host["z_dim"], host["hidden"], host["layers"], 2)
    generator.load_state_dict(trainer["models"]["G"])
    generator.eval()
    locations = trainer["models"]["prior"]["z"]
    output = generator(locations).double()
    means = torch.tensor(host["means"], dtype=torch.float64)
    covariances = torch.tensor(host["covariances"], dtype=torch.float64)
    nearest = torch.cdist(output, means).argmin(1)
    delta = output - means[nearest]
    radius2 = torch.einsum("ni,nij,nj->n", delta, torch.linalg.inv(covariances)[nearest], delta)
    return {"artifact": str(path), "sha256": sha(path), "forward_count": 1,
            "output_law": "G(learned location); no within-kernel perturbation; diagnostic only",
            "sigma": float(trainer["models"]["prior"]["sigma"]),
            "center_counts": nearest.bincount(minlength=len(means)).tolist(),
            "four_sigma_center_tail_counts": nearest[radius2 > 16].bincount(minlength=len(means)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True,
                        help="Artifact root containing nested queue/state.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.queue / "queue/state.json"
    state = json.loads(state_path.read_text())
    for request_id in REQUESTS.values():
        jobs = [job for job in state["jobs"].values() if request_id in job["subscribers"]]
        assert len(jobs) == 15 and all(job["status"] == "terminal" for job in jobs), "comparison incomplete"
    report = {"schema_version": 1, "scientific_source_digest": SOURCE,
              "analysis_source_sha256": sha(Path(__file__)),
              "scope": "saved-state diagnostic; zero updates/sampled-output draws; one endpoint G(z) forward per width arm; no ordinary qualification",
              "metric_projection": "Selected final scalar metrics; complete original rows remain in result.json under recorded hashes",
              "arms": {}}
    for name, request_id in REQUESTS.items():
        request_path = args.queue / f"queue/requests/{request_id}.json"
        request = json.loads(request_path.read_text())
        assert request["source"]["digest"] == SOURCE
        jobs = [job for job in state["jobs"].values() if request_id in job["subscribers"]]
        assert len(jobs) == 15 and all(job["status"] == "terminal" for job in jobs), "comparison incomplete"
        arm = {"request": request_id, "request_sha256": sha(request_path),
               "candidate": request["candidate"]["id"],
               "executed_origin_commit": request["source"]["origin_commit"], "tasks": {}}
        for job in jobs:
            attempt = Path(job["attempts"][-1]["path"])
            result_path = attempt / "result.json"
            result = json.loads(result_path.read_text())
            assert result["raw"]["grading"]["source_digest"] == SOURCE
            assert result["candidate_revision"] == request["candidate_revision"]
            for row in result["task_results"]:
                assert row["task_id"] not in arm["tasks"]
                info = {"attempt_id": result["attempt_id"], "result_sha256": sha(result_path),
                        "status": row["status"], "metrics": {key: value for key, value in row["metrics"].items()
                                                                if key in METRICS},
                        "evaluator_result": {key: value for key, value in row["evaluator_result"].items()
                                             if key != "metrics"},
                        "cost": {key: row["cost"][key] for key in
                                 ("execution_seconds", "completed_steps", "optimizer_updates")}}
                info["curve_summary"] = curve_summary(row, attempt)
                if row["task_id"] in ("grid100", "rotated100", "staggered100"):
                    evaluated = row["evaluator_result"]
                    info["evaluator_result"] = {key: evaluated[key] for key in
                                                ("coverage_status", "passed", "reason", "status", "final_metrics")}
                    info["evaluator_result"]["terminal_checks"] = len(evaluated["terminal_checks"])
                    info["evaluator_result"]["passing_terminal_checks"] = sum(
                        bool(check["passed"]) for check in evaluated["terminal_checks"])
                if row["task_id"] == "img_intensity2":
                    info["pixel_decomposition"] = image_decomposition(attempt / "observed-images.pt")
                if row["task_id"] == "img_blobs4":
                    info["pixel_saturation"] = image_saturation(attempt / "observed-images.pt")
                if row["task_id"] == "vector_unequal_width":
                    info["component_decomposition"] = width_decomposition(
                        attempt / "observed-samples.pt", row["evidence"]["host"]["definition"])
                    model_source = Path(__file__).resolve().parents[4] / "lib/toy_models.py"
                    assert sha(model_source) == request["source"]["files"]["lib/toy_models.py"]
                    info["center_census"] = width_center_census(
                        attempt / "provenance/provenance-state.pt", row["evidence"]["host"]["definition"])
                arm["tasks"][row["task_id"]] = info
        assert set(arm["tasks"]) == set(request["tasks"])
        arm["outcomes"] = {status: sum(row["status"] == status for row in arm["tasks"].values())
                           for status in sorted({row["status"] for row in arm["tasks"].values()})}
        report["arms"][name] = arm
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "tasks_each_arm": 15, "source": SOURCE}))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

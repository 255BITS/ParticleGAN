"""Read saved Ring16 evidence; no model construction, inference, or fresh draws."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.toy_audit.failure_diagnosis import vector_components


ATTEMPT = "b2d55fb3d5434002aa2b443522a83b79"
PREVIOUS_ATTEMPT = "2e6abacb6ffe4423be611480f4bb6496"
CANDIDATE = "bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def analyze(root, prior_root, inventory_root):
    inputs = {}

    def record(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, f"Changed input: {path}")
        inputs[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
        return path

    publication = read(record(root / "reports/forge/gaussian-smoke-inventory/final-v4/readout.json"))
    old_audit = read(record(root / "reports/forge/gaussian-smoke-inventory/ring-binding-audit.json"))
    record(Path(__file__).resolve())
    record(root / "benchmarks/toy_audit/failure_diagnosis.py")
    task = read(record(root / "configs/forge/tasks/ring16_acquisition.json"))
    spec = task["execution"]["host_definition"]
    selected = next(r for r in publication["candidate_rows"] if r["candidate_id"] == CANDIDATE)
    compact = next(t for t in selected["tasks"] if t["task_id"] == "ring16_acquisition")
    attempt = next(a for a in publication["attempt_receipts"] if a["attempt_id"] == ATTEMPT)
    originals = {}
    for name, receipt in attempt["original_files"].items():
        originals[name] = read(record(inventory_root / receipt["path"], receipt["sha256"]))
    row = originals["result"]["task_results"][0]
    request = originals["request"]["request"]
    local = Path(originals["evidence"]["local_artifact_root"])
    saved = row["evidence"]["saved_observer_outputs"]
    current_path = record(local / saved["path"], saved["sha256"])
    current = torch.load(current_path, map_location="cpu", weights_only=True)
    curve = row["evidence"]["observations"]
    certificate = compact["saved_state_certificate"]
    state_path = record(Path(certificate["artifact_root"]) / certificate["path"], certificate["sha256"])
    current_state = torch.load(state_path, map_location="cpu", weights_only=True)
    prior = prior_root / "runs/api/tier1-prior-duration-v1/mog100-n256-ring16_acquisition"
    historical = read(record(prior / "receipt.json", old_audit["original_files"][str(prior / "receipt.json")]["sha256"]))
    historical_source = read(record(prior / "source.json", historical["artifacts"]["source.json"]))
    historical_saved = torch.load(record(prior / "observations.pt", historical["artifacts"]["observations.pt"]),
                                  map_location="cpu", weights_only=True)
    historical_state = torch.load(record(prior / "state.pt", historical["artifacts"]["state.pt"]),
                                  map_location="cpu", weights_only=True)
    historical_curve = read(record(prior / "curve.json", historical["artifacts"]["curve.json"]))
    resume = read(record(prior / "resume-proof.json", historical["artifacts"]["resume-proof.json"]))
    previous_root = inventory_root / "reports/forge/attempts" / PREVIOUS_ATTEMPT
    previous_result = read(record(previous_root / "result.json",
        old_audit["original_files"][str(previous_root / "result.json")]["sha256"]))
    previous_evidence = read(record(previous_root / "evidence.json",
        old_audit["original_files"][str(previous_root / "evidence.json")]["sha256"]))
    previous_row = previous_result["task_results"][0]
    previous_samples = previous_row["evidence"]["saved_observer_outputs"]
    previous = torch.load(record(Path(previous_evidence["local_artifact_root"]) / previous_samples["path"],
                                 previous_samples["sha256"]), map_location="cpu", weights_only=True)

    steps = [r["step"] for r in curve]
    # The historical archive also retains an unscored initial GIF frame.
    # Compare the actual scored schedule, never that illustration.
    historical_saved = [r for r in historical_saved if r["step"] in steps]
    require(len(steps) == 96 and steps[-1] == 1600, "Incomplete current observation schedule")
    require(steps == [r["step"] for r in current] == [r["step"] for r in historical_saved]
            == [r["step"] for r in previous], "Observation schedules differ")
    require(steps == [r["step"] for r in historical_curve], "Historical curve schedule differs")
    require(historical_curve[-1]["metrics"] == historical["final_metrics"], "Historical endpoint changed")
    require(row["metrics"] == curve[-1], "Endpoint metrics differ from retained curve")
    require(compact["metrics"]["component_covariance_error"] == row["metrics"]["component_covariance_error"],
            "Compact endpoint changed")
    historical_matches = [a["step"] for a, b in zip(historical_saved, current)
                          if torch.equal(a["samples"], b["samples"])]
    previous_matches = [a["step"] for a, b in zip(previous, current)
                        if torch.equal(a["samples"], b["samples"])]
    require(historical_matches == [s for s in steps if s <= 400], "Historical prefix identity changed")
    require(previous_matches == steps, "v3/v4 saved outputs differ")
    require(previous_row["evidence"]["observations"] == curve, "v3/v4 metrics differ")
    require(historical_state["recipe"] == current_state["recipe"] and
            json.loads(json.dumps(current_state["recipe"])) == row["recipe"], "Recipe differs")
    require(historical_state["prior"] == current_state["prior"] == row["prior"], "Prior differs")
    require(historical_state["initialization"] == current_state["initialization"], "Initializer differs")
    require(historical_state["streams"]["manifest"] == current_state["streams"]["manifest"], "Stream declarations differ")
    require(historical_state["trainer"]["completed_steps"] == current_state["trainer"]["completed_steps"] == 1600,
            "Checkpoint update count differs")
    stream_matches = {k: torch.equal(v, current_state["streams"]["states"][k])
                      for k, v in historical_state["streams"]["states"].items()}
    require(all(stream_matches.values()) and len(stream_matches) == 14, "Final named cursors differ")
    scientific = {k: v for k, v in historical_source["files"].items()
                  if k.startswith(("particlegan/", "lib/")) or k in {
                      "experiments/forge/api.py", "experiments/forge/rng.py", "experiments/forge/vectorprofiles.py",
                      "benchmarks/transfer_suite/vector_tasks.py", "benchmarks/toy_audit/ring16_quality.py"}}
    changed = [k for k, v in scientific.items() if request["source"]["files"].get(k) != v]
    require(not changed and len(scientific) == 91, "Scientific source mismatch")
    require(task["evaluation"]["thresholds"] == request["tasks"]["ring16_acquisition"]["evaluation"]["thresholds"],
            "Current task gate differs from executed gate")

    def bounds_pass(metrics, thresholds):
        return all(metrics[k] >= v if op == ">=" else metrics[k] <= v for k, op, v in thresholds)

    def streak(values):
        longest = suffix = 0
        for value in values:
            suffix = suffix + 1 if value else 0
            longest = max(longest, suffix)
        return {"passing_checks": sum(values), "longest_streak": longest, "terminal_suffix": suffix}

    thresholds = task["evaluation"]["thresholds"]
    reduced = [b for b in thresholds if b[0] != "component_covariance_error"]
    full = streak([bounds_pass(p, thresholds) for p in curve])
    require(full["passing_checks"] == 0 and row["gate_status"] == "FAIL", "Current grade changed")
    require(historical["full_verdict"] == "PASS", "Historical grade changed")
    require(streak([bounds_pass(p["metrics"], thresholds) for p in historical_curve])["terminal_suffix"] == 6,
            "Historical passing suffix changed")
    decompositions = {}
    for label, observations, metrics in [("historical", historical_saved, historical["final_metrics"]),
                                        ("current", current, row["metrics"])]:
        points = observations[-1]["samples"].numpy().astype(np.float64)
        components = vector_components(points, spec, 1600)
        recomputed = np.mean([c["full_shape"]["covariance_error"] for c in components])
        require(np.isclose(recomputed, metrics["component_covariance_error"], rtol=1e-5, atol=1e-5),
                "Saved covariance reconstruction mismatch")
        means = np.asarray(spec["means"])
        labels = np.square(points[:, None] - means).sum(-1).argmin(1)
        distance = np.linalg.norm(points - means[labels], axis=1)
        extreme = distance > 2.
        decompositions[label] = {"components": components,
            "recomputed_float64_mean_covariance_error": float(recomputed),
            "mean_covariance_error_excluding_component_11": float(np.mean([
                c["full_shape"]["covariance_error"] for c in components if c["component"] != 11])),
            "component_11_share_of_sum_covariance_errors": float(components[11]["full_shape"]["covariance_error"] / (16 * recomputed)),
            "nearest_center_distance_quantiles": dict(zip(["p50", "p90", "p95", "p99", "max"],
                                                           np.quantile(distance, [.5, .9, .95, .99, 1.]).tolist())),
            "beyond_distance_counts": {str(v): int((distance > v).sum()) for v in [.3, .4, 1., 2.]},
            "beyond_2_distance_component_counts": {str(k): int(v) for k, v in Counter(labels[extreme]).items()},
            "beyond_2_distance_centroid": points[extreme].mean(0).tolist() if extreme.any() else None}
        require(max(c["covariance_decomposition_max_error"] for c in components) < 1e-12,
                "Variance decomposition failed")
    parameters = {}
    for role in ("G", "D", "prior"):
        old_model, new_model = (s["trainer"]["models"][role] for s in (historical_state, current_state))
        keys = [k for k, v in old_model.items() if isinstance(v, torch.Tensor) and v.is_floating_point()]
        parameters[role] = {"tensor_count": len(keys), "exact": all(torch.equal(old_model[k], new_model[k]) for k in keys),
            "maximum_absolute_difference": max(float((old_model[k] - new_model[k]).abs().max()) for k in keys)}
    roster = []
    for candidate in publication["candidate_rows"]:
        for cell in candidate["tasks"]:
            if cell["task_id"] == "ring16_acquisition":
                roster.append({"candidate_id": candidate["candidate_id"], "attempt_id": cell["attempt_id"],
                    "gate_status": cell["gate_status"], "raw_status": cell["raw_status"],
                    "passing_observations": cell["evaluator_summary"].get("convergence", {}).get("passing_observations"),
                    "metrics": {k: cell["metrics"][k] for k in ("modes", "hq", "mass_tv", "component_covariance_error",
                                                                "component_min_eigen_ratio") if k in cell["metrics"]}})
    require(Counter(c["gate_status"] for c in roster) == {"FAIL": 22, "BLOCKED": 1}, "Roster coverage changed")
    require(all(c["passing_observations"] == 0 for c in roster if c["gate_status"] == "FAIL"),
            "A completed current candidate has a full passing observation")
    return {"schema_version": 1, "scope": "saved_output_and_checkpoint_diagnosis", "qualification_input": False,
        "qualification_reuse": False, "training_updates_added": 0, "model_forwards_added": 0, "sampling_draws_added": 0,
        "analysis_device": "cpu", "analysis_device_reason": "Saved tensors, scalar math and receipt hashing only; original neural execution CUDA.",
        "base_commit": "2859975707eb3ea4d31958ad1b03e9e69e148a53", "source_commit": publication["source_commit"],
        "source_digest": publication["source_digest"], "attempt_id": ATTEMPT, "candidate_id": CANDIDATE,
        "historical_source_commit": historical_source["origin_commit"], "historical_resume": resume,
        "bindings": {"recipe_equal": True, "prior_equal": True, "initialization_equal": True,
                     "scientific_source_files_checked": len(scientific), "changed_scientific_files": changed,
                     "final_named_streams_equal": stream_matches,
                     "ambient_global_rng_equal": {k: torch.equal(historical_state["trainer"][k], current_state["trainer"][k])
                                                  for k in ("cpu_rng", "cuda_rng")}, "model_comparison": parameters},
        "sample_parity": {"historical_exact_prefix_observations": len(historical_matches),
                          "last_equal_step": max(historical_matches), "first_divergent_observation": 417,
                          "v3_v4_exact_observations": len(previous_matches), "v3_v4_metrics_equal": True},
        "current_curve": {"full_gate": full, "without_covariance_bound_diagnostic_only": streak([
                          bounds_pass(p, reduced) for p in curve]),
                          "minimum_covariance_error": min(curve, key=lambda p: p["component_covariance_error"])["component_covariance_error"],
                          "minimum_covariance_error_step": min(curve, key=lambda p: p["component_covariance_error"])["step"],
                          "cuts": [{k: p[k] for k in ("step", "modes", "hq", "mass_tv", "component_covariance_error")}
                                   for p in curve if p["step"] in (400, 800, 1200, 1600)]},
        "endpoints": {"historical": historical["final_metrics"], "current": row["metrics"]},
        "tail_decomposition": decompositions, "roster_status_counts": dict(Counter(c["gate_status"] for c in roster)),
        "roster": roster, "inputs": inputs,
        "limitations": ["Matching final RNG cursors does not prove intermediate learned-state equality or restore causality.",
                        "Ambient global RNG states differ; this comparison alone does not establish their relevance.",
                        "No retained row IDs link the extreme outputs to particular learned prior locations.",
                        "Float64 decompositions explain saved outputs and do not replace original float32 grades.",
                        "No post-1600 current uninterrupted trajectory or new qualification was measured."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.prior_root.resolve(), args.inventory_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"event": "saved_ring_analysis_complete", "output": str(args.output),
                      "training_updates_added": 0, "named_stream_matches": 14,
                      "exact_v3_v4_observations": 96, "ring_status_counts": result["roster_status_counts"]}), flush=True)

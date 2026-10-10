"""Compare saved ring receipts and arrays without models, sampling or regrading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def audit(prior_root, inventory_root):
    prefix = prior_root / "runs/api/tier1-prior-smoke-v1/mog100-n256/ring16_acquisition"
    duration = prior_root / "runs/api/tier1-prior-duration-v1/mog100-n256-ring16_acquisition"
    attempt_id = "2e6abacb6ffe4423be611480f4bb6496"
    original = inventory_root / "reports/forge/attempts" / attempt_id
    envelope, result, certificate = [read(original / (name + ".json")) for name in ("request", "result", "evidence")]
    request = envelope["request"]
    row = result["task_results"][0]
    local = Path(certificate["local_artifact_root"])
    old, continued = read(prefix / "receipt.json"), read(duration / "receipt.json")
    old_source, old_duration_source = read(prefix / "source.json"), read(duration / "source.json")
    old_task = read(prior_root / "reports/forge/tier1-prior-smoke/frozen-tasks/ring16_acquisition.json")
    new_task = request["tasks"]["ring16_acquisition"]
    old_curve, new_curve = read(duration / "curve.json"), row["evidence"]["observations"]
    old_saved = torch.load(duration / "observations.pt", map_location="cpu", weights_only=True)
    new_path = local / row["evidence"]["saved_observer_outputs"]["path"]
    if digest(new_path) != row["evidence"]["saved_observer_outputs"]["sha256"]:
        raise ValueError("current saved observation bytes differ from original receipt")
    old_by_step, new_by_step = ({r["step"]: r for r in points} for points in
        (old_saved, torch.load(new_path, map_location="cpu", weights_only=True)))
    steps = [point["step"] for point in old_curve]
    if steps != [point["step"] for point in new_curve] or len(steps) != 96:
        raise ValueError("expected identical complete 96-observation schedules")
    equal_metrics, equal_samples = [], []
    tensor_receipts = []
    old_prefix_digest, new_prefix_digest = hashlib.sha256(), hashlib.sha256()
    for old_row, new_row in zip(old_curve, new_curve):
        step = old_row["step"]
        a, b = old_by_step[step]["samples"], new_by_step[step]["samples"]
        if a.shape != b.shape or a.shape != (4096, 2) or a.dtype != b.dtype:
            raise ValueError("saved ring sample dimensions/dtypes differ")
        if old_row["metrics"] == {key: value for key, value in new_row.items() if key != "step"}:
            equal_metrics.append(step)
        if torch.equal(a, b):
            equal_samples.append(step)
        if step <= 400:
            old_prefix_digest.update(step.to_bytes(8, "little") + a.contiguous().numpy().tobytes())
            new_prefix_digest.update(step.to_bytes(8, "little") + b.contiguous().numpy().tobytes())
        if step in {17, 400, 417, 434, 1600}:
            distance = (a - b).norm(dim=1)
            tensor_receipts.append(dict(step=step, shape=list(a.shape), dtype=str(a.dtype),
                old_sample_bytes_sha256=tensor_digest(a), new_sample_bytes_sha256=tensor_digest(b),
                exact=torch.equal(a, b), median_point_displacement=float(distance.median()),
                maximum_point_displacement=float(distance.max())))
    scientific = {name: value for name, value in old_source["files"].items()
        if name.startswith(("particlegan/", "lib/")) or name in {
            "experiments/forge/api.py", "experiments/forge/rng.py", "experiments/forge/vectorprofiles.py",
            "benchmarks/transfer_suite/vector_tasks.py", "benchmarks/toy_audit/ring16_quality.py"}}
    changed = [name for name, value in scientific.items() if request["source"]["files"].get(name) != value]
    recipe_diff = {key: [continued["recipe"].get(key), row["recipe"].get(key)]
        for key in continued["recipe"].keys() | row["recipe"].keys()
        if continued["recipe"].get(key) != row["recipe"].get(key)}
    initialization = {role: old["initial"][role] == row["initialization"][role]
                      for role in ("generator", "discriminator", "prior")}
    paths = [prefix / name for name in ("receipt.json", "source.json", "state.pt")]
    paths += [duration / name for name in ("receipt.json", "source.json", "resume-proof.json", "observations.pt")]
    paths += [original / (name + ".json") for name in ("request", "evidence", "result")]
    paths.append(new_path)
    return dict(schema_version=1, qualification_input=False, scope="read_only_original_ring_binding_audit",
        archived_prefix=dict(completed_updates=old["completed_updates"], full_verdict=old["full_verdict"],
            source_commit=old_source["origin_commit"]),
        archived_continuation=dict(completed_updates=continued["completed_updates"], full_verdict=continued["full_verdict"],
            terminal_suffix=continued["full_terminal_suffix"], source_commit=old_duration_source["origin_commit"],
            resume=read(duration / "resume-proof.json")),
        current=dict(attempt_id=attempt_id, candidate_id=request["candidate"]["id"],
            source_commit=request["source"]["origin_commit"], source_digest=request["source"]["digest"],
            raw_status=result["raw"]["attempt_status"], gate_status=row["gate_status"],
            completed_updates=row["cost"]["completed_steps"], original_schedule_horizon=row["recipe"]["total_steps"]),
        bindings=dict(recipe_differences=recipe_diff, prior_equal=continued["prior"] == row["prior"],
            initializer_equal=initialization, named_rng_manifest_equal=old["rng"] == row["rng"],
            thresholds_equal=old_task["evaluation"]["thresholds"] == new_task["evaluation"]["thresholds"],
            all_96_scored_steps_equal=True, clean_samples_per_observation=4096,
            training_data_stream="data/target/training/cpu", evaluation_stream="eval/live/samples/cuda:0",
            scientific_source_files_checked=len(scientific), changed_scientific_source_files=changed,
            source_file_map_sha256=hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()),
        saved_comparison=dict(equal_sample_steps=equal_samples, equal_metric_steps=equal_metrics,
            old_400_prefix_sha256=old_prefix_digest.hexdigest(), new_400_prefix_sha256=new_prefix_digest.hexdigest(),
            first_divergent_observation=next(step for step in steps if step not in equal_samples),
            tensor_receipts=tensor_receipts),
        endpoints={label: {name: metrics[name] for name in ("modes", "hq", "mass_tv", "sw1_normalized",
            "component_covariance_error", "component_core_covariance_error", "component_min_eigen_ratio")}
            for label, metrics in (("archived", continued["final_metrics"]), ("current", row["metrics"]))},
        component_11_covariance_errors=dict(archived=continued["final_metrics"]["component_covariance_errors"][11],
            current=row["metrics"]["component_covariance_errors"][11]),
        observations_added=0, training_updates_added=0, sampling_draws_added=0,
        limitations=["Archived run restores a certified 400-update checkpoint; current run is uninterrupted.",
            "Current generic ring receipt saves scored arrays but no learned-model checkpoint or training-data digest.",
            "Post-400 learned-state and consumed-RNG equality cannot be established from current retained receipts.",
            "The observed divergence does not identify a causal restore or numerical defect."],
        original_files={str(path): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in paths})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ring-binding-audit.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.prior_root.resolve(), args.inventory_root.resolve()),
                                     sort_keys=True, indent=2, allow_nan=False) + "\n")

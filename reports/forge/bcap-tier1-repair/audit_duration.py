"""Compare paid duration diagnostics with byte-bound originals; never train.

Use retained observations and scored draws. Missing original checkpoints remain
an explicit evidence limit; a digest without its original counterpart is not
proof of complete training-state parity.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from benchmarks.toy_audit.gaussian1d_quality import score_samples as gaussian_score
from benchmarks.toy_audit.ring16_quality import score_samples as ring_score
from experiments.forge.contracts import file_hash, read_json, stable_hash
from experiments.forge.state import state_digest


ROOT = Path(__file__).resolve().parents[3]
ORIGINALS = {
    "gaussian1d_acquisition": "d0638ad5ce5a47e5b2fcb00b369768b2",
    "ring16_acquisition": "d8185ca486b54af79ef422395eac8065",
}
CANDIDATE = "bcap-original-horizon-diagnostics-v1"


def compare_prefix(old, new, old_records, new_records):
    """Compare all numeric values and tensor bytes, without tolerant matching."""
    points = new[:len(old)]
    samples = new_records[:len(old_records)]
    rows, max_metric_error, max_sample_error = [], 0., 0.
    for a, b, x, y in zip(old, points, old_records, samples, strict=True):
        assert a["step"] == x["step"] and b["step"] == y["step"]
        errors = [abs(a[k] - b[k]) for k in a.keys() & b.keys()
                  if type(a[k]) in (int, float) and type(b[k]) in (int, float)]
        metric_error = max(errors, default=0.)
        shape_same = x["samples"].shape == y["samples"].shape
        dtype_same = x["samples"].dtype == y["samples"].dtype
        sample_error = float((x["samples"] - y["samples"]).abs().max()) if shape_same else None
        identity = (shape_same and dtype_same and torch.equal(x["samples"], y["samples"]))
        rows.append({"step": a["step"], "observation_identical": a == b,
                     "samples_identical": identity, "max_numeric_metric_error": metric_error,
                     "max_sample_abs_error": sample_error})
        max_metric_error = max(max_metric_error, metric_error)
        if sample_error is not None:
            max_sample_error = max(max_sample_error, sample_error)
    return {"check_count": len(rows), "all_observations_identical": all(r["observation_identical"] for r in rows),
            "all_samples_identical": all(r["samples_identical"] for r in rows),
            "max_numeric_metric_error": max_metric_error, "max_sample_abs_error": max_sample_error,
            "checks": rows, "observations_sha256": stable_hash(points),
            "scored_samples_sha256": state_digest(samples)}


def failed_bounds(point, thresholds):
    compare = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}
    return [key for key, op, bound in thresholds
            if point.get(key) is None or not math.isfinite(point[key]) or not compare[op](point[key], bound)]


def load_attempt(directory):
    envelope = read_json(directory / "request.json")
    raw, graded = read_json(directory / "raw-result.json"), read_json(directory / "graded-result.json")
    request = envelope["request"]
    assert stable_hash(raw) == graded["raw_hash"]
    assert request["source"]["digest"] == graded["source_digest"] == stable_hash(request["source"]["files"])
    task = request["tasks"][envelope["job"]["task_id"]]
    receipt = raw["evidence"]["saved_observer_outputs"]
    path = directory / receipt["path"]
    assert file_hash(path) == receipt["sha256"] and path.stat().st_size == receipt["bytes"]
    assert receipt["optimizer_updates_added"] == receipt["sampling_draws_added"] == 0
    records = torch.load(path, map_location="cpu", weights_only=True)
    assert len(records) == receipt["observation_count"] == len(raw["evidence"]["observations"])
    return request, task, raw, graded, records


def audit(original, old_dir, new_dir, manifest):
    original_id = ORIGINALS[original]
    for name in ("request.json", "raw-result.json", "graded-result.json", "observed-samples.pt"):
        assert file_hash(old_dir / name) == manifest[f"attempts/{original_id}/{name}"]
    old_request, old_task, old, _, old_records = load_attempt(old_dir)
    request, task, new, graded, records = load_attempt(new_dir)
    assert request["candidate"]["id"] == CANDIDATE
    assert task["execution"]["budget_diagnostic"]["original_task"] == original
    assert old_request["candidate"]["recipe_overrides"] == request["candidate"]["recipe_overrides"]
    for key in ("recipe", "prior", "initializer", "initialization", "rng"):
        assert old[key] == new[key], key
    old_fields, new_fields = old["field_ownership"]["recipe_fields"], new["field_ownership"]["recipe_fields"]
    assert {k: v["value"] for k, v in old_fields.items()} == {k: v["value"] for k, v in new_fields.items()}
    for key in ("host", "sampling_law", "eval_output_noise", "sampling_contract_version"):
        assert old["evidence"][key] == new["evidence"][key], key
    assert old_request["runtime"] == request["runtime"]
    assert old_task["evaluation"]["thresholds"] == task["evaluation"]["thresholds"]
    critical = [k for k in old_request["source"]["files"]
                if k.startswith("particlegan/") or k in {
                    "experiments/forge/api.py", "experiments/forge/rng.py", "experiments/forge/vectorprofiles.py",
                    "benchmarks/toy_audit/gaussian1d_quality.py", "benchmarks/toy_audit/ring16_quality.py",
                    "benchmarks/transfer_suite/vector_tasks.py"}]
    changed = [k for k in critical if request["source"]["files"].get(k) != old_request["source"]["files"][k]]
    assert not changed, changed
    score = gaussian_score if original.startswith("gaussian") else ring_score
    max_recompute_error, max_gated_recompute_error = 0., 0.
    gated = {bound[0] for bound in task["evaluation"]["thresholds"]}
    for raw, draws in ((old, old_records), (new, records)):
        for point, record in zip(raw["evidence"]["observations"], draws, strict=True):
            measured = score(record["samples"], raw["evidence"]["host"]["definition"], record["step"])
            assert set(point) == {"step", *measured} and point["step"] == record["step"]
            for key, value in measured.items():
                if type(value) in (int, float):
                    error = abs(point[key] - value)
                    max_recompute_error = max(max_recompute_error, error)
                    if key in gated:
                        max_gated_recompute_error = max(max_gated_recompute_error, error)
                        assert point[key] == value, (record["step"], key)
                    else:
                        # SW1 projection reductions can vary slightly with CPU
                        # kernels. This tolerance never applies to prefix identity
                        # or any declared pass/fail metric.
                        assert math.isclose(point[key], value, rel_tol=1e-6, abs_tol=1e-7), (record["step"], key)
                else:
                    assert point[key] == value, (record["step"], key)
    prefix = compare_prefix(old["evidence"]["observations"], new["evidence"]["observations"], old_records, records)
    assert prefix["check_count"] == 24
    proof = new["evidence"]["budget_diagnostic"]["prefix"]
    assert proof["observations_sha256"] == prefix["observations_sha256"]
    assert proof["scored_samples_sha256"] == prefix["scored_samples_sha256"]
    state_comparison = {"verified": False, "reason": "original task produces_state=false; no original checkpoint retained",
                        "new_prefix_receipt": proof}
    old_state = old_dir / "state.pt"
    new_state = new_dir / "prefix-state.pt"
    if old_state.exists() and new_state.exists():
        a = torch.load(old_state, map_location="cpu", weights_only=True)
        b = torch.load(new_state, map_location="cpu", weights_only=True)
        for state in (a, b):
            state["trainer"].pop("max_steps", None)
        state_comparison = {"verified": True, "normalized_fields": ["trainer.max_steps"],
                            "complete_state_identical": state_digest(a) == state_digest(b),
                            "named_rng_identical": state_digest(a["streams"]) == state_digest(b["streams"])}
    thresholds = task["evaluation"]["thresholds"]
    summary_keys = gated | {"mean", "std", "component_core_covariance_error", "component_core_min_eigen_ratio",
                           "max_component_spill"}
    def compact(point):
        return {key: value for key, value in point.items() if key in summary_keys | {"step"}}
    terminal = [{**compact(point), "failed_bounds": failed_bounds(point, thresholds)}
                for point in new["evidence"]["observations"][-5:]]
    assert len(records) == 29
    assert new["cost"]["completed_steps"] == task["execution"]["steps"]
    assert new["cost"]["optimizer_updates"] == dict.fromkeys(("generator", "discriminator", "prior"), task["execution"]["steps"])
    assert new["evidence"]["guards"]["all_finite"] and new["evidence"]["guards"]["unintended_rng_deviations"] == 0
    actual = "PASS" if all(not p["failed_bounds"] for p in terminal) else "FAIL"
    assert graded["grades"][task["id"]]["gate_status"] == actual
    return {"original_attempt": original_id, "diagnostic_attempt": new_dir.name, "task_id": task["id"],
            "original_source_digest": old_request["source"]["digest"], "diagnostic_source_digest": request["source"]["digest"],
            "unchanged_critical_source_files": len(critical), "recipe_prior_initialization_rng_binding_runtime_identical": True,
            "original_budget": old_task["execution"]["steps"], "diagnostic_budget": task["execution"]["steps"],
            "schedule_horizon": new["recipe"]["total_steps"], "rescored_observations": 53,
            "max_recomputed_metric_error": max_recompute_error, "max_gated_recomputed_metric_error": max_gated_recompute_error,
            "prefix": prefix, "training_state_comparison": state_comparison,
            "terminal_checks": terminal, "original_final": compact(old["evidence"]["live"]), "gate_status": actual,
            "raw_sha256": file_hash(new_dir / "raw-result.json"), "graded_sha256": file_hash(new_dir / "graded-result.json"),
            "samples_sha256": file_hash(new_dir / "observed-samples.pt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("/home/martyn/dev/ParticleGAN/runs/forge/bcap-tier1-repair/queue"))
    parser.add_argument("--original-queue", type=Path, default=Path("/mnt/ml7tb/experiments/ParticleGAN/tier1-completion-v1/queue"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = torch.get_rng_state().clone()
    directories = {}
    for path in sorted((args.queue / CANDIDATE).glob("*/request.json")):
        envelope = read_json(path)
        if envelope["request"]["candidate"]["id"] != CANDIDATE:
            continue
        task = envelope["request"]["tasks"][envelope["job"]["task_id"]]
        original = task["execution"]["budget_diagnostic"]["original_task"]
        assert original not in directories, "multiple paid diagnostics for one original task"
        directories[original] = path.parent
    assert set(directories) == set(ORIGINALS), "both paid duration attempts must be available"
    manifest = read_json(ROOT / "reports/forge/tier1-completion/artifact-inventory.json")["manifest"]["original_receipts_sha256"]
    result = {"schema_version": 1, "id": "bcap-original-horizon-independent-audit-v1", "qualification_input": False,
              "training_updates_added": 0, "model_sampling_draws_added": 0,
              "tasks": {name: audit(name, args.original_queue / attempt, directories[name], manifest)
                        for name, attempt in ORIGINALS.items()}}
    assert torch.equal(rng, torch.get_rng_state())
    result["cpu_global_rng_unchanged"] = True
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"event": "duration_audit_complete", "output": str(args.output),
                      "gates": {k: v["gate_status"] for k, v in result["tasks"].items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

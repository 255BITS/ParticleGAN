"""Reduce saved word-factorial evidence; compare tensor bytes on CUDA, no updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash

REPORT = Path(__file__).parent
ARMS = ("truncated-serial", "full-serial", "truncated-threaded", "full-threaded")


@torch.no_grad()
def compare_tensors(left, right):
    """Compare saved observations/checkpoints without building a neural model."""
    count, differences, missing = 0, [], []
    def walk(a, b, path):
        nonlocal count
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            count += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                differences.append({"path": path, "shape_or_dtype": True})
            elif not torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                 b.contiguous().reshape(-1).view(torch.uint8)):
                differences.append({"path": path, "max_absolute_error": float((a.double() - b.double()).abs().max())})
        elif isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b), key=lambda x: (type(x).__name__, repr(x))):
                if key not in a or key not in b:
                    missing.append(path + "/" + str(key))
                else:
                    walk(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                missing.append(path + "/length")
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + "/" + str(index))
    walk(left, right, "")
    return {"comparison_device": "cuda:0", "equality_rule": "exact contiguous tensor bytes, including signed zero",
            "tensor_count": count, "equal_tensors": count - len(differences),
            "different_tensors": len(differences), "missing_paths": missing,
            "maximum_absolute_error": max((row.get("max_absolute_error", 0) for row in differences), default=0),
            "first_differences": differences[:8], "all_tensors_equal": not differences and not missing}


def load(path):
    return torch.load(path, map_location="cuda:0", weights_only=False)


def consumed_state(state):
    # These global RNG snapshots preserve caller restoration metadata. Public
    # word training consumes only the individually checkpointed named streams.
    fixture = dict(state["fixture"])
    api_state = dict(fixture["api_state"])
    api_state.pop("cpu_rng", None)
    api_state.pop("cuda_rng", None)
    fixture["api_state"] = api_state
    return {**state, "fixture": fixture}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/forge/bcap-word-regression")
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPORT / "readout.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for saved-tensor numerical comparisons")
    receipts = {arm: read_json(args.runs / arm / "compact-receipt.json") for arm in ARMS}
    audits = {arm: read_json(args.runs / arm / "factor-audit.json") for arm in ARMS}
    original = {"v4": read_json(args.runs / "historical-v4/adapter-receipt.json"),
                "v6": read_json(args.v6 / "adapter-receipt.json")}
    tensor_comparisons = {}
    for key, first, second in (
            ("current_control_vs_v6", args.runs / "truncated-serial", args.v6),
            ("historical_settings_vs_v4", args.runs / "full-threaded", args.runs / "historical-v4"),
            ("full_polar_modes", args.runs / "full-serial", args.runs / "full-threaded"),
            ("truncated_polar_modes", args.runs / "truncated-serial", args.runs / "truncated-threaded")):
        tensor_comparisons[key] = {
            "observed_outputs": compare_tensors(load(first / "observed-records.pt"), load(second / "observed-records.pt")),
            "full_checkpoint": compare_tensors(load(first / "state.pt"), load(second / "state.pt")),
            "models_optimizers_and_consumed_streams": compare_tensors(
                consumed_state(load(first / "state.pt")), consumed_state(load(second / "state.pt")))}
    comparisons = {}
    for name, a, b in (("truncation_serial", "full-serial", "truncated-serial"),
                       ("truncation_threaded", "full-threaded", "truncated-threaded"),
                       ("autograd_full", "full-serial", "full-threaded"),
                       ("autograd_truncated", "truncated-serial", "truncated-threaded")):
        states_a = {row["step"]: row for row in audits[a]["state_rows"]}
        states_b = {row["step"]: row for row in audits[b]["state_rows"]}
        rank_a = {(r["step"], r["call"]): r for r in audits[a]["rank_rows"]}
        rank_b = {(r["step"], r["call"]): r for r in audits[b]["rank_rows"]}
        first_state = next((step for step in sorted(states_a) if states_a[step]["model_state_sha256"] != states_b[step]["model_state_sha256"]), None)
        first_grad = next((key for key in sorted(set(rank_a) | set(rank_b))
            if rank_a.get(key, {}).get("gradient_sha256") != rank_b.get(key, {}).get("gradient_sha256")), None)
        first_update = next((key for key in sorted(set(rank_a) | set(rank_b))
            if rank_a.get(key, {}).get("update_sha256") != rank_b.get(key, {}).get("update_sha256")), None)
        comparisons[name] = {"arms": [a, b], "first_measured_state_difference": first_state,
            "first_measured_gradient_difference": first_grad, "first_measured_polar_difference": first_update,
            "same_observations": receipts[a]["observations"] == receipts[b]["observations"]}
    history = read_json(args.runs / "history-audit.json")
    result = {"schema_version": 1, "scope": "task_only_nonqualifying_causal_diagnostic",
        "qualification_input": False, "default_adoption": False, "ordinary_gates_unchanged": True,
        "task": "five_word_joint_acquisition", "candidate": read_json(REPORT / "protocol.json")["candidate_path"],
        "history_audit": history,
        "historical_observations": {era: raw["evidence"]["observations"] for era, raw in original.items()},
        "current_control_metadata_vs_v6": {key: receipts["truncated-serial"][key] == original["v6"].get(key)
            for key in ("initialization", "prior")},
        "current_control_observations_equal_v6": receipts["truncated-serial"]["observations"] == original["v6"]["evidence"]["observations"],
        "historical_settings_observations_equal_v4": receipts["full-threaded"]["observations"] == original["v4"]["evidence"]["observations"],
        "one_data_sequence": len({r["data_sequence_sha256"] for r in receipts.values()}) == 1,
        "one_initialization": len({stable_hash(r["initialization"]) for r in receipts.values()}) == 1,
        "one_recipe": len({r["resolved_recipe_sha256"] for r in receipts.values()}) == 1,
        "one_task": len({r["task_fingerprint"] for r in receipts.values()}) == 1,
        "comparisons": comparisons, "tensor_comparisons": tensor_comparisons,
        "arms": receipts,
        "rank_audit_interpretation": {
            arm: {"scope": "only sampled audited matrix updates at steps 1, 2 and scheduled scoring checkpoints",
                  "below_threshold_directions": sum(r["removed"] for r in audits[arm]["rank_rows"]),
                  "actual_removed_directions": sum(r["removed"] for r in audits[arm]["rank_rows"])
                      if receipts[arm]["arm"]["truncation"] else 0,
                  "available_directions": sum(r["available_rank"] for r in audits[arm]["rank_rows"]),
                  "matrix_updates_with_below_threshold_directions": sum(r["removed"] > 0 for r in audits[arm]["rank_rows"])}
            for arm in ARMS},
        "rank_receipt_field_note": "Frozen raw rank_summary removed fields count directions below the diagnostic cutoff in every arm. Full-polar arms retain those directions; actual removal is zero there.",
        "step1_rank_measurements": {arm: [r for r in audits[arm]["rank_rows"] if r["step"] == 1] for arm in ARMS},
        "paid_seconds": sum(r["cost"]["wall_seconds"] for r in receipts.values()), "reserved_seconds": 3600,
        "additional_training_updates_for_analysis": 0,
        "warning_counts": {arm: (args.runs / (arm + ".log")).read_text().count(
            "torch.linalg.svd: During SVD computation") for arm in ARMS},
        "source_artifact_hashes": {arm: {name: file_hash(args.runs / arm / name)
            for name in ("compact-receipt.json", "factor-audit.json")} for arm in ARMS}}
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "control_reproduces": result["current_control_observations_equal_v6"],
        "old_settings_reproduce": result["historical_settings_observations_equal_v4"], "comparisons": comparisons,
        "rows": {arm: r["grade"]["gate_status"] for arm, r in receipts.items()}}))


if __name__ == "__main__":
    main()

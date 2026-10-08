"""Verify and publish saved diagnostic outputs; adds no training or sampling."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess
import tarfile

import torch

from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.gaussian_tasks import bounds, grade
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import render

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ["current", "old_polar_serial", "truncated_parallel", "historical"]
EXECUTION_COMMITS = {name: ("0b3ef2e96" if name == "historical" else "f8153c516") for name in ARMS}


def load(path):
    return torch.load(path, weights_only=True, map_location="cuda:0")


def compare(reference, actual):
    left, right = [load(path / "evaluator/observed-samples.pt") for path in (reference, actual)]
    result = {"saved_observations": len(left), "primary_samples_bitexact": len(left) == len(right)
        and all(torch.equal(a["samples"], b["samples"]) for a, b in zip(left, right)),
        "confirmation_samples_bitexact": len(left) == len(right)
        and all(torch.equal(a["confirmation_samples"], b["confirmation_samples"]) for a, b in zip(left, right)),
        "primary_and_confirmation_metrics_exact": len(left) == len(right)
        and all(a["metrics"] == b["metrics"] and a["confirmation"]["metrics"] == b["confirmation"]["metrics"]
                for a, b in zip(left, right))}
    for label in ("initial", "final"):
        filename = "initial-state.pt" if label == "initial" else "state.pt"
        a, b = [load(path / "evaluator" / filename) for path in (reference, actual)]
        result[label] = {key + "_bitexact": state_digest(a["trainer"][key]) == state_digest(b["trainer"][key])
                         for key in ("models", "optimizers", "streams")}
        result[label]["all_named_streams_bitexact"] = state_digest(a["streams"]) == state_digest(b["streams"])
    if not all(result[key] for key in ("primary_samples_bitexact", "confirmation_samples_bitexact",
                                      "primary_and_confirmation_metrics_exact")):
        raise ValueError("historical observation reproduction failed")
    if not all(all(result[key].values()) for key in ("initial", "final")):
        raise ValueError("historical state reproduction failed")
    return result


def publish(runs, v4, v6):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("saved numerical state comparisons require one visible CUDA device")
    task = json.loads((HERE / "inputs.json").read_text())["tasks"]["gaussian1d_smoke"]
    raws = {name: json.loads((runs / name / "adapter-receipt.json").read_text()) for name in ARMS}
    trajectories = {name: json.loads((runs / name / "trajectory.json").read_text()) for name in ARMS}
    endpoints = {name: load(runs / name / "evaluator/state.pt") for name in ARMS}
    initials = {name: load(runs / name / "evaluator/initial-state.pt") for name in ARMS}
    for name in ARMS:
        if len(trajectories[name]) != 1000 or endpoints[name]["trainer"]["completed_steps"] != 1000:
            raise ValueError("diagnostic arm is incomplete")
        if raws[name]["gaussian_grade"] != grade(task, raws[name]["evidence"]):
            raise ValueError("saved grade differs from original declared evaluator")
    invariants = {
        "all_initial_models_bitexact": len({state_digest(state["trainer"]["models"]) for state in initials.values()}) == 1,
        "all_initial_named_streams_bitexact": len({state_digest(state["streams"]) for state in initials.values()}) == 1,
        "all_final_named_streams_bitexact": len({state_digest(state["streams"]) for state in endpoints.values()}) == 1,
        "same_data_digest": len({stable_hash(raw["evidence"]["data_sha256"]) for raw in raws.values()}) == 1,
        "same_resolved_recipe": len({stable_hash(raw["recipe"]) for raw in raws.values()}) == 1,
        "same_prior": len({stable_hash(raw["prior"]) for raw in raws.values()}) == 1,
        "all_1000_role_updates": all(raw["evidence"]["guards"]["optimizer_updates"] ==
            {"generator": 1000, "discriminator": 1000, "prior": 1000} for raw in raws.values()),
        "zero_rng_deviations": all(raw["evidence"]["guards"]["unintended_rng_deviations"] == 0 for raw in raws.values()),
        "finite_and_mechanisms_exercised": all(raw["evidence"]["guards"]["all_finite"] and
            raw["evidence"]["guards"]["hooks_exercised"] for raw in raws.values()),
    }
    if not all(invariants.values()):
        raise ValueError("matched-comparison invariants failed")
    comparisons = {"current_vs_archived_v6": compare(v6, runs / "current"),
                   "historical_vs_archived_v4": compare(v4, runs / "historical")}
    divergences = [{"left": left, "right": right,
        **{key: next((a["step"] for a, b in zip(trajectories[left], trajectories[right]) if a[key] != b[key]), None)
           for key in ("models_sha256", "gradients_sha256")}} for left, right in combinations(ARMS, 2)]
    arms = []
    media = {}
    for name in ARMS:
        raw = raws[name]
        original_summary = runs / name / "summary.json"
        summary = json.loads(original_summary.read_text()) if original_summary.exists() else None
        evidence, verdict = raw["evidence"], raw["gaussian_grade"]
        commit = subprocess.check_output(["git", "rev-parse", EXECUTION_COMMITS[name]], cwd=ROOT, text=True).strip()
        original_runner = subprocess.check_output(["git", "show", commit + ":reports/forge/bcap-gaussian-regression/run.py"], cwd=ROOT)
        import hashlib
        row = {"arm": name, "truncation": name in ("current", "truncated_parallel"),
            "autograd_multithreading_enabled": name in ("historical", "truncated_parallel"),
            "scope": "non_qualifying_causal_diagnostic", "qualification": False,
            "execution_commit": commit, "executed_runner_sha256": hashlib.sha256(original_runner).hexdigest(),
            "grade": verdict, "adapter_loop_seconds": raw["cost"]["adapter_loop_seconds"],
            "complete_arm_wall_seconds": summary["wall_seconds"] if summary else None,
            "primary_passes": [{"step": p["step"], "primary_cdf_ks": p["cdf_ks"],
                                "confirmation_cdf_ks": c["metrics"]["cdf_ks"],
                                "confirmed_full_pass": not bounds(c["metrics"])}
                               for p, c in zip(evidence["observations"], evidence["confirmations"]) if not bounds(p)],
            "mode_audit": summary["mode_audit"] if summary else None,
            "singular_rank_histogram": summary["singular_ranks"] if summary else None,
            "model_endpoint_sha256": trajectories[name][-1]["models_sha256"],
            "raw_receipt_sha256": file_hash(runs / name / "adapter-receipt.json"),
            "trajectory_sha256": file_hash(runs / name / "trajectory.json"),
            "observer_sha256": file_hash(runs / name / "evaluator/observed-samples.pt"),
            "final_checkpoint_sha256": file_hash(runs / name / "evaluator/state.pt"),
            "checkpoint_serial_backward_marker": endpoints[name]["trainer"]["serial_backward"]}
        if not summary:
            row["post_run_audit_error"] = {
                "type": "ValueError", "message": "actual forward/higher-order/update scheduling did not match arm",
                "timing": "after all 1000 updates, all samples, checkpoints, original grade and trajectory were saved",
                "cause": "the assertion included no-grad inference forwards after a lazy benchmarks import reset ambient autograd mode to False",
                "recovery": "metadata-only publication from saved original receipt/checkpoints; no repeated training or sampling",
                "missing": ["live counter totals", "singular rank histogram", "complete arm wall seconds"],
                "training_mode_evidence": "executed _execute_step explicitly scoped every entire _step to True; checkpoint serial_backward=False; identical factor enforcement passed the fourth-arm graph audit",
                "limitation": "actual per-call third-arm audit counters were not saved; no counts are reconstructed"}
        arms.append(row)
        media[name] = render(task, {"gate_status": verdict["status"], "evidence": evidence},
                             runs / name, HERE / "media" / (name + ".gif"))
    history = {}
    for name, path in (("v4", v4), ("v6", v6)):
        request = json.loads((path / "request.json").read_text())
        raw = json.loads((path / "raw-result.json").read_text())
        history[name] = {"attempt": path.name, "request_sha256": file_hash(path / "request.json"),
            "source_commit": request["request"]["source"]["origin_commit"],
            "source_digest": request["request"]["source"]["digest"],
            "runtime": request["request"]["runtime"], "compute": request["job"]["science"]["compute"],
            "recipe_sha256": stable_hash(raw["recipe"]), "execution_sha256": stable_hash(request["request"]["tasks"]["gaussian1d_smoke"]["execution"]),
            "evaluation_sha256": stable_hash(request["request"]["tasks"]["gaussian1d_smoke"]["evaluation"]),
            "initialization": raw["initialization"], "prior": raw["prior"],
            "sampling_law": raw["evidence"]["sampling_law"], "grade": raw["gaussian_grade"]}
    atomic_json(HERE / "readout.json", {"schema_version": 1, "scope": "non_qualifying_causal_diagnostic",
        "qualification": False, "protocol_sha256": file_hash(HERE / "protocol.json"),
        "inputs_sha256": file_hash(HERE / "inputs.json"), "arms": arms,
        "historical_audit": history, "first_post_update_divergences": divergences,
        "new_public_training_updates": 4000, "reserved_seconds": 480,
        "total_measured_adapter_loop_seconds": sum(raw["cost"]["adapter_loop_seconds"] for raw in raws.values()),
        "retries": 0, "new_seeds": 0, "inventory_qualification_changes": 0})
    atomic_json(HERE / "verification.json", {"schema_version": 1, "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0), "matched_arm_invariants": invariants,
        "archive_reproduction": comparisons, "media": media,
        "publisher_sha256": file_hash(Path(__file__)), "new_updates": 0, "new_sampling": 0,
        "note": "Global caller RNG envelopes differ because fresh processes are intentionally separate; model/optimizer and all consumed named streams reproduce exactly."})
    archive = ROOT / "artifacts/bcap-gaussian-regression-v1.tar.gz"
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        raise ValueError("do not overwrite a completed diagnostic archive")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(runs, arcname="bcap-gaussian-regression-v1")
        for name in ("run.py", "publish.py", "inputs.json", "protocol.json"):
            handle.add(HERE / name, arcname="reproduction/" + name)
    members = []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.isfile():
                payload = handle.extractfile(member).read()
                members.append({"path": member.name, "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()})
    atomic_json(HERE / "archive.json", {"schema_version": 1, "path": str(archive),
        "sha256": file_hash(archive), "bytes": archive.stat().st_size,
        "file_count": len(members), "members": members,
        "bulk_artifacts_committed": False, "new_training": 0})
    print(json.dumps({"published": str(HERE), "arms": [{"arm": row["arm"],
        "grade": row["grade"]["status"]} for row in arms], "all_invariants": all(invariants.values())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--v4-attempt-root", type=Path, required=True)
    parser.add_argument("--v6-attempt-root", type=Path, required=True)
    args = parser.parse_args()
    publish(args.runs.resolve(), args.v4_attempt_root.resolve(), args.v6_attempt_root.resolve())

"""Publish bounded saved-output evidence; zero training or new sampling.

Tensor byte comparisons and aggregation use retained CPU copies. No models,
derivatives, optimizer calls, SVD or new random draws are constructed here.
"""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest, require_same_formulation, require_optimizer_steps

REPORT = ROOT / "reports/forge/ring16-serialized-live"


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def compact_metrics(metrics, bounds):
    return {key: metrics[key] for key, _, _ in bounds}


def summarize(directory, bounds):
    receipt = read(directory / "receipt.json")
    curve = read(directory / "curve.json")
    first = receipt["first_full_pass"]
    after = [row for row in curve if first is not None and row["step"] > first]
    longest = current = 0
    for row in curve:
        current = current + 1 if row["full_pass"] else 0
        longest = max(longest, current)
    result = {key: receipt[key] for key in (
        "arm", "status", "error", "completed_updates", "elapsed_seconds", "scoring_draws",
        "batch_sequence_sha256", "confirmed_smoke", "first_full_pass", "terminal_suffix",
        "five_terminal_verdict")}
    result.update(final_metrics=compact_metrics(receipt["final_metrics"], bounds),
        final_failed_bounds=curve[-1]["failed_bounds"],
        full_passing_observations=sum(row["full_pass"] for row in curve),
        longest_full_passing_streak=longest,
        terminal_streak_start=curve[-receipt["terminal_suffix"]]["step"] if receipt["terminal_suffix"] else None,
        scheduled_observations=len(curve),
        post_first_pass={"scheduled_observations": len(after),
            "passing_observations": sum(row["full_pass"] for row in after),
            "failures": [{"step": row["step"], "failed_bounds": row["failed_bounds"],
                          "covariance_error": row["metrics"]["component_covariance_error"]}
                         for row in after if not row["full_pass"]]},
        confirmation=None if receipt["confirmation"] is None else {
            **{key: receipt["confirmation"][key] for key in ("step", "full_pass", "failed_bounds", "state_digest_before_draw")},
            "metrics": compact_metrics(receipt["confirmation"]["metrics"], bounds)},
        source=read(directory / "source.json"), receipt_sha256=file_hash(directory / "receipt.json"))
    return result


def project(state):
    value = deepcopy(state)
    for key in list(value["streams"]["states"]):
        binding = value["streams"]["manifest"]["bindings"][key]
        if (binding["component"], binding["purpose"]) in {
            ("ring16_intervention", "weak_directions"), ("live", "confirmation")}:
            del value["streams"]["states"][key]
            del value["streams"]["manifest"]["bindings"][key]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/api/ring16-serialized-live-v1")
    parser.add_argument("--prior-root", type=Path, default=Path("/home/martyn/dev/ParticleGAN/runs/api/ring16-restart-diagnostic-v1"))
    parser.add_argument("--runtime-root", type=Path, default=Path("/tmp/particlegan-ring16-runtime/runs/api/ring16-runtime-rounding-v1"))
    parser.add_argument("--truncation-root", type=Path, default=Path("/tmp/particlegan-ring16-truncation/runs/api/ring16-spectral-truncation-v1"))
    parser.add_argument("--noise-root", type=Path, default=Path("/tmp/particlegan-ring16-noise/runs/api/ring16-tiny-noise-v1"))
    parser.add_argument("--archive", type=Path, default=Path("/home/martyn/dev/ParticleGAN/artifacts/ring16-followup/ring16-serialized-live-v1-cuda-evidence.tar.gz"))
    args = parser.parse_args()
    protocol = read(REPORT / "protocol.json")
    bounds = protocol["gates"]["bounds"]
    assert read(args.run / "frozen-protocol.json") == protocol
    assert all(file_hash(ROOT / name) == digest for name, digest in protocol["bindings"].items())
    software = ROOT / "runs/software/ring16-serialized-live"
    software_receipt = read(software / "receipt.json")
    assert software_receipt["status"] == "PASS" and software_receipt["updates_debit"] == 3
    assert software_receipt["elapsed_seconds"] <= 30
    attempts = read(args.run / "controller-attempts.json")
    assert len(attempts) == 2 and all(row["status"] == "FINISHED" for row in attempts)
    assert all(row["child_wall_seconds"] <= 300 for row in attempts)
    arms, sources, verified = [], [], 0
    initial_models = None
    media = read(args.run / "media-index.json")
    for arm in protocol["schedules"]:
        directory = args.run / arm
        r = read(directory / "receipt.json")
        assert r["status"] == "COMPLETE" and r["completed_updates"] == 1600
        assert r["device"] == "cuda:0" and r["gpu_model"] == "NVIDIA RTX A6000"
        assert r["protocol_sha256"] == file_hash(REPORT / "protocol.json")
        assert r["intervention_calls"] == (1600 if arm == "every_step" else 1)
        for name, digest in r["artifacts"].items():
            assert file_hash(directory / name) == digest
            verified += 1
        summary = summarize(directory, bounds)
        sources.append(summary["source"])
        assert all(file_hash(ROOT / path) == digest for path, digest in summary["source"]["files"].items())
        initial, final = load(directory / "initial-state.pt"), load(directory / "state.pt")
        models = state_digest(initial["trainer"]["models"])
        if initial_models is None:
            initial_models = models
        assert models == initial_models
        assert final["trainer"].get("serial_backward", False) == (arm == "every_step")
        assert final["extensions"] == ({"serial_backward": True} if arm == "every_step" else {})
        require_same_formulation(initial, final)
        require_optimizer_steps(final, 1600)
        assert initial["recipe"] == final["recipe"]
        assert json.loads(json.dumps(final["recipe"])) == r["recipe"]
        assert initial["prior"] == final["prior"]
        assert initial["trainer"]["streams"].keys() == final["trainer"]["streams"].keys()
        assert state_digest(initial["streams"]["manifest"]) == state_digest(final["streams"]["manifest"])
        assert all(key in final["streams"]["states"] for key in final["streams"]["manifest"]["bindings"])
        # Serialization does not consume the shared harness's perturbation stream.
        for key, binding in final["streams"]["manifest"]["bindings"].items():
            if binding["component"] == "ring16_intervention":
                assert torch.equal(initial["streams"]["states"][key], final["streams"]["states"][key])
        curve = read(directory / "curve.json")
        snapshots = load(directory / "observations.pt")
        assert len(snapshots) == len(curve) == 96
        assert [row["step"] for row in curve] == [row["step"] for row in snapshots]
        assert all(state_digest(row["metrics"]) == state_digest(snap["metrics"])
                   for row, snap in zip(curve, snapshots))
        entry = next(row for row in media["entries"] if row["arm"] == arm)
        assert entry["source_sha256"] == file_hash(directory / "observations.pt")
        assert entry["gif_sha256"] == file_hash(directory / "actual-training.gif")
        assert Image.open(directory / "actual-training.gif").n_frames == 9
        shutil.copyfile(directory / "actual-training.gif", REPORT / (arm + ".gif"))
        # Publish compact identities; full source manifests remain in the archive.
        summary["source"] = {"origin_commit": summary["source"]["origin_commit"],
                             "digest": summary["source"]["digest"]}
        summary.update(initial_models_digest=models, final_context_digest=state_digest(final),
                       prefix_identity=r["prefix_identity"], boundary_identity=r["boundary_identity"], media=entry)
        arms.append(summary)
    assert sources[0] == sources[1]
    baseline = args.prior_root / "live/prefix-state.pt"
    assert file_hash(baseline) == protocol["baseline_checkpoint_sha256"]
    assert state_digest(project(load(args.run / "boundary_only/prefix-state.pt"))) == state_digest(load(baseline))
    c = project(load(args.run / "boundary_only/state401.pt"))
    c["trainer"]["max_steps"] = 401
    assert state_digest(c) == protocol["serialized401_state_digest"]
    assert state_digest(c) == state_digest(load(args.runtime_root / "fresh_serial401/state401.pt"))
    old = load(args.prior_root / "live/prefix-observations.pt")
    fresh = load(args.run / "boundary_only/observations.pt")[:24]
    assert all(a["step"] == b["step"] and torch.equal(a["samples"], b["samples"]) for a, b in zip(old, fresh))
    comparison = []
    for name, directory, pr in (("continuous_truncation", args.truncation_root / "every_step", 332),
                               ("continuous_weak_gradient_noise", args.noise_root / "every_step", 335)):
        summary = summarize(directory, bounds)
        init = load(directory / "initial-state.pt")
        assert state_digest(init["trainer"]["models"]) == initial_models
        assert init["recipe"] == load(args.run / "every_step/initial-state.pt")["recipe"]
        assert init["prior"] == load(args.run / "every_step/initial-state.pt")["prior"]
        assert summary["batch_sequence_sha256"] == arms[0]["batch_sequence_sha256"]
        assert [p["step"] for p in read(directory / "curve.json")] == [p["step"] for p in read(args.run / "every_step/curve.json")]
        # All public numerical implementation files match; the API gains only
        # the explicit typed serial flag, and driver/hook changes are declared.
        matching = [p for p in sources[0]["files"] if p.startswith("particlegan/")]
        assert all(sources[0]["files"][p] == summary["source"]["files"][p] for p in matching)
        original_protocol = read(directory.parent / "frozen-protocol.json")
        assert file_hash(directory.parent / "frozen-protocol.json") == read(directory / "receipt.json")["protocol_sha256"]
        for p in (protocol["task_path"], protocol["candidate_path"]):
            assert protocol["bindings"][p] == original_protocol["bindings"][p]
        for p in ("benchmarks/toy_audit/ring16_quality.py",
                  "benchmarks/toy_audit/reproducibility.py", "experiments/forge/vectorprofiles.py"):
            assert sources[0]["files"][p] == summary["source"]["files"][p]
        summary["source"] = {"origin_commit": summary["source"]["origin_commit"],
                             "digest": summary["source"]["digest"],
                             "receipt_sha256": file_hash(directory / "source.json")}
        summary.update(technique=name, source_pr=f"https://github.com/255BITS/ParticleGAN/pull/{pr}",
                       repeated_training=False, initial_models_digest=initial_models,
                       scope="original source-bound diagnostic; no qualification transfer")
        comparison.append(summary)
    assert arms[0]["batch_sequence_sha256"] == arms[1]["batch_sequence_sha256"]
    assert arms[0]["batch_sequence_sha256"] == "94734673d1a3559a725f58ac2a33456d4063995bb312591613c381e828121750"
    training_wall = sum(row["child_wall_seconds"] for row in attempts)
    assert training_wall <= protocol["max_reserved_seconds"]
    assert sum(row["scoring_draws"] for row in arms) == 194
    # Archive every retained raw file with an in-archive exact member manifest.
    raw = [*sorted(p for p in args.run.rglob("*") if p.is_file()),
           *sorted(p for p in software.rglob("*") if p.is_file()),
           ROOT / "runs/reports/ring16-serialized-live/execution.log"]
    members = {str(p.relative_to(ROOT)): file_hash(p) for p in raw}
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.archive.exists():
        with tarfile.open(args.archive, "r:gz") as archive:
            assert json.load(archive.extractfile("MANIFEST.json")) == members
    else:
        with tarfile.open(args.archive, "w:gz") as archive:
            for p in raw:
                archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
            payload = (json.dumps(members, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo("MANIFEST.json"); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    import hashlib
    with tarfile.open(args.archive, "r:gz") as archive:
        assert set(archive.getnames()) == {*members, "MANIFEST.json"}
        for name, digest in members.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
    archive_receipt = {"archive_path": str(args.archive), "sha256": file_hash(args.archive),
                       "bytes": args.archive.stat().st_size, "original_files": len(members),
                       "member_count": len(members) + 1, "all_member_hashes_verified": True}
    results = {"schema_version": 1, "protocol_id": protocol["id"], "qualification_input": False,
        "qualification_reuse": False, "scope": "completed_cuda_ring16_serialization_quality_diagnostic",
        "protocol_sha256": file_hash(REPORT / "protocol.json"), "physical_gpu": 0,
        "arms": arms, "existing_comparisons": comparison,
        "cost": {"attempts": 2, "scientific_retries": 0, "new_training_updates": 3200,
                 "new_scored_draws": 194, "training_child_wall_seconds": training_wall,
                 "software_updates": 3, "software_actual_seconds": software_receipt["elapsed_seconds"],
                 "charged_seconds_with_full_software_reservation": training_wall + 30,
                 "total_reserved_seconds": 630},
        "archive": archive_receipt,
        "comparison_scope": "Matched initial tensors/recipe/prior/batches/cadence/gates and unchanged public numerical package; explicit trainer/runtime deltas and original source identities retained. No reruns or ordinary qualification."}
    atomic_json(REPORT / "results.json", results)
    atomic_json(REPORT / "archive.json", archive_receipt)
    atomic_json(REPORT / "verification.json", {"schema_version": 1, "qualification_input": False,
        "new_training_updates": 0, "new_model_calls": 0, "new_sampling_draws": 0,
        "compute": "Saved CPU tensor-byte equality/metadata and previously generated media; no scientific CPU SVD or models",
        "receipt_artifacts_verified": verified, "executed_source_files_verified": len(sources[0]["files"]),
        "frozen_bindings": "PASS", "both_complete1600": "PASS", "boundary400_exact": "PASS",
        "boundary401_exact_third_trajectory": "PASS", "24_boundary_prefix_samples_exact": "PASS",
        "initial_models_across_serialization_truncation_noise": "PASS", "same_recipe_prior_target_batches_cadence": "PASS",
        "all_public_numerical_package_files_unchanged_vs_comparisons": "PASS",
        "checkpointed_named_streams_and_modes": "PASS", "unused_perturbation_stream_unchanged": "PASS",
        "actual_training_media": media,
        "software": {**software_receipt, "raw_log_sha256": file_hash(software / "pytest.log")},
        "archive": archive_receipt, "verifier_sha256": file_hash(Path(__file__))})
    print(json.dumps({"event": "saved_evidence_verified", "artifacts": verified,
                      "training_child_wall_seconds": training_wall, "archive": archive_receipt}), flush=True)


if __name__ == "__main__":
    main()

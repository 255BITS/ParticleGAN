"""Verify and publish retained combined-run evidence; no new neural work."""
import argparse
import hashlib
import importlib.util
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

REPORT = ROOT / "reports/forge/ring16-serialized-truncation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/api/ring16-serialized-truncation-v1")
    parser.add_argument("--serialization-root", type=Path, default=Path("/tmp/particlegan-ring16-serialized-live"))
    parser.add_argument("--truncation-root", type=Path, default=Path("/tmp/particlegan-ring16-truncation"))
    parser.add_argument("--archive", type=Path, default=Path("/home/martyn/dev/ParticleGAN/artifacts/ring16-followup/ring16-serialized-truncation-v1-cuda-evidence.tar.gz"))
    args = parser.parse_args()
    # Reuse the published saved-output reducer from PR337, with byte binding.
    reducer_path = args.serialization_root / "reports/forge/ring16-serialized-live/verify_saved.py"
    previous_verification = json.loads((args.serialization_root / "reports/forge/ring16-serialized-live/verification.json").read_text())
    assert file_hash(reducer_path) == previous_verification["verifier_sha256"]
    spec = importlib.util.spec_from_file_location("ring16_saved_reducer", reducer_path)
    reducer = importlib.util.module_from_spec(spec); spec.loader.exec_module(reducer)
    read, load = reducer.read, reducer.load
    protocol = read(REPORT / "protocol.json")
    assert read(args.run / "frozen-protocol.json") == protocol
    assert all(file_hash(ROOT / p) == digest for p, digest in protocol["bindings"].items())
    attempts = read(args.run / "controller-attempts.json")
    assert len(attempts) == 1 and attempts[0]["status"] == "FINISHED"
    wall = attempts[0]["child_wall_seconds"]
    assert wall <= 300
    directory = args.run / "every_step"
    receipt = read(directory / "receipt.json")
    assert receipt["status"] == "COMPLETE" and receipt["completed_updates"] == 1600
    assert receipt["device"] == "cuda:0" and receipt["gpu_model"] == "NVIDIA RTX A6000"
    assert receipt["intervention_calls"] == 9600
    assert receipt["protocol_sha256"] == file_hash(REPORT / "protocol.json")
    for name, digest in receipt["artifacts"].items():
        assert file_hash(directory / name) == digest
    initial, final = load(directory / "initial-state.pt"), load(directory / "state.pt")
    require_same_formulation(initial, final); require_optimizer_steps(final, 1600)
    assert initial["trainer"]["serial_backward"] is final["trainer"]["serial_backward"] is True
    assert initial["extensions"] == final["extensions"] == {"serial_backward": True}
    for key, binding in final["streams"]["manifest"]["bindings"].items():
        if binding["component"] == "ring16_intervention":
            assert torch.equal(initial["streams"]["states"][key], final["streams"]["states"][key])
    bounds = protocol["gates"]["bounds"]
    result = reducer.summarize(directory, bounds)
    source = result["source"]
    assert all(file_hash(ROOT / p) == digest for p, digest in source["files"].items())
    result["source"] = {"origin_commit": source["origin_commit"], "digest": source["digest"],
                        "source_receipt_sha256": file_hash(directory / "source.json")}
    initial_models = state_digest(initial["trainer"]["models"])
    curve = read(directory / "curve.json"); snapshots = load(directory / "observations.pt")
    assert len(curve) == len(snapshots) == 96
    assert all(row["step"] == snapshot["step"] and state_digest(row["metrics"]) == state_digest(snapshot["metrics"])
               for row, snapshot in zip(curve, snapshots))
    result.update(initial_models_digest=initial_models, final_context_digest=state_digest(final),
                  serial_scope_guard_calls=receipt["intervention_calls"])
    comparisons = []
    for name, prior, pr in (("continuous_serialization", args.serialization_root / "runs/api/ring16-serialized-live-v1/every_step", 337),
                            ("continuous_truncation", args.truncation_root / "runs/api/ring16-spectral-truncation-v1/every_step", 332)):
        old = reducer.summarize(prior, bounds); old_source = old["source"]
        previous = load(prior / "initial-state.pt")
        assert state_digest(previous["trainer"]["models"]) == initial_models
        assert previous["recipe"] == initial["recipe"] and previous["prior"] == initial["prior"]
        assert previous["initializer"] == initial["initializer"] and previous["initialization"] == initial["initialization"]
        assert old["batch_sequence_sha256"] == result["batch_sequence_sha256"]
        assert [row["step"] for row in read(prior / "curve.json")] == [row["step"] for row in curve]
        old_protocol = read(prior.parent / "frozen-protocol.json")
        for p in (protocol["task_path"], protocol["candidate_path"]):
            assert old_protocol["bindings"][p] == protocol["bindings"][p]
        for p in [p for p in source["files"] if p.startswith("particlegan/")] + ["benchmarks/toy_audit/ring16_quality.py", "experiments/forge/vectorprofiles.py", "benchmarks/toy_audit/reproducibility.py"]:
            assert old_source["files"][p] == source["files"][p]
        if name == "continuous_serialization":
            assert old_source["files"]["experiments/forge/api.py"] == source["files"]["experiments/forge/api.py"]
        else:
            assert old_source["files"]["benchmarks/toy_audit/ring16_spectral_truncation.py"] == source["files"]["benchmarks/toy_audit/ring16_spectral_truncation.py"]
        old["source"] = {"origin_commit": old_source["origin_commit"], "digest": old_source["digest"],
                         "source_receipt_sha256": file_hash(prior / "source.json")}
        old.update(technique=name, source_pr=f"https://github.com/255BITS/ParticleGAN/pull/{pr}",
                   repeated_training=False, initial_models_digest=initial_models)
        comparisons.append(old)
    assert result["batch_sequence_sha256"] == "94734673d1a3559a725f58ac2a33456d4063995bb312591613c381e828121750"
    media = read(args.run / "media-index.json")
    assert len(media["entries"]) == 1
    entry = media["entries"][0]
    assert entry["source_sha256"] == file_hash(directory / "observations.pt")
    assert entry["gif_sha256"] == file_hash(directory / "actual-training.gif")
    assert Image.open(directory / "actual-training.gif").n_frames == 9
    shutil.copyfile(directory / "actual-training.gif", REPORT / "actual-training.gif")
    raw = [*sorted(p for p in args.run.rglob("*") if p.is_file()), ROOT / "runs/reports/ring16-serialized-truncation/execution.log"]
    members = {str(p.relative_to(ROOT)): file_hash(p) for p in raw}
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.archive.exists():
        with tarfile.open(args.archive, "w:gz") as archive:
            for p in raw:
                archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
            payload = (json.dumps(members, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo("MANIFEST.json"); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(args.archive, "r:gz") as archive:
        assert json.load(archive.extractfile("MANIFEST.json")) == members
        assert set(archive.getnames()) == {*members, "MANIFEST.json"}
        assert all(hashlib.sha256(archive.extractfile(p).read()).hexdigest() == digest for p, digest in members.items())
    archived = {"archive_path": str(args.archive), "sha256": file_hash(args.archive), "bytes": args.archive.stat().st_size,
                "original_files": len(members), "member_count": len(members) + 1, "all_member_hashes_verified": True}
    results = {"schema_version": 1, "protocol_id": protocol["id"], "scope": "completed_cuda_ring16_serialized_truncation_diagnostic",
               "qualification_input": False, "qualification_reuse": False, "physical_gpu": 0,
               "protocol_sha256": file_hash(REPORT / "protocol.json"), "arm": result, "existing_comparisons": comparisons,
               "cost": {"attempts": 1, "scientific_retries": 0, "new_training_updates": 1600,
                        "new_scored_draws": receipt["scoring_draws"], "whole_training_subprocess_seconds": wall,
                        "reserved_seconds": 300, "new_software_training_updates": 0,
                        "empirical_speed_comparison": False},
               "archive": archived, "media": media}
    atomic_json(REPORT / "results.json", results); atomic_json(REPORT / "archive.json", archived)
    atomic_json(REPORT / "verification.json", {"schema_version": 1, "qualification_input": False,
        "new_training_updates": 0, "new_model_calls": 0, "new_sampling_draws": 0,
        "compute": "Saved CPU tensor-byte comparisons/metrics/media, no models or scientific CPU SVD",
        "frozen_bindings": "PASS", "executed_source_files_verified": len(source["files"]),
        "receipt_artifacts_verified": len(receipt["artifacts"]), "completed1600": "PASS",
        "checkpointed_named_streams_and_serial_mode": "PASS", "serialized_polar_scope_guards": 9600,
        "unchanged_perturbation_stream": "PASS", "matched_initial_models_recipe_prior_batches_cadence_gates": "PASS",
        "unchanged_public_numerical_package": "PASS", "exact_original_truncation_helper": "PASS",
        "reused_serialization_api_software_checks": previous_verification["software"],
        "reused_reducer": {"source_pr": "https://github.com/255BITS/ParticleGAN/pull/337", "sha256": file_hash(reducer_path)},
        "archive": archived, "media": media, "verifier_sha256": file_hash(Path(__file__))})
    print(json.dumps({"event": "saved_evidence_verified", "first_pass": result["first_full_pass"],
                      "confirmed_smoke": result["confirmed_smoke"], "terminal_suffix": result["terminal_suffix"],
                      "whole_training_subprocess_seconds": wall, "archive": archived}), flush=True)


if __name__ == "__main__":
    main()

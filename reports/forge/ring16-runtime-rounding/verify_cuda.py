"""Verify source-bound completed CUDA evidence using retained files only."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    report = repo / "reports/forge/ring16-runtime-rounding"
    protocol = json.loads((report / "protocol.json").read_text())
    result = json.loads((report / "cuda-results.json").read_text())
    campaign = json.loads((args.root / "campaign-summary.json").read_text())
    assert campaign["status"] == "COMPLETE" and campaign["attempts"] == 4
    assert campaign["new_updates_debit"] == 804
    assert campaign["seconds_debit"] <= protocol["max_reserved_seconds"]
    assert all(file_hash(repo / path) == digest for path, digest in protocol["bindings"].items())
    verified_artifacts, source_files = 0, None
    for arm in protocol["arms"]:
        receipt = json.loads((args.root / arm / "receipt.json").read_text())
        assert receipt["completed_updates"] == 401 and receipt["device"] == "cuda:0"
        assert receipt["source_commit"] == "fa60ca4af1fc872e650a6c4b13ca35d0a755a0ca"
        assert receipt["prefix_bit_exact"] and not receipt["qualification_input"]
        assert receipt["protocol_sha256"] == file_hash(report / "protocol.json")
        for path, digest in receipt["artifacts"].items():
            assert file_hash(args.root / arm / path) == digest
            verified_artifacts += 1
        prefix = torch.load(args.root / arm / "prefix-state.pt", weights_only=True, map_location="cpu")
        assert state_digest(prefix) == protocol["prefix_state_digest"]
        source = json.loads((args.root / arm / "source.json").read_text())
        assert source["digest"] == receipt["source_digest"]
        if source_files is None:
            source_files = source["files"]
        assert source["files"] == source_files
    assert all(file_hash(repo / path) == digest for path, digest in source_files.items())
    assert all(v["bit_exact"] for row in result["instrumentation_cohort_parity"].values() for v in row.values())
    for row in result["comparisons"].values():
        assert row["real_batch_bit_exact"] and row["all_named_stream_states_exact"]
        assert all(row["first_six_forward_inputs_bit_exact"]) and all(row["first_six_forward_outputs_bit_exact"])
        assert row["critic_graph_topology_exact"]
    ordinary = result["comparisons"]["ordinary_live_vs_restored"]
    serial = result["comparisons"]["serialized_live_vs_restored"]
    assert not ordinary["whole_context401_exact"] and serial["whole_context401_exact"]
    assert ordinary["critic_graph_sequence_relations"]["changed_relative_relations"] == 1054
    assert serial["critic_graph_sequence_relations"]["changed_relative_relations"] == 0
    assert all(v["bit_exact"] for v in serial["critic_gradient"].values())
    assert all(not result["comparisons"][name]["whole_context401_exact"] for name in
               ("live_serial_effect", "restored_serial_effect"))
    historical = torch.load(args.prior_root / "live/prefix-observations.pt", weights_only=True, map_location="cpu")
    assert len(historical) == 24
    for arm in ("fresh_graph", "fresh_serial401"):
        observations = torch.load(args.root / arm / "observations.pt", weights_only=True, map_location="cpu")
        assert len(observations) == 24
        assert all(a["step"] == b["step"] and torch.equal(a["samples"], b["samples"])
                   for a, b in zip(observations, historical))
    media = json.loads((report / "media/index.json").read_text())
    gif = report / "media" / media["gif"]["file"]
    assert file_hash(gif) == media["gif"]["sha256"]
    assert Image.open(gif).n_frames == media["gif"]["frames"] == 9
    assert media["gif"]["last_frame_update"] == 400
    assert all(file_hash(Path(path)) == digest for path, digest in media["source_artifacts"].items())
    assert media["gif"]["maximum_absolute_prefix_sample"] < max(media["gif"]["axes"])
    include = Path(torch.__file__).resolve().parent / "include"
    header_lines = {"ATen/SequenceNumber.h": [7], "torch/csrc/autograd/node.h": [117, 151, 349],
                    "torch/csrc/autograd/engine.h": [102], "torch/csrc/autograd/input_buffer.h": [52]}
    headers = {name: {"path": str(include / name), "sha256": file_hash(include / name), "cited_lines": lines}
               for name, lines in header_lines.items()}
    archive = json.loads((report / "archive.json").read_text())
    assert file_hash(Path(archive["archive_path"])) == archive["sha256"]
    verification = {"scope": "saved_evidence_source_and_metadata_verification", "qualification_input": False,
        "new_neural_updates": 0, "new_model_calls": 0, "new_random_draws": 0,
        "cuda_arms_completed": 4, "campaign_new_updates": 804,
        "campaign_whole_process_seconds": campaign["seconds_debit"],
        "receipt_artifacts_verified": verified_artifacts, "executed_source_files_verified": len(source_files),
        "all_frozen_bindings": "PASS", "all_four_full400_contexts": "PASS",
        "all24_fresh_prefix_samples_vs_historical": "PASS", "historical_ordinary_gradient_parity": "PASS",
        "serial_full401_context_equality": "PASS", "serial_is_third_trajectory": "PASS",
        "full1600_quality": "INCOMPLETE: not executed in this protocol",
        "media": {"sha256": file_hash(gif), "frames": 9, "end_update": 400, "all_points_within_fixed_axes": True},
        "torch_version": torch.__version__, "installed_primary_headers": headers,
        "archive_sha256": archive["sha256"], "archive_original_files": archive["original_files"],
        "source_scripts": {name: file_hash(report / name) for name in
                           ("compare_boundary.py", "render_prefix.py", "archive_raw.py", "verify_cuda.py")}}
    atomic_json(args.output, verification)
    print(json.dumps({"event": "cuda_evidence_verified", "source_files": len(source_files),
                      "artifacts": verified_artifacts, "new_model_calls": 0}), flush=True)


if __name__ == "__main__":
    main()

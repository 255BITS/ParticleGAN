"""Archived proof integrity and software-only media boundaries; no training."""
import json

import pytest
import torch

from benchmarks.toy_audit import source_cuda_observation_report as report


def prefix_fixture(directory):
    raw = directory / "prefix-observed/raw"
    raw.mkdir(parents=True)
    owners = dict(owners=dict(ids=(torch.tensor([2, 2, 4]), False, None)))
    state, cloud = raw / "state-000.pt", raw / "cloud-000.npz"
    torch.save(owners, state)
    cloud.write_bytes(b"retained raw cloud fixture")
    digest = report.digest(owners)
    row = dict(step=1, owner_state_sha256=digest, complete_owner_and_all_visible_cuda_rng_pure=True,
               state_sha256=report.sha(state), cloud_sha256=report.sha(cloud))
    (raw / "observations.jsonl").write_text(json.dumps(row) + "\n")
    execution = dict(blocked_phase="observer parity prerequisite",
                     prefix_baseline=dict(completed_updates=3, boundaries={"0": "same-init", "1": "baseline-differs"}),
                     prefix_observed=dict(completed_updates=3, boundaries={"0": "same-init", "1": digest}),
                     prefix_parity=dict(software_updates=6))
    return execution, row, raw


def test_before_after_proof_is_bound_to_saved_owners(tmp_path):
    execution, _, _ = prefix_fixture(tmp_path)
    diagnosis = report.prefix_diagnosis(tmp_path, execution)
    assert diagnosis["divergence_predates_first_read"]
    assert diagnosis["first_read_preserved_tracked_owners_and_rngs"]
    assert diagnosis["software_updates"] == 6
    assert diagnosis["root_mechanism"] == "UNRESOLVED"
    assert diagnosis["final_generator_batch_prior_id_multiplicity"] == dict(
        rows=3, unique_rows=2, repeated_rows=1, maximum_multiplicity=2)
    assert diagnosis["baseline_owner_sha256"] != diagnosis["observed_before_read_owner_sha256"]
    assert diagnosis["observed_before_read_owner_sha256"] == diagnosis["independently_recomputed_checkpoint_owner_sha256"]


def test_checkpoint_contents_cannot_be_replaced_with_a_new_file_hash(tmp_path):
    execution, row, raw = prefix_fixture(tmp_path)
    torch.save(dict(owners={}), raw / "state-000.pt")
    row["state_sha256"] = report.sha(raw / "state-000.pt")
    (raw / "observations.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        report.prefix_diagnosis(tmp_path, execution)


def test_software_prefix_capture_never_becomes_training_media(tmp_path):
    prefix_fixture(tmp_path)
    observations = report.rows(tmp_path)
    assert observations == []
    assert report.media(tmp_path, tmp_path / "publication", "source-family-02", "denoising", 28000,
                        "BLOCKED", observations) is None
    assert not (tmp_path / "publication/media").exists()

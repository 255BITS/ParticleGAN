"""Certified-observation binding for resumable media; no rendering/training."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "reports/forge/dualnorm-tier1/export_media.py"
SPEC = importlib.util.spec_from_file_location("bcap_optimizer_media", MODULE)
media = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media)


def test_ordinary_media_is_bound_to_the_original_observation_values(tmp_path):
    observations = [{"step": 1, "hq": .5}, {"step": 2, "hq": .9}]
    row = {"evidence": {"observations": observations}}
    receipt = {"observations_sha256": media.stable_hash(observations),
               "observation_count": 2, "source_inputs": {}}
    media.verify_observations(row, receipt, tmp_path)
    changed = deepcopy(row)
    changed["evidence"]["observations"][1]["hq"] = 1.
    with pytest.raises(ValueError, match="observations differ"):
        media.verify_observations(changed, receipt, tmp_path)


def test_clock_media_uses_the_retained_comparison_proof(tmp_path):
    proof = tmp_path / "comparisons.pt"
    proof.write_bytes(b"retained comparison proof")
    row = {"evidence": {"artifact_root": str(tmp_path), "artifact_manifest": {}, "comparisons": []}}
    receipt = {"observations_sha256": "reconstructed observations", "source_inputs": {str(proof): media.file_hash(proof)}}
    media.verify_observations(row, receipt, tmp_path)
    proof.write_bytes(b"changed proof")
    with pytest.raises(ValueError, match="source inputs differ"):
        media.verify_observations(row, receipt, tmp_path)


def test_cached_progress_cannot_change_the_original_result_identity(tmp_path, monkeypatch):
    certificate = {"result_hash": "original"}
    request = {"candidate": {"id": "candidate"}}
    monkeypatch.setattr(media, "certified", lambda *args: (None, request, {"task_results": []}, certificate))
    cached = {"result_hash": "another", "candidate": "candidate", "items": []}
    with pytest.raises(ValueError, match="original result certificate"):
        media.verify_completed(tmp_path, tmp_path / "queue", "attempt", cached, "renderer")

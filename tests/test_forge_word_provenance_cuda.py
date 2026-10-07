"""Complete word state certification without an ordinary qualification claim."""
import os
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_images import WordFixture
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import file_hash, read_json
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks
from experiments.forge.word_adapter import run_word


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="word provenance fixture requires CUDA")
def test_cuda_word_certifies_models_optimizers_policy_and_consumed_streams(tmp_path, monkeypatch):
    device = os.environ.get("PARTICLEGAN_TEST_CUDA_DEVICE", "cuda:0")
    assert torch.device(device).type == "cuda"
    task = load_tasks(ROOT)["five_word_joint_acquisition"]
    candidate = read_json(ROOT / "configs/forge/ideas/five-word-joint-ka2-v1.json")
    request = {"candidate": candidate, "candidate_revision": "software-provenance-fixture",
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    captured = []
    original = WordFixture.state_dict
    def snapshot(fixture):
        state = original(fixture)
        captured.append(state)
        return state
    monkeypatch.setattr(WordFixture, "state_dict", snapshot)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = run_word(request, task, tmp_path, device, execution_limit=2)
    finally:
        torch.set_num_threads(old_threads)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state(device))
    assert raw["scope"] == "integration_demo_only"
    assert grade_result(task, raw)["gate_status"] == "INCOMPLETE"
    certificate = raw["evidence"]["provenance_checkpoint"]
    root = Path(certificate["artifact_root"])
    verify_artifacts(root, certificate["artifact_manifest"])
    path = root / certificate["path"]
    assert file_hash(path) == certificate["sha256"] and path.stat().st_size == certificate["bytes"]
    state = torch.load(path, weights_only=True, map_location=device)
    assert state_digest(state) == certificate["state_sha256"]
    assert state_digest(state) == state_digest(torch.load(tmp_path / "state.pt", weights_only=True, map_location=device))
    assert state_digest(state["fixture"]) == state_digest(captured[-1])
    assert certificate["completed_steps"] == raw["cost"]["completed_steps"] == 2
    assert certificate["prerequisite_eligible"] is False and certificate["purpose"] == "provenance_only"
    assert certificate["optimizer_updates_added"] == certificate["sampling_draws_added"] == 0
    policy = state["fixture"]["api_state"]
    assert {"generator", "critic", "encoder", "prior"} <= policy["models"].keys()
    assert len(policy["optimizers"]) == 2
    assert policy["completed_steps"] == 2
    assert all(value.device.type == "cuda" for value in policy["models"]["generator"].values())
    streams = state["streams"]
    assert streams["manifest"] == raw["rng"]
    assert streams["states"].keys() == raw["rng"]["bindings"].keys()
    assert certificate["named_stream_keys"] == sorted(streams["states"])
    assert certificate["named_stream_state_sha256"] == {
        key: state_digest(value) for key, value in streams["states"].items()}
    data = [key for key, binding in raw["rng"]["bindings"].items() if binding["family"] == "data"]
    assert len(data) == 1 and torch.equal(state["fixture"]["data_generator"], streams["states"][data[0]])
    assert raw["evidence"]["guards"]["optimizer_updates"] == {
        "generator": 2, "encoder": 2, "prior": 2, "discriminator": 2}
    assert "checkpoint" not in raw["evidence"] and task["execution"]["produces_state"] is False

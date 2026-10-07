"""Separate task execution allowances from original recipe schedule horizons."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import pytest
import torch

from experiments.forge import adapters
from experiments.forge.api import FormulationContext
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import file_hash, read_json
from experiments.forge.state import require_consistent_rng, require_optimizer_steps, state_digest
from experiments.forge.views import validate_view


ROOT = Path(__file__).resolve().parents[1]


def request(task):
    return {"candidate": {"recipe_overrides": {"lr": .0001, "input_noise_std": 0.,
            "output_noise_std": 0.}}, "candidate_revision": "software-budget-fixture",
            "protocol": {"seed": 0}, "tasks": {task["id"]: task}}


@pytest.mark.parametrize("name,adapter,module", [
    ("ring16_acquisition", adapters._vector, "experiments.forge.vectorprofiles.build_vector_models"),
    ("img_bars4", adapters._image, "experiments.forge.imageprofiles.build_image_models"),
])
def test_metadata_passes_task_budget_1600_without_changing_schedule_400(tmp_path, monkeypatch, name, adapter, module):
    task = read_json(ROOT / "configs/forge/tasks" / f"{name}.json")
    task["execution"].update(steps=1600, original_schedule_horizon=400)
    task["execution"]["host_definition"]["steps"] = 1600
    initial = deepcopy(task)
    monkeypatch.setattr(module, lambda *args: (object(), object()))
    class ConstructionBoundary(Exception):
        pass
    def capture(context, generator, discriminator, *, max_steps=None):
        assert context.recipe.total_steps == 400
        assert max_steps == 1600
        raise ConstructionBoundary
    monkeypatch.setattr(FormulationContext, "build_trainer", capture)
    with pytest.raises(ConstructionBoundary):
        adapter(request(task), task, tmp_path, "cpu")
    assert task == initial
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="neural execution-budget fixtures require CUDA")
@pytest.mark.parametrize("name", ["vector_two_broad", "img_bars4"])
@pytest.mark.parametrize("produces_state", [False, True])
def test_cuda_adapter_finishes_four_updates_and_certifies_all_streams(tmp_path, name, produces_state):
    device = os.environ.get("PARTICLEGAN_TEST_CUDA_DEVICE", "cuda:0")
    assert torch.device(device).type == "cuda"
    task = read_json(ROOT / "configs/forge/tasks" / f"{name}.json")
    task["execution"].update(steps=4, original_schedule_horizon=2, produces_state=produces_state)
    task["execution"]["host_definition"].update(steps=4, particles=8)
    if name == "vector_two_broad":
        task["execution"]["host_definition"].update(hidden=4, layers=1, batch=4, fourier=0)
    else:
        task["execution"]["host_definition"].update(width=2, batch_size=4)
    task["evaluation"]["observations"] = 2
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = adapters.run_task(request(task), {"task_id": task["id"]}, tmp_path, device)
    finally:
        torch.set_num_threads(old_threads)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state(device))
    assert raw["recipe"]["total_steps"] == 2
    assert raw["cost"]["completed_steps"] == 4
    assert raw["evidence"]["guards"]["optimizer_updates"] == {
        "generator": 4, "discriminator": 4, "prior": 4}
    assert raw["evidence"]["guards"]["all_finite"] is True
    assert raw["evidence"]["guards"]["unintended_rng_deviations"] == 0
    checkpoint = raw["evidence"]["provenance_checkpoint"]
    assert checkpoint["schema_version"] == 1
    assert checkpoint["purpose"] == "provenance_only"
    assert checkpoint["prerequisite_eligible"] is False
    assert checkpoint["completed_steps"] == 4
    assert checkpoint["optimizer_updates_added"] == checkpoint["sampling_draws_added"] == 0
    certified_root = Path(checkpoint["artifact_root"])
    assert certified_root == tmp_path / "provenance"
    verify_artifacts(certified_root, checkpoint["artifact_manifest"])
    path = certified_root / checkpoint["path"]
    assert checkpoint["artifact_manifest"]["files"] == {
        "provenance-state.pt": {"size": checkpoint["bytes"], "sha256": checkpoint["sha256"]}}
    assert file_hash(path) == checkpoint["sha256"]
    state = torch.load(path, weights_only=True, map_location=device)
    assert state_digest(state) == checkpoint["state_sha256"]
    assert state["trainer"]["max_steps"] == 4
    assert state["trainer"]["recipe"]["total_steps"] == 2
    assert state["trainer"]["completed_steps"] == 4
    assert all(value.device.type == "cuda" for value in state["trainer"]["models"]["G"].values())
    assert {"G", "D", "prior"} <= state["trainer"]["models"].keys()
    require_optimizer_steps(state, 4)
    require_consistent_rng(state)
    streams = state["streams"]["states"]
    bindings = raw["rng"]["bindings"]
    assert streams.keys() == state["streams"]["manifest"]["bindings"].keys() == bindings.keys()
    assert sorted(streams) == checkpoint["named_stream_keys"]
    assert {key: state_digest(value) for key, value in streams.items()} == checkpoint["named_stream_state_sha256"]
    # Data and latent draws survive as consumed states. Image evaluation
    # enumerates finite centers, so its bound eval generator is unconsumed;
    # vector evaluation consumes its separately bound sample generator.
    for family in ("data", "prior", *(("eval",) if name == "vector_two_broad" else ())):
        assert any(binding["family"] == family and
            hashlib.sha256(streams[key].cpu().numpy().tobytes()).hexdigest() != binding["initial_state_sha256"]
            for key, binding in bindings.items())
    assert (tmp_path / "state.pt").exists() is produces_state
    if produces_state:
        assert state_digest(torch.load(tmp_path / "state.pt", weights_only=True, map_location=device)) == checkpoint["state_sha256"]
    else:
        assert "checkpoint" not in raw["evidence"]
        _assert_no_continuation_eligibility(task)
    assert [row["step"] for row in raw["evidence"]["observations"]] == [2, 4]
    # The certificate remains strict after the receipt is written: extra,
    # absent and changed payloads cannot silently become provenance.
    extra = certified_root / "unexpected.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="file set changed"):
        verify_artifacts(certified_root, checkpoint["artifact_manifest"])
    extra.unlink()
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="empty|missing"):
        verify_artifacts(certified_root, checkpoint["artifact_manifest"])
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_artifacts(certified_root, checkpoint["artifact_manifest"])
    path.write_bytes(original)
    verify_artifacts(certified_root, checkpoint["artifact_manifest"])


def _assert_no_continuation_eligibility(task):
    # Eligibility uses a complete declared card, not the intentionally reduced
    # software fixture's off-contract observation count.
    parent = read_json(ROOT / "configs/forge/tasks" / f"{task['id']}.json")
    assert task["execution"]["produces_state"] is parent["execution"]["produces_state"] is False
    child = deepcopy(parent)
    child["id"] = "continuation-fixture"
    child["execution"]["continuation_of"] = parent["id"]
    child["dependencies"] = [{"task": parent["id"], "kind": "checkpoint"}]
    view = {"schema_version": 1, "id": "provenance-fixture", "revision": 1,
        "goal": "provenance eligibility", "assignments": [
            {"task": parent["id"], "qualification_tier": 1, "importance": "required", "order": 0},
            {"task": child["id"], "qualification_tier": 2, "importance": "required", "order": 1}]}
    with pytest.raises(ValueError, match="checkpoint dependency does not produce state"):
        validate_view(view, {parent["id"]: parent, child["id"]: child})


def test_metadata_provenance_does_not_grant_checkpoint_dependency():
    task = read_json(ROOT / "configs/forge/tasks/vector_two_broad.json")
    assert task["execution"]["produces_state"] is False
    _assert_no_continuation_eligibility(task)

"""CUDA execution contracts; shortened fixtures confer no qualification."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch

from experiments.forge.behavior_adapters import BehaviorComponents, HOSTS, run_behavior
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import file_hash
from experiments.forge.state import state_digest
from experiments.forge.views import load_tasks


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("formulation", ["k3p", "ka2", "dualnorm"])
def test_behavior_cuda_models_streams_and_caller_rng(tmp_path, monkeypatch, host, formulation):
    device = os.environ.get("PARTICLEGAN_TEST_CUDA_DEVICE", "cuda:0")
    assert torch.device(device).type == "cuda"
    task = deepcopy(load_tasks(ROOT)[host])
    task["execution"]["steps"] = 2
    captured = []
    original = BehaviorComponents.bind

    def bind(self, **kwargs):
        result = original(self, **kwargs)
        captured.append(self)
        assert self.context.device.type == "cuda"
        assert all(p.device.type == "cuda" for ps in self.role_parameters.values() for p in ps)
        assert all(b.device.type == "cuda" for m in self.models.values() for b in m.buffers())
        return result

    monkeypatch.setattr(BehaviorComponents, "bind", bind)
    caller_cpu = torch.get_rng_state().clone()
    caller_cuda = torch.cuda.get_rng_state_all()
    request = dict(protocol=dict(seed=0), candidate=dict(recipe_overrides={
        "input_noise_std": .01, "output_noise_std": .01,
    }))
    if formulation == "dualnorm":
        request["candidate"]["recipe_preset"] = "bcap"
        request["candidate"]["recipe_overrides"].update(reg_arm="b_cap", optimizer_family="dualnorm", optimizer_momentum=0.)
    else:
        request["candidate"]["recipe_overrides"]["critic_formulation"] = formulation
    result = run_behavior(request, task, tmp_path / host, device=device)
    assert len(captured) == 1
    assert result["device"] == device
    assert result["evidence"]["guards"]["all_finite"]
    assert result["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert all(n == 2 for n in result["evidence"]["guards"]["optimizer_updates"].values())
    # Public deterministic initialization deliberately derives portable values
    # from CPU streams; all training/sampling/measurement streams use CUDA.
    assert all(b["device"].startswith("cuda:") for b in result["applied"]["rng"]["bindings"].values()
               if b["family"] != "init")
    assert torch.equal(caller_cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(caller_cuda, torch.cuda.get_rng_state_all()))
    saved = torch.load(tmp_path / host / "component-state.pt", weights_only=True)
    assert saved["streams"]["manifest"] == result["applied"]["rng"]
    assert saved["models"].keys() == captured[0].models.keys()
    certificate = result["evidence"]["provenance_checkpoint"]
    root = Path(certificate["artifact_root"])
    verify_artifacts(root, certificate["artifact_manifest"])
    assert certificate["schema_version"] == 1 and certificate["purpose"] == "provenance_only"
    assert certificate["prerequisite_eligible"] is False
    assert certificate["completed_steps"] == 2
    assert certificate["optimizer_updates_added"] == certificate["sampling_draws_added"] == 0
    assert file_hash(root / certificate["path"]) == certificate["sha256"]
    assert (root / certificate["path"]).stat().st_size == certificate["bytes"]
    certified = torch.load(root / certificate["path"], weights_only=True)
    assert state_digest(certified) == state_digest(saved) == certificate["state_sha256"]
    assert sorted(saved["streams"]["states"]) == certificate["named_stream_keys"]
    assert {key: state_digest(value) for key, value in saved["streams"]["states"].items()} == certificate["named_stream_state_sha256"]
    assert saved["streams"]["states"].keys() == result["applied"]["rng"]["bindings"].keys()
    for key, value in captured[0].context.streams.state_dict()["states"].items():
        assert torch.equal(value, saved["streams"]["states"][key])
    assert saved["role_parameters"].keys() == captured[0].role_parameters.keys()
    for role, parameters in captured[0].role_parameters.items():
        rows = saved["role_parameters"][role]
        assert [row["index"] for row in rows] == list(range(len(parameters)))
        assert len({row["name"] for row in rows}) == len(parameters)
        for row, parameter in zip(rows, parameters):
            assert torch.equal(row["value"], parameter.detach())
            assert row["requires_grad"] is parameter.requires_grad
    if host == "two_pole":
        assert "generator" not in saved["models"]
        coordinates = saved["role_parameters"]["prior"][0]
        assert coordinates["name"] == "direct_particles.0"
        assert coordinates["representation"] == "direct_sample_coordinates"
        assert coordinates["value"].count_nonzero() > 0
    assert "checkpoint" not in result["evidence"]
    assert task["execution"]["produces_state"] is False


def test_cpu_behavior_resource_lanes_are_removed():
    tasks = load_tasks(ROOT)
    for host in HOSTS:
        assert tasks[host]["execution"]["device"] == "cuda"
        assert tasks[host]["resources"]["gpus"] == 1
        assert tasks[host]["resources"]["allow_cpu"] is False

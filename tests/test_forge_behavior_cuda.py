"""CUDA execution contracts; shortened fixtures confer no qualification."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.forge.behavior_adapters import BehaviorComponents, HOSTS, run_behavior
from experiments.forge.views import load_tasks


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("formulation", ["k3p", "ka2", "dualnorm"])
def test_behavior_cuda_models_streams_and_caller_rng(tmp_path, monkeypatch, host, formulation):
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
    result = run_behavior(request, task, tmp_path / host, device="cuda:0")
    assert len(captured) == 1
    assert result["device"] == "cuda:0"
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


def test_cpu_behavior_resource_lanes_are_removed():
    tasks = load_tasks(ROOT)
    for host in HOSTS:
        assert tasks[host]["execution"]["device"] == "cuda"
        assert tasks[host]["resources"]["gpus"] == 1
        assert tasks[host]["resources"]["allow_cpu"] is False

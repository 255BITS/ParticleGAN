"""CUDA execution/checkpoint contract; no quality qualification."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from benchmarks.toy_audit.reproducibility import construction_rng, reproducible_execution
from experiments.forge.api import CapabilityError, FormulationContext
from experiments.forge.state import state_digest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def build(serial):
    context = FormulationContext(recipe_preset="bcap", seed=0, device="cuda:0",
        extensions={"serial_backward": serial},
        recipe_overrides={"num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 2,
                          "optimizer_family": "dualnorm", "optimizer_momentum": 0.})
    with construction_rng(0, "cuda:0"):
        g = nn.Sequential(nn.Linear(2, 6), nn.LeakyReLU(.2), nn.Linear(6, 2)).cuda()
        d = nn.Sequential(nn.Linear(2, 6), nn.LeakyReLU(.2), nn.Linear(6, 1)).cuda()
    return context, context.build_trainer(g, d)


@reproducible_execution
def check_checkpoint_and_execution(monkeypatch, *, device):
    original, a = build(True)
    restored, b = build(True)
    ordinary, c = build(False)
    initial = deepcopy(a.state_dict())
    assert initial.pop("serial_backward") is True
    assert state_digest(initial) == state_digest(c.state_dict())
    modes = []
    grad = torch.autograd.grad

    def observed_grad(*args, **kwargs):
        modes.append(torch.autograd.is_multithreading_enabled())
        return grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", observed_grad)
    hooks = [p.register_hook(lambda value: modes.append(torch.autograd.is_multithreading_enabled()))
             for p in a.D.parameters()]
    real = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2) / 8
    assert torch.autograd.is_multithreading_enabled()
    a.step(real, collect_stats=False)
    assert modes and not any(modes)
    assert torch.autograd.is_multithreading_enabled()
    for hook in hooks:
        hook.remove()
    checkpoint = deepcopy(original.state_dict())
    assert checkpoint["trainer"]["serial_backward"] is True
    restored.load_state_dict(checkpoint)
    with pytest.raises(ValueError, match="serial_backward"):
        c.load_state_dict(checkpoint["trainer"])
    a.step(real, collect_stats=False)
    b.step(real, collect_stats=False)
    assert state_digest(original.state_dict()) == state_digest(restored.state_dict())
    assert all(p.device.type == "cuda" for model in (a.G, a.D, a.prior) for p in model.parameters())


def test_serialized_cuda_derivatives_and_exact_checkpoint_continuation(monkeypatch):
    check_checkpoint_and_execution(monkeypatch, device="cuda:0")


def test_serial_extension_rejects_nonboolean_before_construction():
    with pytest.raises(CapabilityError, match="requires bool"):
        FormulationContext(device="cuda:0", extensions={"serial_backward": 1})

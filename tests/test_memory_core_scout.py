from dataclasses import asdict

import numpy as np
import pytest
import torch

from experiments.autonomous_memory import frozen
from experiments.memory_path import circles
from experiments import memory_scout as base
from experiments import memory_core_scout as core


def test_strict_context_and_read_only_scoring():
    g, d, prior, _ = core.build(core.Config(), "cpu")
    real = torch.randn(4, 64, 2)
    positions = torch.tensor([0, 1, 8, 32])
    memory = core.selected_context(d.writer, real, positions)
    altered = real.clone()
    for row, position in enumerate(positions):
        altered[row, position:] += 100
        expected = core.context(d.writer, real[row:row+1, :position])
        torch.testing.assert_close(memory[row:row+1], expected)
    torch.testing.assert_close(memory, core.selected_context(d.writer, altered, positions))
    snapshot = memory.detach().clone()
    d.score_candidate(torch.randn(4, 2), memory)
    torch.testing.assert_close(memory, snapshot)
    assert memory[0].count_nonzero() == 0


def test_feedback_matches_cold_and_keeps_gradients_to_early_points():
    g, d, prior, _ = core.build(core.Config(), "cpu")
    z = prior(torch.arange(4))
    expected, _ = base.rollout(g, d.writer, z, 8)
    actual, _ = core.continuation(g, d.writer, z, torch.empty(4, 0, 2), 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    points = []
    def retain(module, args, result):
        result[0].retain_grad()
        points.append(result[0])
    hook = g.register_forward_hook(retain)
    prefix = torch.randn(4, 8, 2)
    path, _ = core.continuation(g, d.writer, z, prefix, 8)
    path[:, -1].square().sum().backward()
    hook.remove()
    assert points[0].grad.abs().sum() > 0
    assert all(p.grad is None for p in d.writer.parameters())
    assert prior.z.grad.abs().sum() > 0


@pytest.mark.parametrize("weights", [(1, 0, 0), (0, 0, 1), (1, 1, 1)])
def test_exact_bcap_and_parameter_ownership(weights):
    cfg = core.Config(handoff_weight=weights[0], cold_weight=weights[1], warm_weight=weights[2])
    g, d, prior, recipe = core.build(cfg, "cpu")
    z = prior(torch.arange(4))
    real = torch.randn(4, 64, 2)
    positions = torch.tensor([0, 3, 8, 32])
    with torch.no_grad():
        items = core.objectives(cfg, g, d, z, real, positions, 8)
    penalty = recipe.make_gradient_penalty()
    sum(view(target).mean()+penalty(view, target, fake, step=1)
        for _, _, view, target, fake in items).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.writer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    assert all(p.grad is None for p in g.parameters())
    d.zero_grad(set_to_none=True)
    with frozen(d):
        items = core.objectives(cfg, g, d, z, real, positions, 8)
        sum(view(fake).mean() for _, _, view, _, fake in items).backward()
    assert all(p.grad is None for p in d.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.parameters())


def test_reference_fidelity_rejects_wrong_orbit_and_direction():
    _, clean = circles(4, 1056, torch.Generator().manual_seed(4), "cpu", noise=0)
    clean = clean.numpy()
    future = clean[:, 32:]
    perfect = core.fidelity(future, clean, 32)
    assert perfect["reference_orbit_fraction"] == 1.
    assert perfect["relative_radial_rmse"] < 1e-5
    assert perfect["startup_error_relative"] == 0.
    assert core.fidelity(future+2, clean, 32)["reference_orbit_fraction"] == 0.
    assert core.fidelity(future[:, ::-1].copy(), clean, 32)["reference_orbit_fraction"] == 0.


def test_resume_and_cold_control_equivalence(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    def evaluate(*args):
        return {"generated_256": {"circle_like_fraction": 0.}}, {}
    monkeypatch.setattr(core, "evaluate", evaluate)
    monkeypatch.setattr(base, "evaluate", evaluate)
    def run(name, steps, weights=(1., 1., 1.), resume=None, original=False):
        out = tmp_path/name
        out.mkdir()
        args = dict(name=name, writer="gru", steps=steps, schedule_steps=10,
                    batch_size=4, eval_batch=4, eval_steps=256, resume=resume)
        if original:
            base.train(base.Config(**args), out, "cpu", lambda **kw: None)
        else:
            cfg = core.Config(**args, handoff_weight=weights[0], cold_weight=weights[1], warm_weight=weights[2])
            core.train(cfg, out, "cpu", lambda **kw: None)
        return torch.load(out/"model.pt", weights_only=False)
    full = run("full", 2)
    run("first", 1)
    resumed = run("resumed", 2, resume=str(tmp_path/"first/model.pt"))
    cold = run("cold", 2, (0., 1., 0.))
    original = run("original", 2, original=True)
    for left, right in ((full, resumed), (original, cold)):
        for key in ("generator", "critic", "prior"):
            for name, value in left[key].items():
                torch.testing.assert_close(value, right[key][name], rtol=0, atol=0)

"""Meaningful contract tests for the optional research LR feedback policy."""
import math

import pytest
import torch

from benchmarks.learned_lr.controller import FEATURES, OptimizerLRAdapter


def make_optimizer():
    x = torch.nn.Parameter(torch.tensor([1., -2.]))
    y = torch.nn.Parameter(torch.tensor([.5]))
    opt = torch.optim.Adam([{"params": [x], "lr": .01}, {"params": [y], "lr": .03}], betas=(0., .99))
    return x, y, opt


def backprop(x, y, opt):
    opt.zero_grad()
    (x.square().sum() + y.square().sum()).backward()


def test_zero_policy_is_exact_constant_adam_and_preserves_rng():
    x, y, opt = make_optimizer()
    rx, ry, reference = make_optimizer()
    adapter = OptimizerLRAdapter({"weights": [[0.] * 6] * 2}, 8, interval=1)
    for step in range(8):
        backprop(x, y, opt)
        before = torch.get_rng_state().clone()
        adapter.step(opt, step, role="g")
        assert torch.equal(before, torch.get_rng_state())
        opt.step()
        backprop(rx, ry, reference)
        reference.step()
    assert torch.equal(x, rx)
    assert torch.equal(y, ry)
    assert [group["lr"] for group in opt.param_groups] == [.01, .03]


def test_bounds_group_ratios_no_compounding_and_time_only_ablation():
    x, y, opt = make_optimizer()
    weights = [[50., 0., 100., 100., 100., 100.]] * 2
    adapter = OptimizerLRAdapter({"weights": weights, "features": list(FEATURES)}, 4,
                                ablation="time_only", interval=1)
    for step in range(4):
        backprop(x, y, opt)
        # Simulate a host schedule. The controller must restore initial ratios.
        if step:
            for group in opt.param_groups:
                group["lr"] *= .01
        value = adapter.step(opt, step, role="g")
        assert .05 <= value <= 2.
        assert opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"] == pytest.approx(3.)
        assert opt.param_groups[0]["lr"] == pytest.approx(.01 * value)
        opt.step()
    assert value == pytest.approx(math.exp(math.log(2.) * (1 - .5 ** 4)))
    assert all(row["features"][2:] == [0.] * 4 for row in adapter.trace)


def test_feedback_changes_action_and_rejects_nonfinite_inputs():
    x, y, opt = make_optimizer()
    weights = [[0., 0., .5, 0., 0., 0.]] * 2
    adapter = OptimizerLRAdapter({"weights": weights}, 4, interval=1)
    backprop(x, y, opt)
    assert adapter.step(opt, 0, role="d") == 1.
    x.grad.mul_(.01)
    y.grad.mul_(.01)
    assert adapter.step(opt, 1, role="d") < 1.
    x.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        adapter.step(opt, 2, role="d")
    with pytest.raises(ValueError, match="weights"):
        OptimizerLRAdapter({"weights": [[float("nan")] * 6] * 2}, 4)

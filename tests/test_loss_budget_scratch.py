import copy
import math
import pytest
import torch
from reports.toy100.loss_budget_scratch import LossBudget


def test_budget_matches_linear_descent_and_preserves_full_moments():
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    z = torch.nn.Parameter(torch.tensor([.5], dtype=torch.float64))
    opt = torch.optim.Adam([dict(params=[p], _comparison_prior=False),
                            dict(params=[z], _comparison_prior=True)], lr=.1, betas=(0., .9))
    p.grad = torch.tensor([.3, -.4], dtype=torch.float64)
    z.grad = torch.tensor([.5], dtype=torch.float64)
    original = p.detach().clone()
    ctrl = LossBudget()
    ctrl.pending = dict(loss=math.log(2)+.035, budget=.035)
    rng = torch.get_rng_state().clone()
    ctrl.step(opt, torch.optim.Adam.step)
    assert float(-torch.dot(p.grad, p.detach()-original)) == pytest.approx(.035)
    assert float(z.detach()) == pytest.approx(.4)
    assert torch.equal(opt.state[p]["exp_avg"], p.grad)
    assert torch.allclose(opt.state[p]["exp_avg_sq"], .1*p.grad.square())
    assert int(opt.state[p]["step"]) == 1
    assert torch.equal(torch.get_rng_state(), rng)


def test_joint_rest_then_signal_wakes_without_moment_reset():
    p = torch.nn.Parameter(torch.tensor([1.]))
    z = torch.nn.Parameter(torch.tensor([.5]))
    opt = torch.optim.Adam([dict(params=[p], _comparison_prior=False),
                            dict(params=[z], _comparison_prior=True)], lr=.1, betas=(0., .9))
    p.grad, z.grad = torch.ones_like(p), torch.ones_like(z)
    ctrl = LossBudget("joint")
    ctrl.observe(torch.ones(4), torch.zeros(4))
    before = [p.detach().clone(), z.detach().clone()]
    ctrl.step(opt, torch.optim.Adam.step)
    assert torch.equal(p, before[0]) and torch.equal(z, before[1])
    ctrl.observe(torch.zeros(4), torch.ones(4))
    ctrl.step(opt, torch.optim.Adam.step)
    assert p < before[0] and z < before[1]
    assert int(opt.state[p]["step"]) == 2


def test_observation_control_is_exact_adam():
    p = torch.nn.Parameter(torch.tensor([1.]))
    z = torch.nn.Parameter(torch.tensor([.5]))
    opt = torch.optim.Adam([dict(params=[p], _comparison_prior=False),
                            dict(params=[z], _comparison_prior=True)], lr=.1, betas=(0., .9))
    p.grad, z.grad = torch.ones_like(p), torch.ones_like(z)
    ref = copy.deepcopy(opt)
    for group in ref.param_groups:
        for value in group['params']:
            value.grad = torch.ones_like(value)
    ctrl = LossBudget(observe_only=True)
    ctrl.observe(torch.ones(4), torch.zeros(4))
    ctrl.step(opt, torch.optim.Adam.step)
    ref.step()
    for left, right in zip(opt.param_groups, ref.param_groups):
        assert torch.equal(left['params'][0], right['params'][0])

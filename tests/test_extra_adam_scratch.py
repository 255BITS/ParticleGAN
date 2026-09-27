"""ExtraAdam must evaluate a joint game and correct from original weights.

The frozen scratch's host-source transform (reports/toy100/extra_adam_scratch.py)
targeted host loops with the removed ``schedule_optimizer`` hook; its record
stays in reports/, and only the optimizer-rule tests remain live.
"""

from copy import deepcopy

import pytest
import torch

from reports.toy100.extra_adam_scratch import ExtraAdamRecorder


@pytest.mark.parametrize("method,evaluations", [("extra_adam", 2), ("sim_adam", 1)])
def test_bilinear_joint_gradients_and_exact_moment_formula(method, evaluations):
    d = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    g = torch.nn.Parameter(torch.tensor(-.4, dtype=torch.float64))
    opt_d = torch.optim.Adam([d], lr=.03, betas=(.2, .8), eps=1e-7)
    opt_g = torch.optim.Adam([g], lr=.05, betas=(.2, .8), eps=1e-7)
    recorder = ExtraAdamRecorder(method)
    expected = torch.tensor([.7, -.4], dtype=torch.float64)
    rates = torch.tensor([.03, .05], dtype=torch.float64)
    first = torch.zeros(2, dtype=torch.float64)
    second = torch.zeros(2, dtype=torch.float64)
    moments = 0
    for outer in range(5):
        base = expected.clone()
        for phase in recorder.phases(outer, opt_d, opt_g):
            gradient = torch.stack((-expected[1], expected[0]))
            first = .2 * first + .8 * gradient
            second = .8 * second + .2 * gradient.square()
            moments += 1
            direction = (first / (1 - .2 ** moments)) / ((second / (1 - .8 ** moments)).sqrt() + 1e-7)
            expected = base - rates * direction
            old = torch.stack((d.detach(), g.detach()))
            opt_d.zero_grad()
            (-d * g).backward()
            recorder.step(opt_d, torch.optim.Adam.step)
            assert torch.equal(torch.stack((d.detach(), g.detach())), old)
            opt_g.zero_grad()
            (d * g).backward()
            recorder.step(opt_g, torch.optim.Adam.step)
            assert torch.allclose(torch.stack((d.detach(), g.detach())), expected, atol=2e-15, rtol=0)
    receipt = recorder.receipt()
    assert receipt["outer_steps"] == 5
    assert receipt["joint_points_verified"] == 5 * evaluations
    assert receipt["base_restores_verified"] == (5 if evaluations == 2 else 0)
    for row in receipt["optimizers"]:
        assert row["calls"] == 5 * evaluations
        assert row["groups"][0]["moment_steps"] == [5 * evaluations]
        assert row["rates"] == [[row["groups"][0]["lr"]]] * (5 * evaluations)


def test_unexpected_parameter_mutation_is_rejected():
    d = torch.nn.Parameter(torch.tensor(.7))
    g = torch.nn.Parameter(torch.tensor(-.4))
    opt_d, opt_g = torch.optim.Adam([d]), torch.optim.Adam([g])
    recorder = ExtraAdamRecorder("extra_adam")
    phases = recorder.phases(0, opt_d, opt_g)
    next(phases)
    d.grad = torch.ones_like(d)
    recorder.step(opt_d, torch.optim.Adam.step)
    with torch.no_grad():
        d.add_(.1)
    g.grad = torch.ones_like(g)
    with pytest.raises(RuntimeError, match="before both game gradients"):
        recorder.step(opt_g, torch.optim.Adam.step)


def test_optimizer_state_preserves_both_moment_evaluations():
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    opponent = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_d, opt_g = torch.optim.Adam([opponent]), torch.optim.Adam([parameter])
    recorder = ExtraAdamRecorder("extra_adam")
    for phase in recorder.phases(0, opt_d, opt_g):
        opponent.grad = torch.tensor([float(phase + 1)], dtype=torch.float64)
        recorder.step(opt_d, torch.optim.Adam.step)
        parameter.grad = torch.tensor([float(3 + phase)], dtype=torch.float64)
        recorder.step(opt_g, torch.optim.Adam.step)
    assert opt_g.state[parameter]["step"] == 2
    assert opt_g.state[parameter]["exp_avg"].item() == pytest.approx(.9 * .3 + .1 * 4)
    saved = deepcopy(opt_g.state_dict())
    restored = torch.optim.Adam([torch.nn.Parameter(parameter.detach().clone())])
    restored.load_state_dict(saved)
    assert restored.state_dict()["state"][0]["step"] == 2

"""Independent closed-form checks for the cross-player game response."""

import pytest
import torch

from reports.toy100.cross_competitive_scratch import CrossCompetitiveRecorder


def test_cross_only_solve_omits_own_quadratic_jacobian():
    # D owns y, G owns x. F_D=y-x and F_G=x+y: each player has an own
    # quadratic term and the players have opposing bilinear cross terms.
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=.1, betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=.1, betas=(0., .9), eps=1e-12)
    recorder = CrossCompetitiveRecorder(krylov_dim=4, linear_tolerance=1e-8)

    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = y.detach().clone() - x.detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = x.detach().clone() + y.detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)

    # First-gradient Adam gives P_D=.1/|F_D|=.1 and P_G=.1/|F_G|=1/30.
    # Cross-only implicit response solves
    # [1, -.1; 1/30, 1] [dy, dx] = [-.1, -.1].
    delta = torch.linalg.solve(
        torch.tensor([[1., -.1], [1 / 30, 1.]], dtype=torch.float64),
        torch.tensor([-.1, -.1], dtype=torch.float64),
    )
    assert y.item() == pytest.approx(2 + delta[0].item(), abs=2e-7)
    assert x.item() == pytest.approx(1 + delta[1].item(), abs=2e-7)
    assert recorder.solves[-1]["accepted"]
    residual = recorder.joint_residuals[-1]
    assert residual["cross_relative_residual"] < 1e-6
    assert residual["full_joint_relative_residual"] > .01
    assert len(recorder.cross_products) >= 1
    assert any(row["own_field_difference_norm"] > 0
               for row in recorder.cross_products)
    for row in recorder.receipt()["optimizers"]:
        assert row["calls"] == 1 + len(recorder.queries)
        assert row["groups"][0]["moment_steps"] == [1]


def test_cross_only_zero_field_stays_still_without_queries():
    x = torch.nn.Parameter(torch.tensor([0.]))
    y = torch.nn.Parameter(torch.tensor([0.]))
    opt_g = torch.optim.Adam([x], lr=.1, betas=(0., .9))
    opt_d = torch.optim.Adam([y], lr=.1, betas=(0., .9))
    recorder = CrossCompetitiveRecorder()

    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = y.detach().clone() - x.detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = x.detach().clone() + y.detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)

    assert x.item() == y.item() == 0
    assert recorder.queries == recorder.cross_products == recorder.joint_residuals == []
    assert recorder.solves[0]["zero_field"]
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1

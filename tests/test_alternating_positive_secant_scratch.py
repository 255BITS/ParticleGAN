"""Independent secant arithmetic and alternating-update checks."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.alternating_positive_secant_scratch import (
    PositiveSecantRecorder, alternating_positive_secant, positive_secant_direction,
)


def test_positive_rank_one_damps_stiff_direction_and_contracts_adam_norm():
    s = torch.tensor([-1., -1.], dtype=torch.float64)
    y = torch.tensor([-1., -4.], dtype=torch.float64)
    metric = torch.ones(2, dtype=torch.float64)
    result, row = positive_secant_direction(s, y, metric, rho=4., fallback_bound=.25)
    assert row["rule"] == "positive_rank_one"
    assert row["curvature"] == 5.
    assert row["y_metric_y"] == 17.
    assert torch.allclose(result, torch.tensor([-17. / 22., -2. / 22.],
                                              dtype=torch.float64))
    assert row["metric_norm_ratio"] < 1.
    assert abs(result[1]) < abs(result[0])  # stiff coordinate damped more


def test_nonpositive_curvature_falls_back_to_declared_g_bound_and_zero_rests():
    metric = torch.ones(2, dtype=torch.float64)
    result, row = positive_secant_direction(
        torch.tensor([-1., -1.]), torch.tensor([1., 1.]), metric,
        rho=2., fallback_bound=.25)
    assert row["rule"] == "scalar_curvature_fallback"
    assert row["factor"] == .125
    assert torch.equal(result, torch.tensor([-.125, -.125], dtype=torch.float64))
    zero, row = positive_secant_direction(torch.zeros(2), torch.zeros(2), metric,
                                           rho=0., fallback_bound=.25)
    assert torch.equal(zero, torch.zeros(2, dtype=torch.float64))
    assert row["factor"] == 1.


def test_analytic_d_then_g_own_secant_with_one_adam_moment_per_player():
    # D descends -x+4y: first Adam step y=2->1, below D's rho=4/7 bound 2.
    # Then G descends y+3x: x=1->0, g0=4, g1=1, s=-1, y_sec=-3,
    # P=1/4. Resolvent gives s_final=-1+(.25*3*3)/(3+2.25)=-4/7.
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    recorder = PositiveSecantRecorder(curvature_bound=.25, d_curvature_bound=2.)
    for _ in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (-x + 4 * y).detach().clone()
        x.grad = (y + 3 * x).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        recorder.step(opt_g, torch.optim.Adam.step)
    assert y.item() == pytest.approx(1.)
    assert x.item() == pytest.approx(3. / 7.)
    assert recorder.records[0]["g"]["rule"] == "positive_rank_one"
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1
    assert recorder.rng_replay_verified == 2


def test_two_step_host_replays_rng_and_keeps_one_moment_update():
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    with alternating_positive_secant(start_step=0) as (recorder, _):
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2),
                                           noise_policy=policy, diagnostics=True)
    assert result["step"] == 2
    assert recorder.outer_steps == 2
    assert recorder.rng_replay_verified == 4
    assert all(int(pstate["step"]) == 2 for opt in recorder.optimizers
               for pstate in opt.state.values())
    assert all(row["gradient_calls"] == row["rate_observations"] == 6
               and row["moment_step_min"] == row["moment_step_max"] == 2
               for row in recorder.receipt()["optimizers"])

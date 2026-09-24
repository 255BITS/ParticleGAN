"""Analytic and host parity checks for alternating trapezoid correction."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.alternating_curvature_scratch import alternating_curvature
from reports.toy100.alternating_heun_scratch import AlternatingHeunRecorder, alternating_heun


def _host(context, steps=3):
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    with context as (recorder, _):
        result = mode_hold.train_mode_hold(
            mode_hold.ModeHoldRecipe(steps=steps), noise_policy=policy, diagnostics=True)
    return result, recorder, torch.get_rng_state().clone(), policy.input_stream.get_state().clone()


def test_zero_correction_matches_alternating_d_bound_exactly():
    old = _host(alternating_curvature(start_step=0, curvature_bound=1e9,
                                      bound_d=True, d_curvature_bound=2.0))
    new = _host(alternating_heun(start_step=0, heun_weight=0.0))
    assert old[0] == new[0]
    assert torch.equal(old[2], new[2]) and torch.equal(old[3], new[3])
    assert old[1].rng_replay_verified == new[1].rng_replay_verified == 6
    assert all(row["g"]["correction_norm_ratio"] == 0 for row in new[1].records)


def test_trapezoid_vector_step_and_one_moment_update():
    # D descends y-field -x+4y; G descends y+3x. At the first step D's
    # curvature ratio is 4/7, below its bound 2, so D moves 2 -> 1.
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    recorder = AlternatingHeunRecorder()
    for _ in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (-x + 4 * y).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = (y + 3 * x).detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)
    assert y.item() == pytest.approx(1.)
    # G base field 4, proposal x=0, replay field 1; P=1/4.
    # Heun correction = -(1/2)(1/4)(1-4) = +3/8.
    assert x.item() == pytest.approx(3. / 8.)
    assert recorder.records[0]["g"]["correction_norm_ratio"] == pytest.approx(3. / 8.)
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1
    assert recorder.rng_replay_verified == 2


def test_zero_field_keeps_parameters_still():
    x = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=1., betas=(0., .9))
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9))
    recorder = AlternatingHeunRecorder()
    for _ in recorder.phases(0, opt_d, opt_g, {}):
        x.grad = torch.zeros_like(x)
        y.grad = torch.zeros_like(y)
        recorder.step(opt_d, torch.optim.Adam.step)
        recorder.step(opt_g, torch.optim.Adam.step)
    assert x.item() == y.item() == 0
    assert recorder.records[0]["g"]["correction_norm_ratio"] == 0


def test_network_scope_keeps_learned_prior_ordinary():
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    z = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([{"params": [x]},
                              {"params": [z], "_comparison_prior": True}],
                             lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    recorder = AlternatingHeunRecorder(scope="network")
    for _ in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (-x + 4 * y).detach().clone()
        x.grad = (y + 3 * x).detach().clone()
        z.grad = (2 * z).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        recorder.step(opt_g, torch.optim.Adam.step)
    assert x.item() == pytest.approx(3. / 8.)
    assert z.item() == pytest.approx(0.)
    assert recorder.records[0]["g"]["group_norms"]["prior"]["correction"] == 0
    assert opt_g.state[x]["step"] == opt_g.state[z]["step"] == 1

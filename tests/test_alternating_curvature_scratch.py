"""Alternating adapter: exact plain-Adam parity when inactive, correct bound."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.alternating_curvature_scratch import AlternatingCurvatureRecorder, BothBoundRecorder, alternating_curvature


def _host(context=None, steps=3):
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    if context is None:
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps), noise_policy=policy,
                                           diagnostics=True)
        recorder = None
    else:
        with context as (recorder, _):
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps), noise_policy=policy,
                                               diagnostics=True)
    return result, torch.get_rng_state().clone(), policy.input_stream.get_state().clone(), recorder


def test_unbounded_alternating_adapter_matches_plain_alternating_adam_exactly():
    plain = _host()
    wrapped = _host(alternating_curvature(start_step=0, curvature_bound=1e9, advantage_gate=None))
    assert plain[0] == wrapped[0]
    assert all(torch.equal(a, b) for a, b in zip(plain[1:3], wrapped[1:3]))
    recorder = wrapped[3]
    assert recorder.rng_replay_verified == recorder.outer_steps == 3
    assert all(row["factor"] == 1. for row in recorder.records)


def test_open_gate_skips_replay_and_matches_plain_adam():
    plain = _host()
    wrapped = _host(alternating_curvature(start_step=0, curvature_bound=.25, advantage_gate=1e-9))
    recorder = wrapped[3]
    if all(row["gate_open"] for row in recorder.records):
        assert plain[0] == wrapped[0]
        assert recorder.rng_replay_verified == 0


def test_bound_uses_alternating_field_and_scales_only_g():
    # D=y, G=x. D steps first; G's field is evaluated at the new D.
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    a = 3.
    recorder = AlternatingCurvatureRecorder(curvature_bound=.25, advantage_gate=None)
    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (-x).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = (y + a * x).detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)
    y_new = 2. + 1.  # Adam beta1=0 first step moves by lr * sign(grad)
    g0 = y_new + a * 1.
    delta = -1. * g0 / abs(g0)
    p = 1. / abs(g0)
    rho = abs(a * delta) * p ** .5 / (abs(delta) / p ** .5)
    assert y.item() == pytest.approx(y_new)
    assert recorder.records[-1]["rho"] == pytest.approx(rho)
    assert x.item() == pytest.approx(1. + delta * min(1., .25 / rho))
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1


def test_both_bound_unbounded_matches_plain_alternating_adam_exactly():
    plain = _host()
    wrapped = _host(alternating_curvature(start_step=0, curvature_bound=1e9, bound_d=True))
    assert plain[0] == wrapped[0]
    assert all(torch.equal(a, b) for a, b in zip(plain[1:3], wrapped[1:3]))
    recorder = wrapped[3]
    assert recorder.rng_replay_verified == 2 * recorder.outer_steps
    assert all(r["d"]["factor"] == r["g"]["factor"] == 1. for r in recorder.records)


def test_both_bound_scales_d_then_g_responds_to_bounded_d():
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    a, b = 3., 4.
    recorder = BothBoundRecorder(curvature_bound=.25)
    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (-x + b * y).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = (y + a * x).detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)
    # D: grad 7, Adam step -1, P=1/7, rho_D=b*P=4/7 -> factor .4375.
    y_star = 2. - .25 / (4. / 7.)
    g0 = y_star + a
    rho_g = a / g0
    assert recorder.records[-1]["d"]["rho"] == pytest.approx(4. / 7.)
    assert y.item() == pytest.approx(y_star)
    assert recorder.records[-1]["g"]["rho"] == pytest.approx(rho_g)
    assert x.item() == pytest.approx(1. - min(1., .25 / rho_g))
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1


@pytest.mark.parametrize("a,b", [(3., 4.), (.3, 4.), (30., .8)])
def test_ratio_controller_scales_g_bound_by_clipped_reference_over_ratio(a, b):
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    recorder = BothBoundRecorder(curvature_bound=.25, d_curvature_bound=1e9, ratio_reference=2.)
    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (-x + b * y).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = (y + a * x).detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)
    row = recorder.records[-1]
    ratio = row["g"]["rho"] / row["d"]["rho"]
    expected_bound = .25 * min(1.5, max(1 / 1.5, 2. / ratio))
    assert row["smoothed_ratio"] == pytest.approx(ratio)
    assert row["g_bound"] == pytest.approx(expected_bound)
    assert row["g"]["factor"] == pytest.approx(min(1., expected_bound / row["g"]["rho"]))


def test_per_group_bound_scales_network_and_prior_groups_separately():
    # One G optimizer with two groups: "network" x1 (own curvature 3) and a
    # prior group x2 (own curvature .1). Joint rho mixes them; per-group does not.
    x1 = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    x2 = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([{"params": [x1]}, {"params": [x2], "_comparison_prior": True}],
                             lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    recorder = BothBoundRecorder(curvature_bound=.25, d_curvature_bound=1e9, per_group=True)
    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (-x1 - x2 + 2. * y).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x1.grad = (y + 3. * x1).detach().clone()
        x2.grad = (y + .1 * x2).detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)
    network, prior = recorder.records[-1]["g_groups"]
    y_new = 2. - 1.
    g1, g2 = y_new + 3., y_new + .1
    assert network["rho"] == pytest.approx(3. / g1) and prior["rho"] == pytest.approx(.1 / g2)
    assert x1.item() == pytest.approx(1. - min(1., .25 / network["rho"]))
    assert x2.item() == pytest.approx(1. - min(1., .25 / prior["rho"]))
    assert prior["factor"] == 1. and network["factor"] < 1.

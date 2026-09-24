"""Independent math and frozen-host checks for the implicit game adapter."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.implicit_extra_scratch import ImplicitExtraRecorder, implicit_extra


def test_bilinear_implicit_step_replays_field_and_advances_moments_once():
    # F(x, y)=(y, -x). The first Adam denominator is (|y|, |x|), so
    # P=diag(.05, .1) and the exact implicit equations have this solution.
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=.1, betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=.1, betas=(0., .9), eps=1e-12)
    # Tighten the linear solve here to compare with the exact two-dimensional
    # implicit solution; the production diagnostic uses a looser tolerance.
    recorder = ImplicitExtraRecorder(krylov_dim=4, linear_tolerance=1e-8)
    host_rng_before = torch.get_rng_state().clone()
    draws = []

    for phase in recorder.phases(0, opt_d, opt_g, {}):
        draws.append(torch.rand(3))
        y.grad = -x.detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = y.detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)

    assert x.item() == pytest.approx(.895522388, abs=2e-7)
    assert y.item() == pytest.approx(2.089552239, abs=2e-7)
    assert all(torch.equal(draws[0], draw) for draw in draws[1:])
    expected_after_one_batch = torch.get_rng_state().clone()
    torch.set_rng_state(host_rng_before)
    torch.rand(3)
    assert torch.equal(torch.get_rng_state(), expected_after_one_batch)
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1
    assert recorder.solves[-1]["accepted"]
    assert recorder.solves[-1]["nonlinear_relative_residual"] < 1e-5
    assert recorder.solves[-1]["linear_relative_residual"] <= recorder.linear_tolerance
    assert len(recorder.queries) >= 2
    for row in recorder.receipt()["optimizers"]:
        assert row["calls"] == 1 + len(recorder.queries)
        assert row["groups"][0]["moment_steps"] == [1]


def test_zero_game_field_stays_still():
    x = torch.nn.Parameter(torch.tensor([0.]))
    y = torch.nn.Parameter(torch.tensor([0.]))
    opt_g = torch.optim.Adam([x], lr=.1, betas=(0., .9))
    opt_d = torch.optim.Adam([y], lr=.1, betas=(0., .9))
    recorder = ImplicitExtraRecorder()

    for phase in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = -x.detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        x.grad = y.detach().clone()
        recorder.step(opt_g, torch.optim.Adam.step)

    assert x.item() == y.item() == 0
    assert recorder.queries == []
    assert recorder.solves[0]["zero_field"]
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1


def test_inactive_adapter_preserves_frozen_host_noise_and_rng():
    torch.set_num_threads(1)

    def run():
        policy = NoisePolicy(.029, .5, .1, 1200)
        result = mode_hold.train_mode_hold(
            mode_hold.ModeHoldRecipe(steps=2), noise_policy=policy,
            diagnostics=True,
        )
        return result, policy.receipt(), torch.get_rng_state().clone(), \
            policy.input_stream.get_state().clone()

    ordinary = run()
    with implicit_extra(start_step=1000) as (recorder, _):
        wrapped = run()
    assert ordinary[:2] == wrapped[:2]
    assert all(torch.equal(left, right) for left, right in
               zip(ordinary[2:], wrapped[2:]))
    assert recorder.outer_steps == 0


def test_active_host_reevaluation_consumes_one_training_batch():
    torch.set_num_threads(1)

    def run(context=None):
        policy = NoisePolicy(.029, .5, .1, 1200)
        if context is None:
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=1),
                                      noise_policy=policy)
            recorder = None
        else:
            with context as (recorder, _):
                mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=1),
                                          noise_policy=policy)
        return (policy.receipt(), torch.get_rng_state().clone(),
                policy.input_stream.get_state().clone(), recorder)

    ordinary = run()
    active = run(implicit_extra(start_step=0, krylov_dim=4))
    assert ordinary[0]["step_calls"] == active[0]["step_calls"] == 1
    assert all(torch.equal(left, right) for left, right in
               zip(ordinary[1:3], active[1:3]))
    recorder = active[3]
    assert recorder.outer_steps == 1
    assert recorder.rng_replay_verified == len(recorder.queries)
    assert recorder.solves[-1]["accepted"]
    for row in recorder.receipt()["optimizers"]:
        assert row["calls"] == 1 + len(recorder.queries)
        assert all(step == 1 for group in row["groups"]
                   for step in group["moment_steps"])

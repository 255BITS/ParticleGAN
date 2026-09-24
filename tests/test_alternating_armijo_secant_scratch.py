"""D accepted-point and G secant accounting for the declared combination."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.alternating_armijo_secant_scratch import (
    ArmijoSecantRecorder, alternating_armijo_secant,
)


def _game(*, loss_reader, max_retries=12):
    x = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_g = torch.optim.Adam([x], lr=1., betas=(0., .9), eps=1e-12)
    opt_d = torch.optim.Adam([y], lr=1., betas=(0., .9), eps=1e-12)
    recorder = ArmijoSecantRecorder(curvature_bound=.25, d_armijo=.1,
                                    max_retries=max_retries,
                                    reject_exhausted=True,
                                    loss_reader=lambda role: loss_reader(y))
    for _ in recorder.phases(0, opt_d, opt_g, {}):
        y.grad = (y - 1.).detach().clone()
        x.grad = (y + 3 * x).detach().clone()
        recorder.step(opt_d, torch.optim.Adam.step)
        recorder.step(opt_g, torch.optim.Adam.step)
    return x, y, opt_g, opt_d, recorder


def test_accepted_d_armijo_then_g_positive_secant_exact_quadratic():
    x, y, opt_g, opt_d, recorder = _game(loss_reader=lambda y: .5 * (float(y) - 1.) ** 2)
    assert y.item() == pytest.approx(1.)
    assert recorder.records[0]["d"]["alpha"] == 1.
    # At realized D=1: G g0=4, s=-1, g1=1, P=1/4; resolvent x=3/7.
    assert x.item() == pytest.approx(3. / 7.)
    assert recorder.records[0]["g"]["rule"] == "positive_rank_one"
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1
    assert recorder.rng_replay_verified == 2


def test_exhausted_d_rejection_uses_exact_zero_replay_before_g():
    x, y, opt_g, opt_d, recorder = _game(loss_reader=lambda y: 0., max_retries=0)
    assert y.item() == 2.
    assert recorder.records[0]["d"]["alpha"] == 0.
    assert recorder.records[0]["d"]["zero_step_identity_verified"] is True
    # At unchanged D=2: G g0=5, s=-1, g1=2, P=1/5, x=3/8.
    assert x.item() == pytest.approx(3. / 8.)
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 1
    assert recorder.rng_replay_verified == 3


def test_two_step_frozen_host_records_one_moment_and_rng_replay():
    torch.set_num_threads(1)
    with alternating_armijo_secant(curvature_bound=.25, d_armijo=.1,
                                    max_retries=12, reject_exhausted=True) as (recorder, _):
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2),
            noise_policy=NoisePolicy(.029, .5, .1, 1200), diagnostics=True)
    assert result["step"] == recorder.outer_steps == 2
    assert recorder.rng_replay_verified >= 4
    receipt = recorder.receipt()
    assert all(all(step == 2 for group in row["moment_steps"] for step in group)
               for row in receipt["optimizers"])
    assert receipt["positive_secant_updates"] + receipt["g_fallback_updates"] == 2

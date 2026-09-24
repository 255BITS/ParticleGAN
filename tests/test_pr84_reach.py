import pytest
import torch

from benchmarks.locked_shared.observation import notify_ring, set_ring_listener
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import (
    ReachRecorder, advantage_collapsed, pr84_reach_candidate, reach_width,
)


@pytest.mark.parametrize("sharpness", [.01, .1, .15, .3])
def test_reach_is_pr84_width_below_slope_utilisation_threshold(sharpness):
    assert reach_width(sharpness) == min(base.SMOOTH_WIDTH_CAP, .5 / sharpness)


def test_reach_peaks_at_the_b_cap_slope():
    assert reach_width(1.) == pytest.approx(.5)
    assert reach_width(.8) == pytest.approx(reach_width(1.25))
    assert reach_width(.6) < reach_width(.9) < reach_width(1.)
    assert reach_width(1., reach=1.) == pytest.approx(1.)


def test_stall_needs_saturated_d_and_collapsed_g_trust():
    with pr84_reach_candidate(task="mode_hold", ramp="stall") as (recorder, _source):
        assert not recorder._stalled(1.)
        recorder.records = [dict(g=dict(factor=.05))] * 60
        assert recorder._stalled(.6) and not recorder._stalled(.59)
        recorder.records = [dict(g=dict(factor=.2))] * 60
        assert not recorder._stalled(1.)


def test_advantage_drop_threshold_is_the_fixed_tenth():
    high = [1.0] * 50
    assert not advantage_collapsed(high + [0.9375] * 50)  # drop 0.0625
    assert advantage_collapsed(high + [0.875] * 50)  # drop 0.125
    assert advantage_collapsed(high + [0.90] * 50)  # drop 0.10
    assert not advantage_collapsed([0.2] * 100)
    assert not advantage_collapsed([1.0] * 99)
    assert advantage_collapsed.__code__.co_varnames[:1] == ("values",)


def test_ring_log_arms_once_on_full_hq_and_reject_stays_off_before_that():
    with pr84_reach_candidate(task="mode_hold", ramp="stall", adv_reject=True) as (recorder, _source):
        recorder.note_ring(100, 6, 1.)
        recorder.note_ring(200, 8, .89)
        assert not recorder.armed
        assert not recorder._adv_watching(object())
        recorder.note_ring(570, 8, .9)
        assert recorder.armed and recorder.arm_update == 570 and recorder.arm_phase == "cold_acquire"
        recorder.note_ring(700, 8, 1.)
        assert recorder.arm_update == 570
    assert set_ring_listener(None) is None


def test_critic_advantage_collapse_skips_g_adam_and_zeros_grads():
    torch.manual_seed(0)
    generator = torch.nn.Linear(3, 2, bias=False)
    critic = torch.nn.Linear(2, 1, bias=False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=.2, betas=(0., .99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=.2, betas=(0., .99))
    data = torch.randn(4, 3)
    loss = generator(data).sum()
    loss.backward()
    real_step = torch.optim.Adam.step
    real_step(opt_g)
    param = next(generator.parameters())
    step_before = int(opt_g.state[param]["step"])
    saved = [p.detach().clone() for p in generator.parameters()]
    d_saved = next(critic.parameters()).detach().clone()
    opt_g.zero_grad()
    generator(data).sum().backward()
    assert param.grad.abs().sum() > 0
    with pr84_reach_candidate(task="mode_hold", ramp="stall", adv_reject=True) as (recorder, _source):
        recorder.armed = True
        recorder.enabled = True
        recorder.passthrough = False
        recorder.phase = 1
        recorder.optimizers = (opt_d, opt_g)
        recorder.rows = {opt_d: dict(calls=0), opt_g: dict(calls=0)}
        recorder.row = dict(outer_step=1800)
        recorder.advantage = 0.0
        recorder.adv_buf.extend([1.0] * 50 + [0.0] * 49)
        recorder.step(opt_g, real_step)
        assert recorder.reject_steps == [1800]
        for live, old in zip(generator.parameters(), saved):
            assert torch.equal(live, old)
        assert int(opt_g.state[param]["step"]) == step_before
        assert param.grad.abs().sum() == 0
        assert torch.equal(next(critic.parameters()), d_saved)
        recorder.phase = 2
        recorder.step(opt_g, real_step)
        assert recorder.row["g"]["rejected"] is True
        assert recorder.row["g"]["factor"] == 0.0
        assert int(opt_g.state[param]["step"]) == step_before


def test_short_advantage_ring_still_takes_the_g_adam_step():
    torch.manual_seed(1)
    generator = torch.nn.Linear(3, 2, bias=False)
    critic = torch.nn.Linear(2, 1, bias=False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=.2, betas=(0., .99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=.2, betas=(0., .99))
    data = torch.randn(4, 3)
    generator(data).sum().backward()
    real_step = torch.optim.Adam.step
    real_step(opt_g)
    param = next(generator.parameters())
    step_before = int(opt_g.state[param]["step"])
    saved = [p.detach().clone() for p in generator.parameters()]
    opt_g.zero_grad()
    generator(data).sum().backward()
    with pr84_reach_candidate(task="mode_hold", ramp="stall", adv_reject=True) as (recorder, _source):
        recorder.armed = True
        recorder.enabled = True
        recorder.passthrough = False
        recorder.phase = 1
        recorder.optimizers = (opt_d, opt_g)
        recorder.rows = {opt_d: dict(calls=0), opt_g: dict(calls=0)}
        recorder.row = dict(outer_step=12)
        recorder.advantage = 0.4
        recorder.step(opt_g, real_step)
        assert recorder.reject_steps == []
        assert any(not torch.equal(live, old) for live, old in zip(generator.parameters(), saved))
        assert int(opt_g.state[param]["step"]) == step_before + 1


def test_full_flat_advantage_ring_still_takes_the_g_adam_step():
    torch.manual_seed(2)
    generator = torch.nn.Linear(3, 2, bias=False)
    critic = torch.nn.Linear(2, 1, bias=False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=.2, betas=(0., .99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=.2, betas=(0., .99))
    data = torch.randn(4, 3)
    real_step = torch.optim.Adam.step
    generator(data).sum().backward()
    real_step(opt_g)
    param = next(generator.parameters())
    step_before = int(opt_g.state[param]["step"])
    saved = [p.detach().clone() for p in generator.parameters()]
    opt_g.zero_grad()
    generator(data).sum().backward()
    with pr84_reach_candidate(task="mode_hold", ramp="stall", adv_reject=True) as (recorder, _source):
        recorder.armed = True
        recorder.enabled = True
        recorder.passthrough = False
        recorder.phase = 1
        recorder.optimizers = (opt_d, opt_g)
        recorder.rows = {opt_d: dict(calls=0), opt_g: dict(calls=0)}
        recorder.row = dict(outer_step=40)
        recorder.adv_buf.extend([0.2] * 100)
        recorder.advantage = 0.2
        recorder.step(opt_g, real_step)
        assert recorder.reject_steps == []
        assert recorder.adv_checks == 1
        assert any(not torch.equal(live, old) for live, old in zip(generator.parameters(), saved))
        assert int(opt_g.state[param]["step"]) == step_before + 1


def test_logged_ring_listener_reaches_the_arm():
    with pr84_reach_candidate(task="mode_hold", ramp="stall", adv_reject=True) as (recorder, _source):
        notify_ring(400, 8, .95)
        assert recorder.arm_update == 400


def test_factory_installs_and_restores_reach_recorder():
    with pr84_reach_candidate(task="mode_hold", reach=.7) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.reach == .7
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)

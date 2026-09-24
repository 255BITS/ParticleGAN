import pytest
import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import (
    ReachRecorder, pr84_reach_candidate, reach_width, trust_collapsed,
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


def test_trust_collapse_uses_only_the_two_trust_numbers():
    assert trust_collapsed(.5, .1)
    assert trust_collapsed(.5, 0.)
    assert trust_collapsed(1., .2)
    assert not trust_collapsed(.49, .1)
    assert not trust_collapsed(.5, .11)
    assert not trust_collapsed(.05, .27)
    assert not trust_collapsed(None, .01)
    assert not trust_collapsed(0., 0.)
    assert trust_collapsed.__code__.co_varnames[:2] == ("pre", "post")


def test_ring_log_arms_only_on_full_hq_and_rollback_stays_off_before_that():
    with pr84_reach_candidate(task="mode_hold", ramp="stall", trust_rollback=True) as (recorder, _source):
        recorder.note_ring(100, 6, 1.)
        recorder.note_ring(200, 8, .89)
        assert not recorder.armed
        assert not recorder._watching_g(object())
        recorder.note_ring(650, 8, .9)
        assert recorder.armed and recorder.arm_update == 650 and recorder.arm_phase == "cold_acquire"
        recorder.note_ring(700, 8, 1.)
        assert recorder.arm_update == 650


def test_trust_rollback_restores_g_params_and_adam_moments_and_leaves_d():
    torch.manual_seed(0)
    g = torch.nn.Linear(3, 2, bias=False)
    d = torch.nn.Linear(2, 1, bias=False)
    opt_g = torch.optim.Adam(g.parameters(), lr=.2, betas=(0., .99))
    opt_d = torch.optim.Adam(d.parameters(), lr=.2, betas=(0., .99))
    x = torch.randn(4, 3)
    y = g(x).sum()
    y.backward()
    opt_g.step()
    opt_g.zero_grad()
    d_param = next(d.parameters())
    d_before = d_param.detach().clone()
    real_adam = torch.optim.Adam.step
    with pr84_reach_candidate(task="mode_hold", ramp="stall", trust_rollback=True) as (recorder, _source):
        recorder.armed = True
        recorder.enabled = True
        recorder.passthrough = False
        recorder.phase = 1
        recorder.optimizers = (opt_d, opt_g)
        recorder.records = [dict(g=dict(factor=.8, rho=.3))] * 4
        recorder.row = dict(outer_step=9)
        recorder._capture_g_rollback(opt_g)
        saved = [p.detach().clone() for p in g.parameters()]
        saved_state = {id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                       for p, s in opt_g.state.items()}
        loss = g(x).sum()
        opt_g.zero_grad()
        loss.backward()
        real_adam(opt_g)
        assert any(not torch.equal(p, s) for p, s in zip(g.parameters(), saved))
        recorder.phase = 2
        recorder.row["g"] = dict(factor=.05, rho=5.)
        recorder._reject_g_on_trust_collapse(opt_g)
        for p, s in zip(g.parameters(), saved):
            assert torch.equal(p, s)
        for p, s in opt_g.state.items():
            for k, v in s.items():
                old = saved_state[id(p)][k]
                if torch.is_tensor(v):
                    assert torch.equal(v, old)
                else:
                    assert v == old
        assert torch.equal(d_param, d_before)
        assert recorder.rollback_steps == [9]
        assert recorder.row["g"]["rolled_back"] is True
        assert recorder.row["g"]["factor"] == pytest.approx(.8)


def test_factory_installs_and_restores_reach_recorder():
    with pr84_reach_candidate(task="mode_hold", reach=.7) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.reach == .7
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)

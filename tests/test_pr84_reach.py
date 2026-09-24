import pytest
import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import (
    ReachRecorder, pr84_reach_candidate, reach_width, window_trust_open,
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


def test_trust_open_is_a_fifty_step_mean_not_a_one_step_ratio():
    # Published dropout onset: the window mean moves .05 → .27.
    opened, mean_now, mean_lag = window_trust_open([.05] * 50 + [.27] * 49, .27)
    assert opened
    assert mean_lag == pytest.approx(.05)
    assert mean_now == pytest.approx(.27)
    # A smaller opening stays under the fixed .10 delta.
    opened, mean_now, mean_lag = window_trust_open([.05] * 50 + [.12] * 49, .12)
    assert not opened
    assert mean_now - mean_lag == pytest.approx(.07)
    # One-step collapse (.08 → .01) does not open the 50-step mean.
    opened, mean_now, mean_lag = window_trust_open([.08] * 99, .01)
    assert not opened
    assert mean_lag == pytest.approx(.08)
    assert mean_now == pytest.approx((49 * .08 + .01) / 50)
    assert mean_now < mean_lag
    # The lag window does not exist before 99 accepted factors.
    assert window_trust_open([.05] * 98, 1.) == (False, None, None)
    assert window_trust_open([], None) == (False, None, None)


def test_ring_log_arms_only_on_full_hq_and_reject_stays_off_before_that():
    from benchmarks.locked_shared.observation import notify_ring, set_ring_listener

    set_ring_listener(None)
    with pr84_reach_candidate(task="mode_hold", ramp="stall", trust_open_reject=True) as (recorder, _source):
        assert recorder.curvature_bound == pytest.approx(.25)
        assert recorder.d_curvature_bound == pytest.approx(3.)
        recorder.note_ring(100, 6, 1.)
        recorder.note_ring(200, 8, .89)
        assert not recorder.armed
        assert not recorder._watching_g(object())
        notify_ring(650, 8, .9)
        assert recorder.armed and recorder.arm_update == 650 and recorder.arm_phase == "cold_acquire"
        notify_ring(700, 8, 1.)
        assert recorder.arm_update == 650
    assert set_ring_listener(None) is None


def test_trust_open_reject_restores_g_and_leaves_d():
    torch.manual_seed(0)
    g = torch.nn.Linear(3, 2, bias=False)
    d = torch.nn.Linear(2, 1, bias=False)
    opt_g = torch.optim.Adam(g.parameters(), lr=.2, betas=(0., .99))
    opt_d = torch.optim.Adam(d.parameters(), lr=.2, betas=(0., .99))
    x = torch.randn(4, 3)
    loss = g(x).sum()
    loss.backward()
    opt_g.step()
    opt_g.zero_grad()
    d_param = next(d.parameters()).detach().clone()
    real_adam = torch.optim.Adam.step
    with pr84_reach_candidate(task="mode_hold", ramp="stall", trust_open_reject=True) as (recorder, _source):
        recorder.armed = True
        recorder.enabled = True
        recorder.passthrough = False
        recorder.phase = 1
        recorder.optimizers = (opt_d, opt_g)
        recorder.records = [dict(g=dict(factor=.05, rho=5.))] * 50 + [dict(g=dict(factor=.27, rho=.9))] * 49
        recorder.row = dict(outer_step=1722, g=dict(factor=.27, rho=.9))
        recorder._capture_g_reject(opt_g)
        saved = [p.detach().clone() for p in g.parameters()]
        saved_state = {id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                       for p, s in opt_g.state.items()}
        opt_g.zero_grad()
        g(x).sum().backward()
        real_adam(opt_g)
        with torch.no_grad():
            next(g.parameters()).add_(1.)
        assert any(not torch.equal(p, s) for p, s in zip(g.parameters(), saved))
        recorder.phase = 2
        recorder._consider_trust_open(opt_g, True)
        for p, s in zip(g.parameters(), saved):
            assert torch.equal(p, s)
        for p, s in opt_g.state.items():
            for k, v in s.items():
                old = saved_state[id(p)][k]
                if torch.is_tensor(v):
                    assert torch.equal(v, old)
                else:
                    assert v == old
        assert torch.equal(next(d.parameters()), d_param)
        assert recorder.reject_steps == [1722]
        assert recorder.row["g"]["rejected"] is True
        assert recorder.row["g"]["factor"] == pytest.approx(.27)
        assert recorder.row["g"]["trust_open"] == pytest.approx(.22)
        # Under the delta, the same snapshot is kept.
        recorder.records = [dict(g=dict(factor=.05, rho=5.))] * 99
        recorder.row = dict(outer_step=1723, g=dict(factor=.12, rho=2.))
        recorder._capture_g_reject(opt_g)
        moved = [p.detach().clone() for p in g.parameters()]
        with torch.no_grad():
            next(g.parameters()).add_(.5)
        recorder._consider_trust_open(opt_g, True)
        for p, s in zip(g.parameters(), moved):
            assert not torch.equal(p, s)
        assert recorder.reject_steps == [1722]
        # Before the arm the opening is only tracked.
        recorder.armed = False
        recorder._g_snap_params = [p.detach().clone() for p in g.parameters()]
        recorder.records = [dict(g=dict(factor=.05, rho=5.))] * 50 + [dict(g=dict(factor=.27, rho=.9))] * 49
        recorder.row = dict(outer_step=1800, g=dict(factor=.27, rho=.9))
        before = [p.detach().clone() for p in g.parameters()]
        recorder._consider_trust_open(opt_g, False)
        for p, s in zip(g.parameters(), before):
            assert torch.equal(p, s)
        assert recorder.reject_steps == [1722]
        assert recorder.row["g"]["trust_mean_probe"] == pytest.approx(.27)


def test_factory_installs_and_restores_reach_recorder():
    with pr84_reach_candidate(task="mode_hold", reach=.7) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.reach == .7
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)

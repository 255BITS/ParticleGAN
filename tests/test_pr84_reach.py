import pytest

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import (
    RECOVERY_REACH, ReachRecorder, note_support, pr84_reach_candidate, reach_width, recovery_width)


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


def test_support_fall_is_a_hard_dip_and_ignores_g_trust():
    dip, ema = note_support(None, -0.10)
    assert dip is False and ema == pytest.approx(-0.10)
    for _ in range(30):
        dip, ema = note_support(ema, ema - 0.01)
        assert dip is False
    fell, ema = note_support(ema, ema - 0.15)
    assert fell is True
    # The next sample at the new level is not another fall. G trust is not an argument.
    again, _ema = note_support(ema, ema)
    assert again is False
    assert "g_factor" not in note_support.__code__.co_varnames


def test_recovery_width_is_one_fixed_override():
    assert recovery_width(1., stalled=True, dip=True) == RECOVERY_REACH
    assert recovery_width(.2, stalled=False, dip=True) == RECOVERY_REACH
    assert recovery_width(1., stalled=True, dip=False) == pytest.approx(.5)
    assert recovery_width(.2, stalled=False, dip=False) == pytest.approx(reach_width(.2))
    assert .5 < RECOVERY_REACH < 1.


def test_recovery_dip_reuses_one_decision_per_outer_step():
    recorder = ReachRecorder(start_step=0)
    recorder.phase = 0
    recorder.records = [dict(g=dict(factor=1.))] * 60
    recorder._critic_support_gap = lambda: -0.1
    assert recorder._recovery_dip() is False
    recorder.outer_steps = 1
    recorder._critic_support_gap = lambda: -0.4
    assert recorder._recovery_dip() is True
    calls = []
    recorder._critic_support_gap = lambda: calls.append(1) or 0.
    recorder.phase = 1
    assert recorder._recovery_dip() is True and calls == []
    recorder.records = [dict(g=dict(factor=.01))] * 60
    assert recorder._recovery_dip() is True and calls == []


def test_factory_installs_and_restores_reach_recorder():
    with pr84_reach_candidate(task="mode_hold", reach=.7) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.reach == .7
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)

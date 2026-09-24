import json

import pytest

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_delayed_g_bound import BUDGET_UPDATE, POST_BUDGET_G_BOUND
from reports.toy100.pr84_reach_candidate import REACH, ReachRecorder, reach_width
from reports.toy100.pr84_stall_counterfactual import (
    METHOD, CounterfactualStallRecorder, stall_counterfactual_g25, trust_factor,
)


def _recorder():
    recorder = CounterfactualStallRecorder(start_step=0)
    recorder.curvature_bound = base.G_CURVATURE_BOUND
    recorder.game_bound = False
    recorder.passthrough = False
    recorder.phase = 2
    recorder.ramp = "stall"
    recorder.reach = REACH
    recorder.optimizers = (object(), object())
    recorder.row = {}
    return recorder


def _parent_step(self, optimizer, ordinary_step, closure=None):
    rho = 1.
    self.row["g"] = dict(rho=rho, factor=trust_factor(rho, self.curvature_bound))
    return None


def test_constants_are_the_host_bet_not_a_retune():
    assert METHOD == "stall_reach_delayed_g125_scored_on_counterfactual_g25"
    assert base.G_CURVATURE_BOUND == pytest.approx(.25)
    assert BUDGET_UPDATE == 1200
    assert POST_BUDGET_G_BOUND == pytest.approx(.125)
    assert REACH == pytest.approx(.5)


def test_pre_budget_apply_matches_141_and_does_not_rewrite_stall(monkeypatch):
    seen = []

    def parent_step(self, optimizer, ordinary_step, closure=None):
        seen.append(self.curvature_bound)
        self.row["g"] = dict(rho=1., factor=trust_factor(1., self.curvature_bound))
        return None

    monkeypatch.setattr(base.SmoothedBothBoundRecorder, "step", parent_step)
    recorder = _recorder()
    recorder.note_update(1198)
    recorder.step(recorder.optimizers[1], lambda *args, **kwargs: None)
    assert seen == [pytest.approx(.25)]
    assert "stall_score" not in recorder.row
    assert recorder.row["g"]["factor"] == pytest.approx(.25)
    assert recorder.fires == 0
    assert recorder._scored_trust(recorder.row) == pytest.approx(.25)


@pytest.mark.parametrize("rho", [0., .1, .2, .5, 1., 2., 4.])
def test_latched_apply_stays_125_and_stall_score_is_the_25_cap(monkeypatch, rho):
    monkeypatch.setattr(
        base.SmoothedBothBoundRecorder, "step",
        lambda self, optimizer, ordinary_step, closure=None: self.row.__setitem__(
            "g", dict(rho=rho, factor=trust_factor(rho, self.curvature_bound))) or None)
    recorder = _recorder()
    recorder.note_update(1199)
    assert recorder.armed is True
    recorder.step(recorder.optimizers[1], lambda *args, **kwargs: None)
    assert recorder.curvature_bound == pytest.approx(.25)
    assert recorder.row["g"]["factor"] == pytest.approx(trust_factor(rho, .125))
    scored = recorder.row["stall_score"]
    assert scored["factor"] == pytest.approx(trust_factor(rho, .25))
    assert scored["cap"] == pytest.approx(.25)
    assert scored["applied_cap"] == pytest.approx(.125)
    assert scored["applied_factor"] == pytest.approx(recorder.row["g"]["factor"])
    assert recorder.row["delayed_g_bound"]["g_bound_after"] == pytest.approx(.125)


def test_stall_predicate_ignores_the_clamp_shortened_factor():
    recorder = _recorder()
    # Mean applied factor .08 would trip #107. The .25-cap factor .16 would not.
    recorder.records = [dict(g=dict(factor=.08), stall_score=dict(factor=.16))] * 50
    assert recorder._stalled(.6) is False
    assert recorder._stalled(.59) is False
    recorder.records = [dict(g=dict(factor=.08), stall_score=dict(factor=.08))] * 50
    assert recorder._stalled(.6) is True
    # Pre-budget rows have no stall_score, so the actual .25 factor is the score.
    recorder.records = [dict(g=dict(factor=.05))] * 50
    assert recorder._stalled(.6) is True
    recorder.records = [dict(g=dict(factor=.2))] * 50
    assert recorder._stalled(1.) is False


def test_width_rule_is_unchanged_once_the_predicate_is_scored(monkeypatch):
    def parent_arm(self):
        self._smooth_on = True
        self.row["critic_sharpness"] = .6

    monkeypatch.setattr(base.SmoothedBothBoundRecorder, "_arm_smoothed_critic", parent_arm)
    recorder = _recorder()
    recorder.records = [dict(g=dict(factor=.08), stall_score=dict(factor=.16))] * 50
    recorder._arm_smoothed_critic()
    assert recorder._smooth_width == pytest.approx(reach_width(.6, REACH))
    assert recorder._smooth_width < .5
    recorder.records = [dict(g=dict(factor=.05), stall_score=dict(factor=.05))] * 50
    recorder._arm_smoothed_critic()
    assert recorder._smooth_width == pytest.approx(.5)


def test_armed_log_line_carries_the_counterfactual_factor(monkeypatch, capsys):
    monkeypatch.setattr(base.SmoothedBothBoundRecorder, "step", _parent_step)
    recorder = _recorder()
    recorder.note_update(1199)
    recorder.step(recorder.optimizers[1], lambda *args, **kwargs: None)
    line = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert line["event"] == "G_BOUND"
    assert line["g_bound_after"] == pytest.approx(.125)
    assert line["stall_cap"] == pytest.approx(.25)
    assert line["applied_factor"] == pytest.approx(.125)
    assert line["stall_factor"] == pytest.approx(.25)
    assert "cpu" in line


def test_factory_is_stall_reach_on_the_delayed_clamp_and_restores():
    with stall_counterfactual_g25(task="mode_hold") as (recorder, _source):
        assert isinstance(recorder, CounterfactualStallRecorder)
        assert isinstance(recorder, ReachRecorder)
        assert recorder.ramp == "stall"
        assert recorder.game_bound is False
        assert recorder.reach == pytest.approx(.5)
        assert recorder.index_offset == 1
        assert recorder.curvature_bound == pytest.approx(.25)
    assert base.SmoothedBothBoundRecorder is not CounterfactualStallRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, CounterfactualStallRecorder)


def test_trajectory_still_cannot_arm():
    with stall_counterfactual_g25(task="trajectory") as (recorder, _source):
        assert recorder.index_offset == 0
        recorder.note_update(400)
        assert recorder.update_index == 400 and recorder.armed is False

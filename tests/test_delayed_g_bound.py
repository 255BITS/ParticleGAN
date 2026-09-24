import json

import pytest

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_delayed_g_bound import (
    BUDGET_UPDATE, POST_BUDGET_G_BOUND, DelayedBudgetReachRecorder, delayed_budget_g_bound,
)
from reports.toy100.pr84_reach_candidate import ReachRecorder


def _recorder():
    recorder = DelayedBudgetReachRecorder(start_step=0)
    recorder.curvature_bound = base.G_CURVATURE_BOUND
    recorder.game_bound = False
    recorder.passthrough = False
    recorder.phase = 2
    recorder.optimizers = (object(), object())
    recorder.row = {}
    return recorder


def test_budget_constants_match_the_bet():
    assert base.G_CURVATURE_BOUND == pytest.approx(.25)
    assert BUDGET_UPDATE == 1200
    assert POST_BUDGET_G_BOUND == pytest.approx(.125)


def test_pre_budget_updates_stay_on_the_stall_reach_cap():
    recorder = _recorder()
    for host_step in (0, 650, 1198):
        recorder.note_update(host_step)
        assert recorder.update_index == host_step + 1
        assert recorder.armed is False


def test_arm_latches_at_update_1200_and_stays_latched():
    recorder = _recorder()
    recorder.note_update(1198)
    assert recorder.update_index == 1199 and recorder.armed is False
    recorder.note_update(1199)
    assert recorder.update_index == 1200 and recorder.armed is True
    recorder.note_update(1198)
    assert recorder.armed is True
    recorder.note_update(2399)
    assert recorder.update_index == 2400 and recorder.armed is True


def test_g_step_uses_125_only_after_the_budget(monkeypatch, capsys):
    seen = []

    def parent_step(self, optimizer, ordinary_step, closure=None):
        seen.append(self.curvature_bound)
        self.row["g"] = dict(rho=1., factor=self.curvature_bound)
        return None

    monkeypatch.setattr(base.SmoothedBothBoundRecorder, "step", parent_step)
    recorder = _recorder()
    opt_g = recorder.optimizers[1]

    recorder.note_update(1198)
    recorder.step(opt_g, lambda *args, **kwargs: None)
    assert seen[-1] == pytest.approx(.25)
    assert recorder.curvature_bound == pytest.approx(.25)
    assert recorder.fires == 0
    assert recorder.row["delayed_g_bound"]["g_bound_after"] == pytest.approx(.25)
    assert recorder.row["delayed_g_bound"]["armed"] is False

    recorder.note_update(1199)
    recorder.step(opt_g, lambda *args, **kwargs: None)
    assert seen[-1] == pytest.approx(.125)
    assert recorder.curvature_bound == pytest.approx(.25)
    assert recorder.d_curvature_bound == pytest.approx(3.)
    assert recorder.fires == 1
    logged = recorder.row["delayed_g_bound"]
    assert logged["armed"] is True
    assert logged["g_bound_before"] == pytest.approx(.25)
    assert logged["g_bound_after"] == pytest.approx(.125)
    assert logged["update_index"] == 1200 and logged["fires"] == 1

    recorder.note_update(1200)
    recorder.step(opt_g, lambda *args, **kwargs: None)
    assert recorder.fires == 2
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert [line["update_index"] for line in lines] == [1199, 1200, 1201]
    assert [line["armed"] for line in lines] == [False, True, True]
    assert lines[-1]["fires"] == 2
    assert "cpu" in lines[-1]


def test_discriminator_steps_do_not_fire_the_clamp(monkeypatch):
    monkeypatch.setattr(base.SmoothedBothBoundRecorder, "step", lambda *args, **kwargs: None)
    recorder = _recorder()
    recorder.note_update(1500)
    recorder.step(recorder.optimizers[0], lambda *args, **kwargs: None)
    assert recorder.fires == 0 and recorder.applies == 0
    assert recorder.curvature_bound == pytest.approx(.25)


def test_factory_is_stall_reach_and_restores_the_smoothed_recorder():
    with delayed_budget_g_bound(task="mode_hold") as (recorder, _source):
        assert isinstance(recorder, DelayedBudgetReachRecorder)
        assert isinstance(recorder, ReachRecorder)
        assert recorder.ramp == "stall"
        assert recorder.game_bound is False
        assert recorder.curvature_bound == pytest.approx(.25)
        assert recorder.index_offset == 1
    assert base.SmoothedBothBoundRecorder is not DelayedBudgetReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, DelayedBudgetReachRecorder)


def test_trajectory_index_is_already_one_based_and_cannot_arm():
    with delayed_budget_g_bound(task="trajectory") as (recorder, _source):
        assert recorder.index_offset == 0
        recorder.note_update(400)
        assert recorder.update_index == 400 and recorder.armed is False

import json

import pytest
import torch
from unittest.mock import patch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_delayed_arm_g_lr import ARM_UPDATE_INDEX, DelayedArmRecorder
from reports.toy100.pr84_delayed_arm_width_hold import (
    HELD_WIDTH, WidthHoldRecorder, pr84_delayed_arm_width_hold,
)
from reports.toy100.pr84_reach_candidate import REACH, ReachRecorder


def _pair(g_lr=.00425, prior_lr=.0085, d_lr=.00425):
    generator = torch.nn.Parameter(torch.zeros(2))
    prior = torch.nn.Parameter(torch.zeros(2))
    critic = torch.nn.Parameter(torch.zeros(2))
    opt_g = torch.optim.Adam([
        dict(params=[generator], lr=g_lr),
        dict(params=[prior], lr=prior_lr, _comparison_prior=True),
    ])
    opt_d = torch.optim.Adam([critic], lr=d_lr)
    return opt_d, opt_g


def _recorder():
    recorder = WidthHoldRecorder(start_step=0)
    recorder.ramp = "stall"
    recorder.curvature_bound = base.G_CURVATURE_BOUND
    opt_d, opt_g = _pair()
    recorder.optimizers = (opt_d, opt_g)
    recorder.rows = {opt_d: dict(role="d", calls=0), opt_g: dict(role="g", calls=0)}
    recorder.row = dict(outer_step=500)
    recorder.phase = 1
    return recorder


def _arm(recorder, sharp):
    def stub(self):
        self._smooth_on = True
        self._smooth_width = base.SMOOTH_WIDTH_CAP
        self.row["critic_sharpness"] = sharp
        self.row["critic_width"] = base.SMOOTH_WIDTH_CAP

    with patch.object(base.SmoothedBothBoundRecorder, "_arm_smoothed_critic", stub):
        recorder._arm_smoothed_critic()


def test_pre_arm_stall_still_widens_to_half_and_peak_opening_is_unchanged(capsys):
    stalled = _recorder()
    stalled.records = [dict(g=dict(factor=.05))] * 50
    _arm(stalled, .9)
    assert stalled._stalled(.9)
    assert stalled._smooth_width == pytest.approx(REACH)
    assert stalled.row["critic_width"] == pytest.approx(.5)
    assert stalled.width_fires == 0
    assert capsys.readouterr().out == ""

    opening = _recorder()
    opening.records = [dict(g=dict(factor=.21))] * 50
    opening.row["outer_step"] = 2190
    _arm(opening, .8916551470756531)
    assert not opening._stalled(.8916551470756531)
    assert opening._smooth_width == pytest.approx(.44582757353782654)
    assert opening.width_fires == 0


def test_post_arm_hold_replaces_stall_half_and_the_peak_opening(capsys):
    stalled = _recorder()
    stalled.records = [dict(g=dict(factor=.05))] * 50
    stalled.consider_diagnostic(ARM_UPDATE_INDEX, 8, .9)
    stalled.row["outer_step"] = 1524
    capsys.readouterr()
    _arm(stalled, .9)
    assert stalled._smooth_width == pytest.approx(HELD_WIDTH)
    assert stalled.row["critic_width"] == pytest.approx(.15)
    row = json.loads(capsys.readouterr().out.strip())
    assert row["event"] == "width_hold_apply" and row["armed"] is True
    assert row["update_index"] == 1524
    assert row["width_before"] == pytest.approx(.5)
    assert row["width_after"] == pytest.approx(.15)
    assert row["fires"] == 1 and row["phase"] == 1

    opening = _recorder()
    opening.consider_diagnostic(1200, 8, 1.)
    opening.records = [dict(g=dict(factor=.21))] * 50
    opening.row["outer_step"] = 2190
    capsys.readouterr()
    _arm(opening, .8916551470756531)
    assert opening._smooth_width == pytest.approx(.15)
    row = json.loads(capsys.readouterr().out.strip())
    assert row["width_before"] == pytest.approx(.44582757353782654)
    assert row["width_after"] == pytest.approx(.15)
    assert row["update_index"] == 2190 and row["fires"] == 1


def test_hold_is_sticky_and_does_not_arm_early_or_when_disabled(capsys):
    recorder = _recorder()
    recorder.records = [dict(g=dict(factor=.05))] * 50
    recorder.consider_diagnostic(650, 8, 1.)
    _arm(recorder, 1.)
    assert not recorder.armed
    assert recorder._smooth_width == pytest.approx(.5)
    assert recorder.width_fires == 0

    recorder.consider_diagnostic(1200, 8, .9)
    recorder.consider_diagnostic(2190, 0, 0.)
    assert recorder.armed and recorder.arm_update_index == 1200
    recorder.row["outer_step"] = 2191
    capsys.readouterr()
    _arm(recorder, .2)
    assert recorder._smooth_width == pytest.approx(.15)
    assert json.loads(capsys.readouterr().out)["width_before"] == pytest.approx(.15)

    recorder.enabled = False
    recorder.row["critic_width"] = .5
    recorder._smooth_width = .5
    _arm(recorder, 1.)
    assert recorder._smooth_width == pytest.approx(.5)
    assert recorder.width_fires == 1


def test_g_adam_half_still_applies_only_after_the_same_arm():
    recorder = _recorder()
    recorder.passthrough = True
    seen = {}

    def ordinary(opt, closure=None):
        seen["lr"] = opt.param_groups[0]["lr"]
        return "stepped"

    _, opt_g = recorder.optimizers
    recorder._host_step = 1199
    assert recorder.step(opt_g, ordinary) == "stepped"
    assert seen["lr"] == pytest.approx(.00425)
    assert recorder.fires == 0
    recorder.consider_diagnostic(1200, 8, .9)
    recorder._host_step = 1200
    recorder.step(opt_g, ordinary)
    assert seen["lr"] == pytest.approx(.002125)
    assert [group["lr"] for group in opt_g.param_groups] == pytest.approx([.00425, .0085])
    assert recorder.fires == 1


def test_factory_is_stall_reach_on_the_delayed_arm_and_restores_the_host():
    from benchmarks.locked_shared.observation import Recorder
    import benchmarks.toy100.continuous_probe as probe

    record, run = Recorder.record, probe._run_extended
    with pr84_delayed_arm_width_hold(task="mode_hold") as (recorder, _source):
        assert isinstance(recorder, WidthHoldRecorder)
        assert isinstance(recorder, DelayedArmRecorder)
        assert isinstance(recorder, ReachRecorder)
        assert recorder.ramp == "stall" and recorder.game_bound is False
        assert recorder.held_width == pytest.approx(.15)
        assert recorder.curvature_bound == pytest.approx(base.G_CURVATURE_BOUND)
        assert probe._run_extended is not run and Recorder.record is not record
    assert base.SmoothedBothBoundRecorder is not WidthHoldRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, WidthHoldRecorder)
    assert Recorder.record is record and probe._run_extended is run

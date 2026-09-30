import json

import pytest
import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_delayed_arm_g_lr import (
    ARM_UPDATE_INDEX, DelayedArmRecorder, diagnostic_log, pr84_delayed_arm_g_lr,
)
from reports.toy100.pr84_reach_candidate import ReachRecorder


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
    recorder = DelayedArmRecorder(start_step=0)
    recorder.ramp = "stall"
    recorder.curvature_bound = base.G_CURVATURE_BOUND
    opt_d, opt_g = _pair()
    recorder.optimizers = (opt_d, opt_g)
    recorder.rows = {opt_d: dict(role="d", calls=0), opt_g: dict(role="g", calls=0)}
    return recorder, opt_d, opt_g


def test_early_full_ring_does_not_arm_and_1200_does(capsys):
    recorder, _, opt_g = _recorder()
    for step, modes, hq in ((650, 8, 1.), (1199, 8, .95), (1200, 7, 1.), (1200, 8, .899)):
        recorder.consider_diagnostic(step, modes, hq)
        assert not recorder.armed
    recorder.consider_diagnostic(ARM_UPDATE_INDEX, 8, .9)
    assert recorder.armed and recorder.arm_update_index == 1200
    assert recorder.arm_g_lr == pytest.approx(.00425)
    assert recorder._pre_arm == pytest.approx([.00425, .0085])
    recorder.consider_diagnostic(1300, 0, 0.)
    assert recorder.armed and recorder.arm_update_index == 1200
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["event"] == "g_lr_arm" and row["armed"] is True
    assert row["update_index"] == 1200
    assert row["g_lr_before"] == pytest.approx(.00425)
    assert row["g_lr_after"] == pytest.approx(.002125)
    assert row["fires"] == 0
    assert opt_g.param_groups[0]["lr"] == pytest.approx(.00425)


def test_disabled_or_unready_recorder_does_not_arm():
    recorder, _, _ = _recorder()
    recorder.enabled = False
    recorder.consider_diagnostic(1300, 8, 1.)
    assert not recorder.armed
    recorder.enabled = True
    recorder.optimizers = None
    recorder.consider_diagnostic(1300, 8, 1.)
    assert not recorder.armed


def test_apply_halves_g_groups_once_and_leaves_d(capsys):
    recorder, opt_d, opt_g = _recorder()
    recorder.consider_diagnostic(1200, 8, .9)
    capsys.readouterr()
    recorder._host_step = 1200
    recorder.passthrough = True
    seen = {}

    def ordinary(opt, closure=None):
        seen["lr"] = [group["lr"] for group in opt.param_groups]
        return "stepped"

    assert recorder.step(opt_g, ordinary) == "stepped"
    assert seen["lr"] == pytest.approx([.002125, .00425])
    assert [group["lr"] for group in opt_g.param_groups] == pytest.approx([.00425, .0085])
    opt_g.param_groups[0]["lr"] = .00425
    opt_g.param_groups[1]["lr"] = .0085
    recorder.step(opt_g, ordinary)
    assert seen["lr"] == pytest.approx([.002125, .00425])
    assert recorder.fires == 2
    recorder.step(opt_d, ordinary)
    assert seen["lr"] == pytest.approx([.00425])
    assert recorder.fires == 2
    apply = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert apply["event"] == "g_lr_apply" and apply["armed"] is True
    assert apply["update_index"] == 1201
    assert apply["g_lr_before"] == pytest.approx(.00425)
    assert apply["g_lr_after"] == pytest.approx(.002125)
    assert apply["fires"] == 2
    assert apply["prior_lr_before"] == pytest.approx(.0085)
    assert apply["prior_lr_after"] == pytest.approx(.00425)


def test_unarmed_adam_step_keeps_full_lr_and_metric_uses_half_after_arm():
    recorder, _, opt_g = _recorder()
    for group in opt_g.param_groups:
        group["params"][0].grad = torch.ones_like(group["params"][0])
    parameter = opt_g.param_groups[0]["params"][0]
    recorder.phase = 1
    full = parameter.detach().clone()
    recorder.step(opt_g, torch.optim.Adam.step)
    full_delta = (parameter.detach() - full).clone()
    full_metric = recorder.metric_g[0].clone()
    assert opt_g.param_groups[0]["lr"] == pytest.approx(.00425)

    recorder, _, opt_g = _recorder()
    for group in opt_g.param_groups:
        group["params"][0].grad = torch.ones_like(group["params"][0])
    parameter = opt_g.param_groups[0]["params"][0]
    recorder.phase = 1
    recorder.consider_diagnostic(1200, 8, 1.)
    base_param = parameter.detach().clone()
    seen = {}
    raw = torch.optim.Adam.step

    def ordinary(opt, closure=None):
        seen["during"] = opt.param_groups[0]["lr"]
        return raw(opt, closure)

    recorder.step(opt_g, ordinary)
    half_delta = parameter.detach() - base_param
    assert seen["during"] == pytest.approx(.002125)
    assert opt_g.param_groups[0]["lr"] == pytest.approx(.00425)
    assert half_delta == pytest.approx(full_delta / 2)
    assert recorder.metric_g[0] == pytest.approx(full_metric / 2)


def test_probe_log_uses_the_same_arm_rule():
    recorder, _, _ = _recorder()
    log = diagnostic_log(recorder)
    log(dict(event="checkpoint", step=650, modes=8, hq=1.))
    log(dict(event="checkpoint", step=1500, identity_mse=.01))
    assert not recorder.armed
    log(dict(event="checkpoint", step=1210, modes=8, hq=.91))
    assert recorder.armed and recorder.arm_update_index == 1210


def test_factory_is_stall_reach_and_restores_the_host():
    from benchmarks.locked_shared.observation import Recorder
    import benchmarks.toy100.continuous_probe as probe

    record, run = Recorder.record, probe._run_extended
    with pr84_delayed_arm_g_lr(task="mode_hold") as (recorder, _source):
        assert isinstance(recorder, DelayedArmRecorder)
        assert isinstance(recorder, ReachRecorder)
        assert recorder.ramp == "stall" and recorder.game_bound is False
        assert recorder.curvature_bound == pytest.approx(base.G_CURVATURE_BOUND)
        assert probe._run_extended is not run and Recorder.record is not record
        recorder.records = [dict(g=dict(factor=.05))] * 60
        assert recorder._stalled(.6)
    assert base.SmoothedBothBoundRecorder is not DelayedArmRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, DelayedArmRecorder)
    assert Recorder.record is record and probe._run_extended is run

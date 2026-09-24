import json

import pytest
import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_delayed_arm_extra_d import (
    ARM_UPDATE_INDEX, EXTRA_D_STEPS, DelayedArmExtraDRecorder, diagnostic_log,
    pr84_delayed_arm_extra_d,
)
from reports.toy100.pr84_reach_candidate import ReachRecorder


def _recorder():
    recorder = DelayedArmExtraDRecorder(start_step=0)
    recorder.ramp = "stall"
    recorder.curvature_bound = base.G_CURVATURE_BOUND
    return recorder


def test_early_full_ring_does_not_arm_and_1200_sticks(capsys):
    recorder = _recorder()
    for step, modes, hq in ((650, 8, 1.), (1199, 8, .95), (1200, 7, 1.), (1200, 8, .899)):
        recorder.consider_diagnostic(step, modes, hq)
        assert not recorder.armed
    recorder.consider_diagnostic(ARM_UPDATE_INDEX, 8, .9)
    assert recorder.armed and recorder.arm_update_index == 1200
    recorder.consider_diagnostic(1800, 0, 0.)
    assert recorder.armed and recorder.arm_update_index == 1200
    assert recorder._last_modes == 0 and recorder._last_hq == 0.
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["event"] == "extra_d_arm" and row["armed"] is True
    assert row["update_index"] == 1200 and row["modes"] == 8 and row["HQ"] == pytest.approx(.9)
    assert row["extra_d_steps_this_turn"] == 0
    assert recorder.fires == 0


def test_non_diagnostic_values_do_not_arm():
    recorder = _recorder()
    recorder.consider_diagnostic(1300, True, 1.)
    recorder.consider_diagnostic(1300, 8, float("nan"))
    recorder.consider_diagnostic("1200", 8, 1.)
    recorder.enabled = False
    recorder.consider_diagnostic(1300, 8, 1.)
    assert not recorder.armed


def test_fire_counts_split_acquire_and_dropout_window():
    recorder = _recorder()
    recorder._fire_updates = [1199, 1201, 1689, 1690, 2300, 2301]
    assert recorder.fire_counts() == (3, 2, 1, 1)


def test_probe_log_uses_the_same_arm_rule():
    recorder = _recorder()
    log = diagnostic_log(recorder)
    log(dict(event="checkpoint", step=650, modes=8, hq=1.))
    log(dict(event="checkpoint", step=1500, identity_mse=.01))
    assert not recorder.armed
    log(dict(event="checkpoint", step=1210, modes=8, hq=.91))
    assert recorder.armed and recorder.arm_update_index == 1210


def _adam_steps(optimizer):
    found = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        found.append(int(value.item() if torch.is_tensor(value) else value))
    return found


def _flat_state(recorder):
    blobs = []
    for opt in recorder.optimizers:
        for group in opt.param_groups:
            for param in group["params"]:
                blobs.append(param.detach().cpu())
                state = opt.state.get(param, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        blobs.append(state[key].detach().cpu())
                blobs.append(torch.tensor([float(group["lr"])]))
    return blobs


def test_unarmed_matches_stall_reach_and_keeps_bounds():
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)

    def run(delayed):
        policy = NoisePolicy(.029, .5, .1, 1200)
        if delayed:
            context = pr84_delayed_arm_extra_d(task="mode_hold")
        else:
            from reports.toy100.pr84_reach_candidate import pr84_reach_candidate
            context = pr84_reach_candidate(task="mode_hold", ramp="stall")
        with context as (recorder, _source):
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2), noise_policy=policy)
        return recorder

    plain, extra = run(False), run(True)
    assert not extra.armed and extra.fires == 0 and extra.extra_d_steps == 0
    assert extra.curvature_bound == pytest.approx(base.G_CURVATURE_BOUND)
    assert extra.d_curvature_bound == pytest.approx(base.D_CURVATURE_BOUND)
    assert plain.rng_replay_verified == extra.rng_replay_verified
    for old, new in zip(_flat_state(plain), _flat_state(extra), strict=True):
        assert torch.equal(old, new)
    assert set(_adam_steps(extra.optimizers[0])) == set(_adam_steps(extra.optimizers[1])) == {2}


def test_every_turn_after_arm_adds_two_d_steps_even_at_zero_modes(capsys):
    """The arm is not a mode-count latch: a later 0/0 still takes +2 D steps."""
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    original = DelayedArmExtraDRecorder.phases

    def phases(self, step, opt_d, opt_g, local):
        if step == 0:
            self.consider_diagnostic(650, 8, 1.)
        elif step == 1:
            self.consider_diagnostic(1200, 8, .95)
        elif step == 2:
            self.consider_diagnostic(1210, 0, 0.)
        yield from original(self, step, opt_d, opt_g, local)

    DelayedArmExtraDRecorder.phases = phases
    try:
        with pr84_delayed_arm_extra_d(task="mode_hold") as (recorder, _source):
            g_lr = mode_hold.LR
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy)
    finally:
        DelayedArmExtraDRecorder.phases = original
    assert recorder.arm_update_index == 1200
    assert recorder.fires == 2 and recorder.extra_d_steps == 2 * EXTRA_D_STEPS
    assert recorder._fire_updates == [2, 3]
    assert [row.get("extra_d_apply") for row in recorder.records] == [False, True, True]
    assert set(_adam_steps(recorder.optimizers[0])) == {3 + 2 * EXTRA_D_STEPS}
    assert set(_adam_steps(recorder.optimizers[1])) == {3}
    assert recorder.curvature_bound == pytest.approx(.25)
    assert recorder.d_curvature_bound == pytest.approx(3.)
    assert [group["lr"] for group in recorder.optimizers[1].param_groups] == pytest.approx(
        [g_lr] * len(recorder.optimizers[1].param_groups))
    assert [group["lr"] for group in recorder.optimizers[0].param_groups] == pytest.approx([g_lr])
    apply = [json.loads(line) for line in capsys.readouterr().out.splitlines()
             if '"extra_d_apply"' in line]
    assert len(apply) == 2
    assert apply[0]["armed"] is True and apply[0]["update_index"] == 2
    assert apply[0]["modes"] == 8 and apply[0]["HQ"] == pytest.approx(.95)
    assert apply[0]["extra_d_steps_this_turn"] == 2
    assert apply[1]["modes"] == 0 and apply[1]["HQ"] == 0.
    assert apply[1]["extra_d_steps_this_turn"] == 2
    acquire, dropout, after, before = recorder.fire_counts()
    # These synthetic updates are 2 and 3, so they sit before the calendar index.
    assert (acquire, dropout, after, before) == (2, 0, 0, 2)


def test_factory_is_stall_reach_and_restores_the_host():
    from benchmarks.locked_shared.observation import Recorder
    import benchmarks.toy100.continuous_probe as probe

    record, run = Recorder.record, probe._run_extended
    with pr84_delayed_arm_extra_d(task="mode_hold") as (recorder, _source):
        assert isinstance(recorder, DelayedArmExtraDRecorder)
        assert isinstance(recorder, ReachRecorder)
        assert recorder.ramp == "stall" and recorder.game_bound is False
        assert recorder.curvature_bound == pytest.approx(base.G_CURVATURE_BOUND)
        assert probe._run_extended is not run and Recorder.record is not record
        recorder.records = [dict(g=dict(factor=.05))] * 60
        assert recorder._stalled(.6)
    assert base.SmoothedBothBoundRecorder is not DelayedArmExtraDRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, DelayedArmExtraDRecorder)
    assert Recorder.record is record and probe._run_extended is run

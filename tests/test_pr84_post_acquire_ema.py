import copy

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from benchmarks.locked_shared.mode_hold import ModeHoldRecipe
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_post_acquire_ema_candidate import (
    ARM_HQ, ARM_MODES, TAU, PostAcquireEmaRecorder, pr84_post_acquire_ema_candidate)
from reports.toy100.pr84_reach_candidate import pr84_reach_candidate


def _signature(recorder):
    return [(row.get("critic_sharpness"), row.get("critic_width"), row["d"]["factor"],
             row["g"]["factor"], row.get("critic_advantage")) for row in recorder.records]


def test_constants_are_the_single_suggested_mechanism():
    assert TAU == pytest.approx(.99)
    assert ARM_MODES == 8 and ARM_HQ == pytest.approx(.9)
    recorder = PostAcquireEmaRecorder(start_step=0)
    assert recorder.ramp == "stall" and recorder.tau == pytest.approx(.99)
    assert recorder._ema_module is None


def test_unarmed_updates_match_stall_reach_and_restore_rng():
    torch.set_num_threads(1)

    def run(factory):
        torch.manual_seed(0)
        with factory(task="mode_hold") as (recorder, _source):
            mode_hold.train_mode_hold(ModeHoldRecipe(steps=2), seed=0)
            rng = torch.get_rng_state().clone()
        return recorder, rng

    stall, stall_rng = run(lambda **kwargs: pr84_reach_candidate(ramp="stall", **kwargs))
    ema, ema_rng = run(pr84_post_acquire_ema_candidate)
    assert ema._ema_module is None and ema._arm_step is None
    assert _signature(ema) == _signature(stall)
    assert torch.equal(ema_rng, stall_rng)


def test_forced_acquire_changes_the_generator_update():
    torch.set_num_threads(1)

    def run(factory):
        torch.manual_seed(0)
        with factory(task="mode_hold") as (recorder, _source):
            mode_hold.train_mode_hold(ModeHoldRecipe(steps=1), seed=0)
        return recorder

    stall = run(lambda **kwargs: pr84_reach_candidate(ramp="stall", **kwargs))

    def acquired(self, completed):
        return dict(modes=8, hq=1.)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(PostAcquireEmaRecorder, "_grade", acquired)
        ema = run(pr84_post_acquire_ema_candidate)
    assert ema._arm_step == 0 and ema._ema_updates == 1
    assert ema.records[-1]["ema_on"] is True
    assert _signature(ema) != _signature(stall)


def test_ema_update_is_one_mix_per_outer_step_on_phase_1_only():
    live = SimpleMLPDiscriminator(2, 4, 1, 0)
    ema = copy.deepcopy(live)
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.add_(.5)
    recorder = PostAcquireEmaRecorder(start_step=0, tau=.99)
    recorder._live_critic = live
    recorder._ema_module = ema
    before = [parameter.detach().clone() for parameter in ema.parameters()]
    recorder.phase = 0
    recorder._update_ema()
    assert recorder._ema_updates == 0
    recorder._ema_refreshed = False
    recorder.phase = 1
    recorder._update_ema()
    recorder._update_ema()
    assert recorder._ema_updates == 1
    for updated, old, current in zip(ema.parameters(), before, live.parameters()):
        assert torch.allclose(updated, old * .99 + current * .01)


def test_armed_generator_forward_reads_ema_weights_including_stencil():
    torch.manual_seed(0)
    with pr84_post_acquire_ema_candidate(task="mode_hold") as (recorder, _source):
        live = SimpleMLPDiscriminator(2, 8, 1, 1)
        ema = copy.deepcopy(live)
        with torch.no_grad():
            for parameter in ema.parameters():
                parameter.mul_(0.).add_(.25)
        x = torch.randn(6, 2)
        recorder._ema_module = ema.eval().requires_grad_(False)
        recorder._smooth_on = False
        recorder._smooth_width = 0.
        recorder._g_reads_ema = False
        plain_live = live(x).detach().clone()
        plain_ema = ema(x).detach().clone()
        recorder._g_reads_ema = True
        assert torch.allclose(live(x), plain_ema)
        assert not torch.allclose(plain_ema, plain_live)
        recorder._smooth_on = True
        recorder._smooth_width = .2
        recorder._g_reads_ema = False
        stenciled_ema = ema(x).detach().clone()
        recorder._g_reads_ema = True
        assert torch.allclose(live(x), stenciled_ema)
        recorder._g_reads_ema = False
        assert not torch.allclose(live(x), stenciled_ema)
    assert base.SmoothedBothBoundRecorder is not PostAcquireEmaRecorder

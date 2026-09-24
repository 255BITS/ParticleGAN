import pytest

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import ReachRecorder, pr84_reach_candidate, reach_width


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


def _adam_steps(optimizer):
    import torch
    found = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        found.append(int(value.item() if torch.is_tensor(value) else value))
    return found


def _flat_state(recorder):
    import torch
    blobs = []
    for opt in recorder.optimizers:
        for group in opt.param_groups:
            for param in group["params"]:
                blobs.append(param.detach().cpu())
                state = opt.state.get(param, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        blobs.append(state[key].detach().cpu())
    return blobs


def test_false_stall_matches_stall_reach_weights_and_moments():
    import torch
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)

    def run(extra_d):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with pr84_reach_candidate(task="mode_hold", ramp="stall", extra_d=extra_d) as (recorder, _source):
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy)
        return recorder

    plain, extra = run(False), run(True)
    assert extra.extra_d_steps == 0
    assert plain.rng_replay_verified == extra.rng_replay_verified
    for old, new in zip(_flat_state(plain), _flat_state(extra), strict=True):
        assert torch.equal(old, new)
    assert set(_adam_steps(plain.optimizers[0])) == set(_adam_steps(extra.optimizers[0])) == {3}


def test_stall_predicate_adds_exactly_one_d_adam_step():
    import torch
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    with pr84_reach_candidate(task="mode_hold", ramp="stall", extra_d=True) as (recorder, _source):
        recorder._stalled = lambda sharpness: True
        mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2), noise_policy=policy)
    assert recorder.extra_d_steps == 2
    assert all(row.get("extra_d") for row in recorder.records)
    assert set(_adam_steps(recorder.optimizers[0])) == {4}
    assert set(_adam_steps(recorder.optimizers[1])) == {2}
    assert recorder.reach == .5
    assert recorder.game_bound is False


def test_factory_installs_and_restores_reach_recorder():
    with pr84_reach_candidate(task="mode_hold", reach=.7) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.reach == .7
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)

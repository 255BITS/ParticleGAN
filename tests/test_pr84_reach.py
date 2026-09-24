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


def test_factory_installs_and_restores_reach_recorder():
    with pr84_reach_candidate(task="mode_hold", reach=.7) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.reach == .7
        assert recorder.mode_ttur is False
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)


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


def test_pre_arm_mode_ttur_matches_stall_reach():
    import torch
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)

    def run(mode_ttur):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with pr84_reach_candidate(task="mode_hold", ramp="stall", mode_ttur=mode_ttur) as (recorder, _source):
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2), noise_policy=policy)
        return recorder

    plain, extra = run(False), run(True)
    assert extra._fire_count == 0 and not extra._armed
    assert plain.rng_replay_verified == extra.rng_replay_verified
    for old, new in zip(_flat_state(plain), _flat_state(extra), strict=True):
        assert torch.equal(old, new)
    assert set(_adam_steps(plain.optimizers[0])) == set(_adam_steps(extra.optimizers[0])) == {2}


def test_mode_ttur_fires_two_d_steps_only_after_arm_when_modes_at_most_six():
    """≤6 after an 8-mode arm adds exactly two D Adam steps. G still steps. 7 does not fire."""
    import torch
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    script = [(8, 1.), (6, .4), (7, .95), (0, 0.)]
    with pr84_reach_candidate(task="mode_hold", ramp="stall", mode_ttur=True) as (recorder, _source):
        recorder._ring_coverage = lambda local: script.pop(0)
        mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=4), noise_policy=policy)
    assert recorder._armed_at == 0
    assert recorder._fire_count == 2
    assert recorder.extra_d_steps == 4
    assert [row.get("ttur_fire") for row in recorder.records] == [False, True, False, True]
    assert [fire["modes"] for fire in recorder._fires] == [6, 0]
    assert set(_adam_steps(recorder.optimizers[0])) == {8}
    assert set(_adam_steps(recorder.optimizers[1])) == {4}
    assert recorder.curvature_bound == base.G_CURVATURE_BOUND
    assert recorder.d_curvature_bound == base.D_CURVATURE_BOUND
    assert recorder.reach == .5


def test_mode_count_before_arm_does_not_add_d_steps():
    import torch
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200)
    script = [(6, .2), (0, 0.), (8, .91)]
    with pr84_reach_candidate(task="mode_hold", ramp="stall", mode_ttur=True) as (recorder, _source):
        recorder._ring_coverage = lambda local: script.pop(0)
        mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy)
    assert recorder._armed_at == 2 and recorder._fire_count == 0
    assert set(_adam_steps(recorder.optimizers[0])) == {3}
    assert set(_adam_steps(recorder.optimizers[1])) == {3}


def test_armed_full_ring_matches_stall_reach_weights():
    import torch
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    torch.set_num_threads(1)

    def run(mode_ttur):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with pr84_reach_candidate(task="mode_hold", ramp="stall", mode_ttur=mode_ttur) as (recorder, _source):
            if mode_ttur:
                recorder._ring_coverage = lambda local: (8, 1.)
            mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy)
        return recorder

    plain, held = run(False), run(True)
    assert held._armed_at == 0 and held._fire_count == 0
    for old, new in zip(_flat_state(plain), _flat_state(held), strict=True):
        assert torch.equal(old, new)

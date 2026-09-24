"""DD-exit shrink applies after the curvature bound and cannot loosen it."""

import torch

from reports.toy100.pr84_dd_exit_bounded import bounded_exit_shrink, pr84_dd_exit_bounded
from reports.toy100.pr84_reach_candidate import pr84_reach_candidate
from reports.toy100.pr84_smoothed_candidate import D_CURVATURE_BOUND, G_CURVATURE_BOUND


def test_shrink_applies_to_a_clipped_step_and_never_exceeds_one():
    shrink, applied = bounded_exit_shrink(-1.0, 1.0)
    assert applied and shrink == 0.3
    assert G_CURVATURE_BOUND * shrink < G_CURVATURE_BOUND

    shrink, applied = bounded_exit_shrink(0.2, 1.0)
    assert not applied and shrink == 1.0

    shrink, applied = bounded_exit_shrink(-1e-4, 1.0)
    assert applied and 0.3 < shrink < 1.0

    shrink, applied = bounded_exit_shrink(-1.0, 0.0)
    assert not applied and shrink == 1.0


def test_floor_on_a_strong_exit():
    shrink, applied = bounded_exit_shrink(-100.0, 1.0, floor=0.3)
    assert applied and shrink == 0.3


def test_disabled_projection_matches_stall_reach_factors():
    torch.set_num_threads(1)
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    def run(factory):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with factory as (recorder, _):
            mode_hold.train_mode_hold(
                mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy, diagnostics=False)
        return [(row["g"]["rho"], row["g"]["factor"]) for row in recorder.records]

    stall = run(pr84_reach_candidate(ramp="stall"))
    disabled = run(pr84_dd_exit_bounded(projection=False))
    assert stall == disabled


def test_enabled_step_does_not_loosen_the_bound():
    torch.set_num_threads(1)
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    policy = NoisePolicy(.029, .5, .1, 1200)
    with pr84_dd_exit_bounded(projection=True) as (recorder, _):
        mode_hold.train_mode_hold(
            mode_hold.ModeHoldRecipe(steps=2), noise_policy=policy, diagnostics=False)
    assert recorder.curvature_bound == G_CURVATURE_BOUND == .25
    assert recorder.d_curvature_bound == D_CURVATURE_BOUND == 3.
    assert recorder.ramp == "stall"
    assert recorder.game_bound is False
    assert recorder.records
    for row in recorder.records:
        g = row["g"]
        expect = min(1., G_CURVATURE_BOUND / g["rho"]) if g["rho"] else 1.
        assert g["curv_factor"] == expect
        assert g["factor"] <= g["curv_factor"] + 1e-12
        assert abs(g["factor"] - g["curv_factor"] * g["exit_shrink"]) < 1e-12

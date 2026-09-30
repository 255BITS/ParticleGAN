"""Mode-exit projection: math, and disabled path matches PR84 factors."""

import torch

from reports.toy100.pr84_mode_exit_projection import (
    mode_exit_factor, pr84_mode_exit_projection,
)
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


def test_projection_only_when_mean_dd_negative_and_pr84_accepts():
    factor, applied = mode_exit_factor(1.0, -0.5, 1.0, True)
    assert applied and factor < 1.0

    factor, applied = mode_exit_factor(1.0, 0.1, 1.0, False)
    assert not applied and factor == 1.0

    factor, applied = mode_exit_factor(0.5, -0.5, 1.0, True)
    assert not applied and factor == 0.5

    factor, applied = mode_exit_factor(1.0, 0.1, 1.0, True)
    assert not applied and factor == 1.0


def test_floor_respected():
    factor, applied = mode_exit_factor(1.0, -100.0, 1.0, True, floor=0.3)
    assert applied and factor == 0.3


def test_disabled_projection_matches_pr84_step_factors():
    torch.set_num_threads(1)
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    def run(factory):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with factory as (recorder, _):
            mode_hold.train_mode_hold(
                mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy, diagnostics=False)
        return [(row["g"]["rho"], row["g"]["factor"]) for row in recorder.records]

    pr84 = run(pr84_smoothed_candidate())
    disabled = run(pr84_mode_exit_projection(projection=False))
    assert pr84 == disabled

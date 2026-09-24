"""Focused integration checks for the uninterrupted mode-hold screen."""

import json

from benchmarks.toy100.continuous_probe import (
    DEFAULT_CONFIG, _window, run_probe,
)


def _simple():
    return json.loads(DEFAULT_CONFIG.read_text())


def test_frozen_mode_hold_separates_scheduled_and_constant_rates():
    scheduled = run_probe(_simple(), mode="scheduled")
    constant = run_probe(_simple(), mode="constant")

    assert scheduled["status"] == "PASS"
    assert scheduled["stationary"]["checks"] == 5
    assert scheduled["stationary"]["pass_all"]
    assert constant["status"] == "FAIL"
    assert constant["stationary"]["failing_steps"]
    assert constant["rate_ranges"] == {
        "g": {"min": .00425, "max": .00425, "observations": 1200},
        "d": {"min": .00425, "max": .00425, "observations": 1200},
        "prior": {"min": .0085, "max": .0085, "observations": 1200},
    }
    assert scheduled["noise"]["input_nonzero_steps"] == 120
    assert constant["noise"]["input_nonzero_steps"] == 120


def test_shifted_frozen_control_uses_same_target_sample_and_no_updates():
    result = run_probe(_simple(), mode="scheduled", steps=1300,
                       shift_step=1200, freeze_after_shift=True)
    pair = result["shift_pair"]
    assert pair["before"]["modes"] == 8
    assert pair["before"]["hq"] >= .9
    assert pair["after"]["modes"] == 0
    assert pair["after"]["hq"] == 0
    assert all(row["calls"] == 1300 and row["updates"] == 1200
               for row in result["optimizer_final"])
    assert result["noise"]["horizon"] == 1200
    assert result["status"] == "FAIL"


def test_final_recovery_does_not_erase_intermediate_collapse():
    points = [dict(step=step, modes=modes, hq=hq) for step, modes, hq in
              ((10, 8, 1.), (20, 0, 0.), (30, 8, .96), (40, 8, .98),
               (50, 8, 1.))]
    window = _window(points)
    assert window["passing_suffix"] == 3
    assert window["failing_steps"] == [20]
    assert not window["pass_all"]
    assert not window["pass_suffix"]

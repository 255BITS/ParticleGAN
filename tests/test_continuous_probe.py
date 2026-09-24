"""Focused integration checks for the uninterrupted mode-hold screen."""

import json
from copy import deepcopy

import pytest

from benchmarks.toy100.continuous_probe import (
    DEFAULT_CONFIG, ROOT, _provenance, _window, archive_executable_sources,
    match_frozen_control, run_probe,
)


def _simple():
    return json.loads(DEFAULT_CONFIG.read_text())


def test_archive_binds_noise_source_omitted_by_standard_snapshot(tmp_path):
    name = "benchmarks/toy100/models.py"
    hashes = _provenance()["source_sha256"]
    archive = archive_executable_sources(tmp_path, hashes)
    assert name not in archive["source_sha256"]
    supplemental = archive["supplemental_sources"][name]
    assert supplemental["sha256"] == hashes[name]
    assert (tmp_path / "toy100-models-source.py").read_bytes() == (ROOT / name).read_bytes()


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
    assert result["status"] == "INCOMPLETE"
    assert result["continued_hold"]["checks"] == 0
    assert not result["shift_recovery"]["deadline_assessable"]


def test_shift_before_stationary_window_is_rejected():
    with pytest.raises(ValueError, match="at or after 1,200"):
        run_probe(_simple(), steps=2400, shift_step=1000)


def test_final_recovery_does_not_erase_intermediate_collapse():
    points = [dict(step=step, modes=modes, hq=hq) for step, modes, hq in
              ((10, 8, 1.), (20, 0, 0.), (30, 8, .96), (40, 8, .98),
               (50, 8, 1.))]
    window = _window(points)
    assert window["passing_suffix"] == 3
    assert window["failing_steps"] == [20]
    assert not window["pass_all"]
    assert not window["pass_suffix"]


def test_adaptation_requires_matching_frozen_sensitivity_control():
    prior = dict(step=2400, modes=8, hq=1.)
    active = dict(mode="constant", config_sha256="same", source_sha256={"source": "same"},
                  runtime={"torch": "same"}, steps=3600, noise_horizon=1200,
                  diagnostic_every=10, shift_step=2400, shift=[.35, 0.],
                  freeze_after_shift=False,
                  diagnostic=[prior], stationary=dict(pass_all=True),
                  continued_hold=dict(pass_all=True), shift_pair=dict(before=prior),
                  shift_recovery=dict(deadline_pass=True), status="UNCONFIRMED")
    frozen = deepcopy(active)
    frozen["freeze_after_shift"] = True
    frozen["shift_recovery"] = dict(deadline_window=dict(checks=81, passing_checks=0))
    frozen["optimizer_final"] = [dict(updates=2400), dict(updates=2400)]
    assert match_frozen_control(active, frozen)["status"] == "PASS"
    frozen["config_sha256"] = "different"
    with pytest.raises(ValueError, match="config_sha256"):
        match_frozen_control(active, frozen)

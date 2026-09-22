"""Ranking must reflect measurements and reject incomplete/invalid evidence."""

from copy import deepcopy
import math
import json
import sys

import pytest

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.baseline import Candidate, METRICS, SHARED, run_toy, score_metrics, score_row
from benchmarks.locked_shared import trajectory


def passing_row():
    row = {"config": {"name": "arbitrary_approach"}, "toys": {}}
    for name, requirements in METRICS.items():
        values = {}
        for key, op, threshold in requirements:
            if key not in values:
                values[key] = threshold
        row["toys"][name] = {"live": values}
    return row


def shared_pass():
    return {name: {"status": "PASS"} for name in SHARED}


def test_all_metrics_required_and_config_name_does_not_award_pass():
    row = passing_row()
    result = score_row(row, shared_pass())
    assert result["status"] == "PASS"
    assert result["passed_metrics"] == result["total_metrics"] == 29
    for toy, requirements in METRICS.items():
        for name, op, bound in requirements:
            broken = deepcopy(row)
            broken["toys"][toy]["live"][name] = bound - 0.001 if op == ">=" else bound + 0.001
            assert score_row(broken, shared_pass())["status"] == "FAIL"


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), True, "1.0"])
def test_non_numeric_and_nonfinite_evidence_never_passes(invalid):
    row = passing_row()
    row["toys"]["mode_hold"]["live"]["hq"] = invalid
    assert score_row(row, shared_pass())["status"] == "INCOMPLETE"


def test_missing_error_shared_failure_and_ema_cannot_rescue_live():
    row = passing_row()
    row["toys"]["mode_hold"]["ema"] = {"modes": 8, "hq": 1.0}
    row["toys"]["mode_hold"]["live"]["hq"] = 0.74
    assert score_row(row, shared_pass())["status"] == "FAIL"
    row["toys"]["mode_hold"]["error"] = "training crashed"
    assert score_row(row, shared_pass())["status"] == "INCOMPLETE"
    row = passing_row()
    del row["toys"]["trajectory"]
    assert score_row(row, shared_pass())["status"] == "INCOMPLETE"
    assert score_row(passing_row(), {})["status"] == "INCOMPLETE"
    failed_shared = shared_pass()
    failed_shared["orbit_hold"]["status"] = "FAIL"
    assert score_row(passing_row(), failed_shared)["status"] == "FAIL"


def test_candidate_settings_reach_training_and_restore_host(monkeypatch):
    before = dict(trajectory.PROTOCOL)
    seen = {}
    def train(**kwargs):
        seen.update(trajectory.PROTOCOL)
        seen["gan"] = kwargs["gan_factory"]().mode
        seen["penalty"] = kwargs["cap_factory"]().arm
        return {"identity_mse": 0.0}
    monkeypatch.setattr(trajectory, "train", train)
    run_toy("trajectory", Candidate("alternative", gan_mode="ra", reg_arm="a_r1r2", reg_coeff=0.1,
                                     particle_l2=0, cover_weight=1, vicreg_weight=0.2, lr_multiplier=0.5))
    assert seen["particle_l2"] == 0 and seen["cover_weight"] == 1 and seen["vicreg_weight"] == 0.2
    assert seen["lr"] == before["lr"] * 0.5
    assert seen["gan"] == "ra" and seen["penalty"] == "a_r1r2"
    assert trajectory.PROTOCOL == before


def test_measured_alternative_is_allowed_and_cloud_still_moves():
    result = run_toy("two_pole", Candidate("no_l2", particle_l2=0))
    cells = score_metrics(result["live"], METRICS["two_pole"])
    assert all(c["status"] == "PASS" for c in cells)
    assert math.isclose(result["live"]["mean_abs"], 0.52926749, rel_tol=1e-5)


def test_unknown_and_nonfinite_settings_are_rejected():
    with pytest.raises(TypeError):
        Candidate("bad", steps=10000)
    with pytest.raises(ValueError):
        Candidate("bad", particle_l2=float("nan"))


def test_resume_rejects_changed_source_and_candidate_before_training(tmp_path, monkeypatch):
    config_file = tmp_path / "input.json"
    config_file.write_text(json.dumps([{"name": "candidate"}]))
    argv = ["baseline", "--configs", str(config_file), "--output", str(tmp_path / "output")]
    monkeypatch.setattr(sys, "argv", argv)
    fingerprint = {"version": "test", "source_sha256": "original"}
    monkeypatch.setattr(baseline, "protocol", lambda: fingerprint.copy())
    trained = []
    def fake_train(toy, cfg):
        trained.append(toy)
        return passing_row()["toys"][toy]
    monkeypatch.setattr(baseline, "run_toy", fake_train)
    assert baseline.main() == 1  # Shared application evidence is missing.
    assert len(trained) == len(METRICS)
    trained.clear()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    fingerprint["source_sha256"] = "changed"
    with pytest.raises(SystemExit) as error:
        baseline.main()
    assert error.value.code == 2 and not trained
    fingerprint["source_sha256"] = "original"
    config_file.write_text(json.dumps([{"name": "candidate", "particle_l2": 0}]))
    with pytest.raises(SystemExit) as error:
        baseline.main()
    assert error.value.code == 2 and not trained

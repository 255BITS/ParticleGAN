"""Bounded source/clock/functional checks; no 50-update recovery run."""

import json
from pathlib import Path

import pytest
import torch

from benchmarks.toy100.continuous_probe import prepared_config
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100.pr84_critic_refinement_capture import _sha
from reports.toy100.pr84_model_error_recovery import (
    BIAS, RESPONSE_UPDATES, frozen_perturbed_points, grade_response,
    require_qualified_cold,
    run_continuation,
)
from tests.test_pr84_critic_refinement_resume import run


def _recipe_noise():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/toy100/constraints_simple_regularization.json").read_text())
    recipe, noise, _ = declared_recipe(prepared_config(config, "constant"))
    return recipe, noise


def test_one_update_unperturbed_helper_is_exact_frozen_resumer_state_and_metrics():
    recipe, noise = _recipe_noise()
    saved = run(2, split=1)["state"]
    old = run(2, saved=saved)
    new = run_continuation(saved, recipe, noise, completed_steps=1, target_steps=2)
    assert _sha(new["state"]) == _sha(old["state"])
    assert new["receipt"]["checkpoints"] == old["points"]
    assert new["receipt"]["actual_adam_updates"] == {"d": 1, "g": 1}
    assert new["receipt"]["optimizer_callbacks"] == {"d": 3, "g": 3}
    assert new["state"]["noise_policy"]["_step_calls"] == 2


def test_fixed_bias_moves_only_live_and_ema_output_and_frozen_model_stays_frozen():
    recipe, noise = _recipe_noise()
    saved = run(2, split=1)["state"]
    result = run_continuation(saved, recipe, noise, completed_steps=1,
                              target_steps=2, perturb=True)
    row = result["model_error"]
    assert row["bias"] == BIAS
    assert row["actual_clean_shift_rms"] == pytest.approx(.35, abs=1e-6)
    assert row["actual_clean_shift_max"] == pytest.approx(.35, abs=1e-6)
    assert _sha(result["perturbed_start"]["critic"]) == _sha(saved["critic"])
    assert _sha(result["perturbed_start"]["prior"]) == _sha(saved["prior"])
    assert _sha(result["perturbed_start"]["optimizer_d"]) == _sha(saved["optimizer_d"])
    assert _sha(result["perturbed_start"]["optimizer_g"]) == _sha(saved["optimizer_g"])
    assert _sha(result["perturbed_start"]["rng"]) == _sha(saved["rng"])
    assert _sha(result["perturbed_start"]["noise_policy"]) == _sha(saved["noise_policy"])
    frozen = frozen_perturbed_points(result["perturbed_start"], noise,
                                     completed_steps=1, target_steps=2)
    assert frozen["noise_step_calls"] == 2 and len(frozen["points"]) == 1
    assert frozen["points"][0]["step"] == 2
    assert frozen["effective_step_trace"] == result["state"]["noise_policy"]["_effective_step_trace"][1:]
    assert result["receipt"]["actual_adam_updates"] == {"d": 1, "g": 1}


def test_local_response_grade_uses_original_thresholds_without_movement_floor():
    good = lambda step: dict(step=step, modes=8, hq=.91)
    bad = lambda step: dict(step=step, modes=7, hq=.89)
    steps = range(1201, 1201 + RESPONSE_UPDATES)
    control = {"receipt": {"checkpoints": [good(step) for step in steps]}}
    perturbed = {"receipt": {"checkpoints": [bad(step) if step < 1246 else good(step)
                                                  for step in steps]}}
    frozen = {"points": [bad(step) for step in steps]}
    passed = grade_response(control, perturbed, frozen)
    assert passed["status"] == "PASS_LOCAL_RESPONSE"
    assert passed["no_signal_movement_floor"] is False
    perturbed["receipt"]["checkpoints"][-1] = bad(1250)
    failed = grade_response(control, perturbed, frozen)
    assert failed["status"] == "FAIL_LOCAL_RESPONSE_FILTER"
    assert "not irrecoverability" in failed["interpretation"]
    frozen["points"][0]["step"] = 999
    with pytest.raises(ValueError, match="different absolute checkpoint clocks"):
        grade_response(control, perturbed, frozen)


def test_failed_cold_ring_is_a_hard_gate_before_any_response_training(tmp_path):
    from reports.toy100.pr84_critic_refinement_finite import METHOD
    rows = {
        "declaration.json": {"method": METHOD, "source": {}},
        "summary.json": {"method": METHOD, "status": "FAIL", "stages": []},
        "mode_hold.json": {"verdict": {"status": "FAIL"}},
        "trajectory.json": {"verdict": {"status": "PASS"}},
    }
    for name, value in rows.items():
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="cold acquisition did not pass"):
        require_qualified_cold(tmp_path)

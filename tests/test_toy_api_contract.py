"""Fail closed at the API/metric/media boundary, independently of model quality."""
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.toy_audit.api_contract import (
    coverage, evaluation_steps, validate_case, validate_observation,
)


def observation():
    return {"metrics": {"rmse": .01}, "passed": True, "failed_bounds": [],
            "views": [{"kind": "scatter", "title": "Target and generated modes",
                       "target": np.zeros((4, 2)), "samples": np.ones((4, 2))}]}


def test_nonfinite_metric_rejects_even_claimed_success():
    value = observation()
    value["metrics"]["rmse"] = float("nan")
    result = validate_observation(value)
    assert result["passed"] is False
    assert result["failed_bounds"] == ["rmse: nonfinite"]


@pytest.mark.parametrize("field,value", [("passed", "PASS"), ("metrics", {}),
                                         ("views", []), ("failed_bounds", "rmse")])
def test_incomplete_observation_is_not_qualification(field, value):
    result = observation()
    result[field] = value
    with pytest.raises(ValueError):
        validate_observation(result)


def test_failed_observation_must_explain_which_bound_failed():
    result = observation()
    result["passed"] = False
    with pytest.raises(ValueError, match="rejected metric"):
        validate_observation(result)


def test_goal_media_requires_both_reference_and_real_output():
    result = observation()
    result["views"][0]["samples"] = np.empty((0, 2))
    with pytest.raises(ValueError, match="finite actual"):
        validate_observation(result)


def test_nonfinite_model_output_is_a_metric_failure_with_an_actual_view():
    result = observation()
    result["views"][0]["samples"][0, 1] = float("nan")
    result = validate_observation(result)
    assert result["passed"] is False
    assert result["metrics"]["nonfinite_output_values"] == 1
    assert "nonfinite output" in result["failed_bounds"][0]


def test_coverage_does_not_drop_old_questions_or_architecture_controls():
    cases = {"new-query": {"id": "new-query", "legacy_ids": ["old-a"]},
             "architecture-control": {"id": "architecture-control", "legacy_ids": ["old-a"]}}
    result = coverage(cases, ["old-a", "old-b"])
    assert result["missing"] == ["old-b"]
    assert result["mapping"]["old-a"] == ["architecture-control", "new-query"]


def test_evaluation_boundaries_include_initial_and_terminal_states_once():
    assert evaluation_steps(3, 9) == [0, 1, 2, 3]
    assert evaluation_steps(16, 5) == [0, 4, 8, 12, 16]


def test_source_only_description_cannot_register_as_runnable_case():
    case = {"id": "new-query", "title": "A query", "goal": "Recover both modes",
            "kind": "vector", "scope": "Public API variant", "legacy_ids": ["old-a"],
            "default_steps": 16, "batch_size": 32, "eval_samples": 256,
            "thresholds": {"coverage_min": 2}, "sampling": "Public served samples"}
    assert validate_case(case) == case
    incomplete = deepcopy(case)
    del incomplete["thresholds"]
    with pytest.raises(ValueError, match="pass/fail bounds"):
        validate_case(incomplete)

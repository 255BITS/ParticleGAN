"""Public-API toy contract, exercised through ``benchmarks.toy_audit.api_contract``.

Discovery runs every provider's real case list through ``validate_case``;
the defect cases below feed one malformed field at a time into the
production validators and require the specific refusal. A passing
observation is the positive control: a NaN metric or nonfinite output must
turn it into a FAIL with a named bound, never a silent PASS.
"""
from __future__ import annotations

from copy import deepcopy
import sys

import numpy as np
import pytest

from benchmarks.toy_audit.api_contract import (
    PROVIDERS, discover, evaluation_steps, metric_observations, validate_case, validate_observation,
)
from benchmarks.toy_audit.reproducibility import DEFAULT_SEED

CASE = {
    "id": "e2e_probe", "title": "Probe", "goal": "Reject malformed declarations", "kind": "vector",
    "scope": "e2e", "legacy_ids": ["legacy_probe"], "default_steps": 10, "batch_size": 32,
    "eval_samples": 64, "terminal_observations": 5, "thresholds": {"rmse": 0.01}, "sampling": "clean live",
}


def _observation(**changes):
    observation = {
        "metrics": {"rmse": 0.005},
        "passed": True,
        "failed_bounds": [],
        "views": [{"kind": "scatter", "title": "target and output", "target": np.zeros((4, 2)),
                   "samples": np.zeros((4, 2))}],
    }
    observation.update(changes)
    return observation


@pytest.fixture(scope="module")
def cases():
    return discover()


def test_discovery_covers_every_provider_with_the_protocol_contract(cases):
    assert {case["provider"] for case in cases.values()} == set(PROVIDERS)
    for case_id, case in cases.items():
        assert validate_case(dict(case)) is not None
        assert case["protocol_seed"] == DEFAULT_SEED == 0, case_id
        assert 1 <= case["evaluation_observations"] <= case["default_steps"], case_id
        assert case.get("terminal_observations", 5) <= case["evaluation_observations"], case_id


def test_declared_case_passes_validation():
    assert validate_case(dict(CASE))["id"] == "e2e_probe"
    assert metric_observations(CASE) == 10


@pytest.mark.parametrize("changes,match", [
    ({"id": "bad id!"}, "invalid toy case ID"),
    ({"goal": "  "}, "missing goal"),
    ({"legacy_ids": []}, "explicit historical/question mapping required"),
    ({"legacy_ids": ["a", "a"]}, "repeated historical mapping"),
    ({"default_steps": 0}, "default_steps must be a positive integer"),
    ({"batch_size": 2.0}, "batch_size must be a positive integer"),
    ({"terminal_observations": 11}, "terminal observations exceed the execution budget"),
    ({"evaluation_observations": 20}, "metric observations exceed distinct execution updates"),
    ({"evaluation_observations": 3}, "terminal observations exceed the declared metric cadence"),
    ({"thresholds": {}}, "frozen pass/fail bounds required"),
    ({"sampling": ""}, "sampling law required"),
])
def test_rejects_malformed_case(changes, match):
    case = {**deepcopy(CASE), **changes}
    with pytest.raises(ValueError, match=match):
        validate_case(case)


def test_passing_observation_is_kept():
    result = validate_observation(_observation())
    assert result["passed"] is True and result["failed_bounds"] == []


def test_nonfinite_metric_or_output_turns_a_claimed_pass_into_a_named_fail():
    result = validate_observation(_observation(metrics={"rmse": float("nan")}))
    assert result["passed"] is False and result["failed_bounds"] == ["rmse: nonfinite"]
    samples = np.zeros((4, 2))
    samples[0, 0] = np.inf
    result = validate_observation(_observation(views=[{"kind": "scatter", "title": "out",
                                                       "target": np.zeros((4, 2)), "samples": samples}]))
    assert result["passed"] is False
    assert result["failed_bounds"] == ["out: 1 nonfinite output values"]
    assert result["metrics"]["nonfinite_output_values"] == 1


@pytest.mark.parametrize("changes,match", [
    ({"passed": 1}, "binary passed value"),
    ({"metrics": {}}, "numeric pass/fail metrics required"),
    ({"metrics": {"rmse": "low"}}, "metric rmse must be scalar"),
    ({"failed_bounds": [3]}, "failed_bounds must identify rejected metric bounds"),
    ({"passed": False}, "FAIL must identify at least one rejected metric bound"),
    ({"views": []}, "goal-illustrating target/output views required"),
    ({"views": [{"kind": "scatter", "title": "x", "target": np.zeros((4, 2)), "samples": np.empty((0, 2))}]},
     "goal view samples must contain finite actual values"),
    ({"views": [{"kind": "scatter", "title": "x", "target": np.full((4, 2), np.nan), "samples": np.zeros((4, 2))}]},
     "goal reference must be finite"),
    ({"views": [{"kind": "histogram", "title": "x", "target": np.zeros(4), "samples": np.zeros(4)}]},
     "unsupported goal view"),
    ({"views": [{"kind": "text", "title": "x", "target": np.zeros(4), "samples": np.zeros(4)}]},
     "text goal views require decoded actual/reference labels"),
])
def test_rejects_malformed_observation(changes, match):
    with pytest.raises(ValueError, match=match):
        validate_observation(_observation(**changes))


def test_evaluation_steps_include_start_end_and_frames():
    assert evaluation_steps(16, frames=5) == [0, 4, 8, 12, 16]
    assert evaluation_steps(3, frames=9) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        evaluation_steps(0)


def test_api_run_cli_exposes_case_and_output(run_cli):
    res = run_cli([sys.executable, "-m", "benchmarks.toy_audit.api_run", "--help"])
    assert res.returncode == 0, res.stderr
    assert "--case" in res.stdout and "--output" in res.stdout

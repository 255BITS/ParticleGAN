"""Opt-in budget diagnostics at the original transfer observation spacing.

These read-only grades describe one uninterrupted 10x execution. Prefixes
cannot substitute for the original task's source-bound qualification receipt.
The ordinary 24-check transfer evaluator is unchanged.
"""
from __future__ import annotations

import math

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.observation import sustained


KIND = "transfer_budget_diagnostic"
EVALUATOR = "experiments.forge.budget_diagnostics:test_verdict"
BASE_OBSERVATIONS = 24
FACTOR = 10
OBSERVATIONS = BASE_OBSERVATIONS * FACTOR
MINIMUM_STABLE_CHECKS = 5
PREFIX_FACTORS = (1, 2, 4, 10)


def validate_declaration(task):
    """Reject unsupported or ambiguous cadence settings before execution."""
    evaluation = task["evaluation"]
    fixed = {"kind": KIND, "evaluator": EVALUATOR, "factor": FACTOR,
             "observations": OBSERVATIONS, "minimum_stable_checks": MINIMUM_STABLE_CHECKS,
             "scoring_weights": "live"}
    for key, expected in fixed.items():
        actual = evaluation.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{task.get('id', '<task>')}: evaluation.{key} must be {expected!r}")
    base = evaluation.get("base_steps")
    if type(base) is not int or base < BASE_OBSERVATIONS:
        raise ValueError("evaluation.base_steps must be an integer >= 24")
    execution = task["execution"]
    if type(execution.get("steps")) is not int or execution["steps"] != base * FACTOR:
        raise ValueError("execution.steps must equal base_steps * factor")
    if (type(execution.get("original_schedule_horizon")) is not int
            or execution["original_schedule_horizon"] != base):
        raise ValueError("execution.original_schedule_horizon must equal base_steps")
    thresholds = evaluation.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("budget diagnostic requires explicit metric thresholds")
    for requirement in thresholds:
        if (not isinstance(requirement, (list, tuple)) or len(requirement) != 3
                or not isinstance(requirement[0], str) or not requirement[0]
                or requirement[1] not in (">=", "<=", "==")
                or type(requirement[2]) not in (int, float)
                or not math.isfinite(requirement[2])):
            raise ValueError("budget diagnostic metric thresholds must be finite named bounds")
    # If callers repeat the fixed prefix list, it must describe this evaluator.
    if "prefixes" in evaluation and evaluation["prefixes"] != list(PREFIX_FACTORS):
        raise ValueError("budget diagnostic prefixes are fixed at 1x, 2x, 4x and 10x")
    return task


def checkpoint_steps(task):
    """ceil(i * base_steps / 24), i=1..240, without floating-point rounding."""
    validate_declaration(task)
    base = task["evaluation"]["base_steps"]
    return [(i * base + BASE_OBSERVATIONS - 1) // BASE_OBSERVATIONS
            for i in range(1, OBSERVATIONS + 1)]


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _prefix_verdict(points, live, requirements, expected, budget):
    convergence = sustained(points, requirements, expected_steps=expected,
                            minimum=MINIMUM_STABLE_CHECKS)
    cells = baseline.score_metrics(live, requirements)
    passed = convergence["confirmed_step"] is not None and all(cell["status"] == "PASS" for cell in cells)
    status = "PASS" if passed else "FAIL" if convergence["complete"] else "INCOMPLETE"
    deficits = [2. if cell["margin"] is None else
                min(2., max(0., -cell["margin"]) / (abs(cell["threshold"]) or 1.)) for cell in cells]
    return {"status": status, "attempted": True, "passed": passed, "metrics": cells,
            "convergence": convergence, "step_budget": budget,
            "shortfall": sum(deficits) / len(deficits) if convergence["complete"] else 2.,
            "confirmation_fraction": convergence["confirmed_step"] / budget if passed else 2.}


def test_verdict(task, evidence):
    """Recompute all bounds and terminal-five stability, with fixed prefixes.

    A structurally invalid or nonfinite curve raises ValueError. Missing checks
    return INCOMPLETE; no recorded status, favourable window or endpoint may
    replace the complete declared observation sequence.
    """
    expected = checkpoint_steps(task)
    requirements = task["evaluation"]["thresholds"]
    points = evidence.get("observations", evidence.get("curve"))
    if points is None or points == []:
        return {"status": "INCOMPLETE", "attempted": True, "passed": False,
                "reason": "all 240 declared observation checkpoints are required",
                "shortfall": 2., "confirmation_fraction": 2.}
    if not isinstance(points, list):
        raise ValueError("budget diagnostic observations must be a list")
    steps = []
    for point in points:
        if (not isinstance(point, dict) or type(point.get("step")) is not int
                or point["step"] <= 0 or not _finite_tree(point)):
            raise ValueError("budget diagnostic observations require finite metrics and integer steps")
        if any(not _finite(point.get(key)) for key, _, _ in requirements):
            raise ValueError("budget diagnostic observation lacks a finite required metric")
        steps.append(point["step"])
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("budget diagnostic steps must be unique and strictly increasing")
    if any(step not in expected for step in steps):
        raise ValueError("budget diagnostic observation cadence differs from the declaration")
    live = evidence.get("live")
    if (not isinstance(live, dict) or not _finite_tree(live)
            or any(not _finite(live.get(key)) for key, _, _ in requirements)):
        return {"status": "INCOMPLETE", "attempted": True, "passed": False,
                "reason": "missing finite final live metrics", "shortfall": 2., "confirmation_fraction": 2.}
    if (any(live[key] != points[-1][key] for key, _, _ in requirements)
            or ("step" in live and live["step"] != points[-1]["step"])):
        raise ValueError("final live metrics must equal the last recorded observation")
    final = _prefix_verdict(points, live, requirements, expected, task["execution"]["steps"])
    snapshots = {}
    base = task["evaluation"]["base_steps"]
    for factor in PREFIX_FACTORS:
        budget = base * factor
        prefix = [point for point in points if point["step"] <= budget]
        prefix_expected = expected[:BASE_OBSERVATIONS * factor]
        # Prefix endpoints are actual recorded observations, never interpolated.
        prefix_live = live if factor == FACTOR else (prefix[-1] if prefix else {})
        snapshots[f"{factor}x"] = _prefix_verdict(prefix, prefix_live, requirements, prefix_expected, budget)
    final["prefix_snapshots"] = snapshots
    final["reason"] = "recomputed complete live budget curve and terminal suffix"
    if steps != expected:
        final["reason"] = "all 240 declared observation checkpoints are required"
    return final

"""Versioned scheduling policy; scientific task compatibility stays unchanged."""
from __future__ import annotations

from copy import deepcopy


VERSION = 1
DEFAULT = {"schema_version": VERSION, "mode": "complete_current_tier"}
LEGACY = {"schema_version": VERSION, "mode": "fail_fast"}


def policy(request: dict) -> dict:
    value = request.get("execution_policy", LEGACY)
    if (not isinstance(value, dict) or set(value) != {"schema_version", "mode"}
            or type(value["schema_version"]) is not int or value["schema_version"] != VERSION
            or not isinstance(value["mode"], str)
            or value["mode"] not in {"complete_current_tier", "fail_fast"}):
        raise ValueError("unsupported Forge execution_policy; expected schema_version=1 and complete_current_tier or fail_fast")
    return deepcopy(value)


def completes_tier(request: dict) -> bool:
    value = policy(request)
    # Registered lanes own their stopping rules, including promotions' frozen
    # fail-fast contract and calibration's explicitly selected diagnostics.
    return (value["mode"] == "complete_current_tier"
            and not request.get("calibration_lane") and not request.get("promotion"))


def group_blockers(request: dict, job: dict) -> list[str]:
    return [f"{member}: {reason}"
            for member in job.get("task_ids", [job["task_id"]])
            for reason in request["tasks"][member].get("preflight_blockers", [])]

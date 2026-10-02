"""Experiment-owned initialization policies, separate from technique settings."""
from __future__ import annotations


INITIALIZERS = frozenset({"deterministic_orthogonal", "supplied"})
MODULE = "experiments/forge/initialization.py"


def task_initializer(task, candidate=None, *, explicit=True):
    """Resolve the task policy and check an explicit candidate constraint.

    Native component policies and fixed host controls remain more specific than
    this declared fallback. ``explicit=False`` is reserved for validation of old
    frozen source requests that predate task-owned initialization; it preserves
    their historical candidate/API fallback without rewriting the receipt.
    """
    candidate = {} if candidate is None else candidate
    execution = task.get("execution", {})
    if "initializer" not in execution:
        if explicit:
            raise ValueError(f"{task.get('id', '<task>')}: execution.initializer must be explicit")
        value = candidate.get("initializer", "deterministic_orthogonal")
    else:
        value = execution["initializer"]
    if not isinstance(value, str) or value not in INITIALIZERS:
        raise ValueError(f"{task.get('id', '<task>')}: execution.initializer must be deterministic_orthogonal or supplied")
    if "initializer" in candidate and candidate["initializer"] != value:
        raise ValueError(f"{task.get('id', '<task>')}: candidate initializer conflicts with experiment-owned execution.initializer")
    return value

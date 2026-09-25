"""Resolve explicit per-problem toy100 recipes into flat runner configs."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

from .problems import PROBLEM_NAMES
from .train import RECIPE_FIELDS, RUN_FIELDS, resolve_config


_MANIFEST_FIELD = "problem_overrides"
_ALLOWED = RECIPE_FIELDS | RUN_FIELDS


def validate_manifest(manifest: Mapping) -> dict:
    """Validate structural scope before any named training run starts.

    The manifest is only a declaration. The runner receives a flat config
    with the selected override applied and the declaration key removed.
    """
    if not isinstance(manifest, Mapping) or not all(isinstance(key, str) for key in manifest):
        raise ValueError("toy100 manifest must be an object with string keys")
    if "problem" in manifest:
        raise ValueError("top-level problem is chosen by the suite, not by the manifest")
    unknown = set(manifest) - _ALLOWED - {_MANIFEST_FIELD}
    if unknown:
        raise ValueError(f"unknown toy100 manifest fields: {', '.join(sorted(unknown))}")
    overrides = manifest.get(_MANIFEST_FIELD, {})
    if not isinstance(overrides, Mapping) or not all(isinstance(key, str) for key in overrides):
        raise ValueError("problem_overrides must map named problems to field objects")
    unknown_problems = set(overrides) - set(PROBLEM_NAMES)
    if unknown_problems:
        raise ValueError(f"unknown problem_overrides keys: {', '.join(sorted(unknown_problems))}")
    for problem, fields in overrides.items():
        if not isinstance(fields, Mapping) or not all(isinstance(key, str) for key in fields):
            raise ValueError(f"problem_overrides.{problem} must be a field object")
        forbidden = set(fields) & {
            "problem", "seed", _MANIFEST_FIELD,
            "toy100_model", "network_lr_horizon_cap", "network_lr_floor",
            "output_noise_rng",
        }
        if forbidden:
            raise ValueError(f"problem_overrides.{problem} cannot set {', '.join(sorted(forbidden))}")
        unknown_fields = set(fields) - _ALLOWED
        if unknown_fields:
            raise ValueError(f"unknown fields in problem_overrides.{problem}: "
                             + ", ".join(sorted(unknown_fields)))
    return deepcopy(dict(manifest))


def resolve_problem_config(manifest: Mapping, problem: str, *,
                           steps: int | None = None, device: str | None = None,
                           validate_runtime: bool = True) -> dict:
    """Return a validated flat config for one named problem.

    Command-line budget/device overrides apply to every problem after the
    declared per-problem fields, so their scope is unambiguous.
    """
    checked = validate_manifest(manifest)
    if problem not in PROBLEM_NAMES:
        raise ValueError(f"unknown toy100 problem: {problem}")
    overrides = checked.pop(_MANIFEST_FIELD, {})
    flat = {**checked, **overrides.get(problem, {}), "problem": problem}
    if steps is not None:
        flat["steps"] = steps
        # A resolved runner config may include total_steps; keep it aligned.
        if "total_steps" in flat:
            flat["total_steps"] = steps
    if device is not None:
        flat["device"] = device
    # Let the same recipe validator as training reject invalid values before
    # creating output directories or starting any of the three runs.
    # Evidence can be regraded on a CPU-only machine after a CUDA run. Check
    # the recipe without demanding that the original execution device exists.
    check = flat if validate_runtime else {**flat, "device": "cpu"}
    resolve_config(check)
    return flat

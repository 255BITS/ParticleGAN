"""Explicit prior declarations, independent of model construction and training."""
from __future__ import annotations

import math


DEFAULT_PRIOR = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
PRIOR_CODE_PATHS = {"mog": "MoGParticlePrior", "particle_cloud": "ParticlePrior"}


def resolve_prior(value=None, *, explicit=False):
    """Resolve API conveniences; task declarations must supply every core field.

    Kind selects the public code path. Sigma never selects or relabels it.
    Ordinary Forge MoG cohorts retain their existing positive-width policy.
    """
    if value is not None and not isinstance(value, dict):
        raise ValueError("prior must be an object")
    given = {} if value is None else dict(value)
    if explicit and DEFAULT_PRIOR.keys() - given.keys():
        raise ValueError(f"prior requires explicit {sorted(DEFAULT_PRIOR.keys() - given.keys())}")
    unknown = set(given) - {"kind", "sigma", "standardize", "learnable", "exception_reason", "init_std"}
    if unknown:
        raise ValueError(f"unsupported prior fields: {sorted(unknown)}")
    prior = {**DEFAULT_PRIOR, **given}
    if not isinstance(prior["kind"], str) or prior["kind"] not in PRIOR_CODE_PATHS:
        raise ValueError("prior kind must be mog or particle_cloud")
    if (type(prior["sigma"]) not in (int, float) or not math.isfinite(prior["sigma"])
            or prior["sigma"] < 0 or type(prior["standardize"]) is not bool
            or type(prior["learnable"]) is not bool):
        raise ValueError("prior sigma/standardize/learnable fields are invalid")
    if prior["kind"] == "particle_cloud":
        if (not {"sigma", "standardize", "exception_reason"}.issubset(given)
                or prior["sigma"] != 0 or prior["standardize"]
                or not isinstance(prior["exception_reason"], str) or not prior["exception_reason"].strip()):
            raise ValueError("particle_cloud requires explicit sigma=0, standardize=False and exception_reason")
    elif prior["sigma"] <= 0:
        raise ValueError("MoG requires nonzero sigma; declare an explicit particle_cloud exception for zero noise")
    if "init_std" in prior and (type(prior["init_std"]) not in (int, float)
                                or not math.isfinite(prior["init_std"]) or prior["init_std"] < 0):
        raise ValueError("prior init_std must be finite and nonnegative")
    return prior


def task_prior(task):
    """Read the experiment-owned prior without candidate or API fallbacks."""
    try:
        return resolve_prior(task["execution"].get("prior"), explicit=True)
    except ValueError as error:
        raise ValueError(f"{task['id']}: execution.{error}") from error

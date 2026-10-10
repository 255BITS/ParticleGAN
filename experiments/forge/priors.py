"""Explicit prior declarations, independent of model construction and training."""
from __future__ import annotations

import math


DEFAULT_PRIOR = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
PRIOR_CODE_PATHS = {"mog": "MoGParticlePrior", "particle_cloud": "ParticlePrior"}
RECIPE_PRIOR_CONTRACT = "recipe_owned_v1"


def recipe_owned_prior(task):
    contract = task.get("execution", {}).get("prior_contract")
    if contract not in (None, RECIPE_PRIOR_CONTRACT):
        raise ValueError(f"unsupported prior_contract: {contract}")
    return contract == RECIPE_PRIOR_CONTRACT


def prior_policy_receipt(recipe, *priors):
    """Record policy and the actual location-table freezing, without draws."""
    return {"contract": RECIPE_PRIOR_CONTRACT, "update": recipe.prior_update,
            "regularizer": recipe.prior_regularizer, "weight": recipe.prior_reg,
            "target_std": recipe.prior_reg_target_std, "eps": recipe.prior_reg_eps,
            "l2": recipe.prior_l2,
            "applicability": "latent_prior" if priors else "not_sampled",
            "trainable_locations": any(prior.z.requires_grad for prior in priors)}


def expected_prior_updates(task, evidence, completed_steps):
    """Only an actual recipe-owned frozen table can have zero prior steps."""
    proof = evidence.get("prior_policy", {})
    if (recipe_owned_prior(task) and proof.get("contract") == RECIPE_PRIOR_CONTRACT
            and task.get("execution", {}).get("prior_applicability") != "not_sampled"
            and proof.get("applicability") == "latent_prior"
            and proof.get("update") == "frozen" and proof.get("trainable_locations") is False):
        return 0
    return completed_steps


def validate_prior_policy(task, evidence, recipe=None):
    """Validate new completed receipts; archived cohorts retain their gates."""
    if (not recipe_owned_prior(task) or task.get("adapter") == "clockfree_audit"
            or task.get("execution", {}).get("prior_applicability") == "not_sampled"):
        return None
    from .boundaries import prior_control_binding
    if not prior_control_binding(task)["latent_table_controls"]:
        return None
    proof = evidence.get("prior_policy")
    if not isinstance(proof, dict):
        return {"status": "INCOMPLETE", "reason": "missing actual recipe-owned prior policy proof"}
    if (proof.get("contract") != RECIPE_PRIOR_CONTRACT or proof.get("applicability") != "latent_prior"
            or proof.get("update") not in {"learned", "frozen"}
            or proof.get("trainable_locations") is not (proof.get("update") == "learned")):
        return {"status": "INVALID", "reason": "prior policy proof contradicts actual trainable locations"}
    if recipe is not None:
        bindings = {"update": ("prior_update", "learned"), "regularizer": ("prior_regularizer", "vicreg"),
                    "weight": ("prior_reg", 0.), "target_std": ("prior_reg_target_std", 1.),
                    "eps": ("prior_reg_eps", 1e-4), "l2": ("prior_l2", 0.)}
        if (not isinstance(recipe, dict)
                or any(proof.get(key) != recipe.get(field, default)
                       for key, (field, default) in bindings.items())):
            return {"status": "INVALID", "reason": "prior policy proof contradicts the actual recipe"}
    return None


def resolve_prior(value=None, *, explicit=False, recipe_owned=False):
    """Resolve API conveniences; task declarations must supply every core field.

    Kind selects the public code path. Sigma never selects or relabels it.
    Ordinary Forge MoG cohorts retain their existing positive-width policy.
    """
    if value is not None and not isinstance(value, dict):
        raise ValueError("prior must be an object")
    given = {} if value is None else dict(value)
    required = DEFAULT_PRIOR.keys() - ({"learnable"} if recipe_owned else set())
    if explicit and required - given.keys():
        raise ValueError(f"prior requires explicit {sorted(required - given.keys())}")
    if recipe_owned and "learnable" in given:
        raise ValueError("recipe-owned initial prior must omit learnable; use Recipe.prior_update")
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
    if recipe_owned:
        prior.pop("learnable")
    return prior


def task_prior(task):
    """Read the experiment-owned prior without candidate or API fallbacks."""
    try:
        return resolve_prior(task["execution"].get("prior"), explicit=True,
                             recipe_owned=recipe_owned_prior(task))
    except ValueError as error:
        raise ValueError(f"{task['id']}: execution.{error}") from error

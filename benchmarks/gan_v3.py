"""Pre-K3P (GAN v3) recipe fields for historical benchmarks and receipts.

The package default is K3P. Benchmarks that replay, regrade or compare
against archived GAN v3 runs resolve their recipes from these explicit
fields instead of the package defaults, so their receipts stay stable.

* ``PRE_K3P_FIELDS`` -- values of every field added with K3P that reproduce
  the pre-K3P trainer (no noise, guard or latent damping; the network LR
  follows the ordinary full-budget schedule). Archived recipe dicts that
  predate these fields resolve with them.
* ``GAN_V3_FIELDS`` -- the full GAN v3 default: pre-K3P fields plus the old
  penalty, optimizer and task-size defaults.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from benchmarks.legacy.recipe import LegacyRecipe as Recipe, get_recipe

PRE_K3P_FIELDS: dict[str, Any] = {
    "network_lr_floor": None,
    "network_lr_horizon_cap": None,
    "reg_anchor_decay": 0.999,
    "d_guard_ratio": 0.0,
    "d_guard_min_steps": 200,
    "latent_damping_max_rate": 0.0,
    "direct_particle_betas": (0.0, 0.9),
    "input_noise_std": 0.0,
    "input_noise_anneal_end": 0.1,
    "output_noise_std": 0.0,
    "output_noise_warmup": 0.2,
}

GAN_V3_FIELDS: dict[str, Any] = {
    **PRE_K3P_FIELDS,
    "z_dim": 4,
    "batch_size": 256,
    "betas": (0.0, 0.99),
    "reg_arm": "b_cap",
    "reg_coeff": 6.0,
    "reg_kappa": 1.25,
    "prior_reg": 0.05,
}


def legacy_recipe(fields: Mapping[str, Any]) -> Recipe:
    """``Recipe`` from an archived (pre-K3P) field dict; missing K3P fields are neutral."""
    return Recipe(**{**PRE_K3P_FIELDS, **fields})


def gan_v3_recipe(name: str = "gan", **overrides) -> Recipe:
    """``get_recipe(name, ...)`` on the GAN v3 defaults (named ``gan_v3`` for the base family)."""
    options = {**GAN_V3_FIELDS, **overrides}
    label = options.pop("name", "gan_v3" if name == "gan" else None)
    recipe = get_recipe(name, **options)
    return recipe if label is None else recipe.replace(name=label)


def _json(value):
    return json.loads(json.dumps(value))


def legacy_dict(recipe: Recipe) -> dict[str, Any]:
    """``recipe.to_dict()`` without K3P-era fields left at their pre-K3P values.

    This is the dict archived pre-K3P receipts recorded; a K3P field set to
    anything else stays visible so a changed run never matches a GAN v3 receipt.
    """
    values = recipe.to_dict()
    neutral = _json(PRE_K3P_FIELDS)
    return {key: value for key, value in values.items()
            if key not in neutral or _json(value) != neutral[key]}

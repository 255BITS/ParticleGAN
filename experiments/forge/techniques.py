"""Mechanism identity for configuration search, separate from family lineage.

Numeric tuning may change a mechanism's strength or timing, but cannot add or
remove it. These projections never replace recorded scientific identities.
"""
from dataclasses import asdict

from particlegan import Recipe

from .contracts import stable_hash


def _recipe(value):
    return value if isinstance(value, Recipe) else Recipe(**value)


def technique_signature(value):
    """Describe fixed public mechanisms and activation boundaries of knobs."""
    from .boundaries import recipe_field_owner

    recipe = _recipe(value)
    resolved = asdict(recipe)
    fixed = {name: item for name, item in resolved.items()
             if name != "name" and recipe_field_owner(name) == "technique"}
    prior_betas = recipe.prior_betas or recipe.betas
    network_floor = recipe.resolved_network_lr_floor
    penalty_end = recipe.reg_coeff if recipe.reg_coeff_end is None else recipe.reg_coeff_end
    beta2_end = recipe.betas[1] if recipe.beta2_end is None else recipe.beta2_end
    mechanisms = {
        "critic_penalty": {"initial": recipe.reg_coeff > 0, "terminal": penalty_end > 0},
        "penalty_schedule": {"declared": recipe.reg_coeff_end is not None,
                             "changing": penalty_end != recipe.reg_coeff},
        "beta2_schedule": {"declared": recipe.beta2_end is not None,
                           "network_changing": beta2_end != recipe.betas[1],
                           "prior_changing": recipe.beta2_end is not None and beta2_end != prior_betas[1]},
        "network_moments": [value > 0 for value in recipe.betas],
        "prior_moments": [value > 0 for value in prior_betas],
        "terminal_second_moment": beta2_end > 0,
        "amsgrad": recipe.amsgrad,
        "prior_regularization": recipe.prior_reg > 0,
        "critic_anchor": recipe.reg_anchor_weight > 0,
        "critic_guard": recipe.d_guard_ratio > 0,
        "latent_damping": recipe.latent_damping_max_rate > 0,
        "training_input_noise": recipe.input_noise_std > 0,
        "training_output_noise": recipe.output_noise_std > 0,
        "output_noise_warmup": recipe.output_noise_std > 0 and recipe.output_noise_warmup > 0,
        "ucd_objective": recipe.ucd_weight > 0 if recipe.conditioning == "ucd" else None,
        "reconstruction_objective": recipe.reconstruction_weight > 0 if recipe.encoder_mode != "none" else None,
        "averaged_serving": recipe.serve_average > 0,
        "gradient_cap": recipe.reg_kappa > 0 if recipe.reg_arm != "a_r1r2" else None,
        "scheduled_learning_rates": recipe.total_steps is not None,
        "network_horizon_cap": recipe.network_lr_horizon_cap is not None,
        "network_terminal_updates": network_floor > 0,
        "prior_terminal_updates": recipe.lr_floor > 0,
        # K3P passes floor=0 when the resolved network floor reaches .5.
        "k3p_penalty_lr_floor": (0 < network_floor < .5
                                 if recipe.effective_critic_formulation == "k3p" and recipe.reg_arm is None
                                 else None),
    }
    contract = {"schema_version": 1, "fixed_recipe_fields": fixed, "mechanisms": mechanisms}
    return {**contract, "digest": stable_hash(contract)}


def validate_same_technique(base, trial):
    """Reject structural changes hidden in otherwise searchable Recipe knobs."""
    before, after = technique_signature(base), technique_signature(trial)
    changed = [name for name in before["mechanisms"]
               if before["mechanisms"][name] != after["mechanisms"][name]]
    changed += [f"Recipe.{name}" for name in before["fixed_recipe_fields"]
                if before["fixed_recipe_fields"][name] != after["fixed_recipe_fields"][name]]
    if changed:
        raise ValueError("search changes technique mechanisms: " + ", ".join(changed)
                         + "; declare a structural idea instead of a hyperparameter trial")
    return after


def recipe_field_active(name, value, *, task=None):
    """Conservative declared activity, without constructing or sampling models."""
    from .boundaries import prior_control_binding

    recipe = _recipe(value)
    execution = {} if task is None else task.get("execution", {})
    prior_active = ((task is None or prior_control_binding(task)["latent_table_controls"])
                    and execution.get("prior", {}).get("learnable", True))
    penalty_end = recipe.reg_coeff if recipe.reg_coeff_end is None else recipe.reg_coeff_end
    if name in {"reg_kappa", "reg_every", "reg_coeff_anneal_end"}:
        if recipe.reg_coeff == 0 and penalty_end == 0:
            return False
    if name == "reg_kappa" and recipe.reg_arm == "a_r1r2":
        return False
    if name == "reg_coeff_anneal_end":
        return recipe.reg_coeff_end is not None and recipe.reg_coeff_end != recipe.reg_coeff
    if name == "beta2_anneal_end":
        return recipe.beta2_end is not None and (recipe.beta2_end != recipe.betas[1]
                or (prior_active and recipe.beta2_end != (recipe.prior_betas or recipe.betas)[1]))
    if name in {"lr_anneal_start", "lr_floor", "network_lr_floor"} and recipe.total_steps is None:
        return False
    if name == "lr_anneal_start" and recipe.lr_floor == recipe.resolved_network_lr_floor == 1:
        return False
    if name == "lr_floor" and task is not None and not prior_active and recipe.network_lr_floor is not None:
        return False
    if task is not None and name in {"prior_lr_mult", "prior_betas", "prior_reg"}:
        if not prior_active:
            return False
    return True

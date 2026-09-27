"""Benchmark-local network LR horizon policy, expressed as a recipe schedule.

The toy100 trainer recipe keeps its full-budget schedule. A policy changes
only the dense generator and discriminator horizon/floor that the recipe-built
optimizers apply; the particle prior always follows the full-budget cosine.
"""

from __future__ import annotations

import math

from particlegan import learning_rate_scale


def policy_multipliers(
    completed_step: int,
    total_steps: int,
    anneal_start: float,
    floor: float,
    network_lr_horizon_cap: int | None = None,
    *,
    network_lr_floor: float | None = None,
) -> tuple[float, float]:
    """Return (G/D multiplier, prior multiplier) for the next update."""
    if type(completed_step) is not int or completed_step < 0:
        raise ValueError("completed_step must be a nonnegative integer")
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if network_lr_horizon_cap is not None and (
        type(network_lr_horizon_cap) is not int or network_lr_horizon_cap <= 0
    ):
        raise ValueError("network_lr_horizon_cap must be a positive integer")
    if network_lr_floor is not None:
        if network_lr_horizon_cap is None:
            raise ValueError("network_lr_floor requires network_lr_horizon_cap")
        if (isinstance(network_lr_floor, bool)
                or not isinstance(network_lr_floor, (int, float))
                or not math.isfinite(network_lr_floor)
                or not 0 <= network_lr_floor <= 1):
            raise ValueError("network_lr_floor must be a finite fraction in [0, 1]")
    horizon = min(total_steps, network_lr_horizon_cap or total_steps)
    network = learning_rate_scale(
        completed_step, horizon, anneal_start,
        floor if network_lr_floor is None else network_lr_floor,
    )
    prior = learning_rate_scale(completed_step, total_steps, anneal_start, floor)
    return network, prior


def policy_recipe(recipe, network_lr_horizon_cap: int | None = None, *,
                  network_lr_floor: float | None = None):
    """The recipe whose own LR schedule is this policy (``policy_multipliers``).

    Only the schedule fields change: G/D over ``min(total, cap)`` to the policy
    floor (default: the recipe's ``lr_floor``), the prior over the full budget.
    """
    cap = network_lr_horizon_cap
    floor = recipe.lr_floor if network_lr_floor is None else float(network_lr_floor)
    return recipe.replace(network_lr_horizon_cap=cap, network_lr_floor=floor)


def step_with_policy(trainer, real, *, network_lr_horizon_cap: int | None = None,
                     network_lr_floor: float | None = None, **step_kwargs):
    """Run one ordinary GANTrainer update with a capped dense-network horizon.

    The trainer's recipe-built optimizers apply their LR schedule inside
    ``step()``; for this update their schedules read ``policy_recipe`` (the
    same update counts, the policy's network horizon and floor), then return
    to the trainer recipe. Nothing writes group rates, the trainer recipe (and
    so the critic penalty) is unchanged, and base rates are never modified,
    including in checkpoints. The schedules are restored even when an update
    raises.
    """
    total = trainer.recipe.total_steps
    if network_lr_horizon_cap is None and network_lr_floor is None:
        return trainer.step(real, **step_kwargs)
    policy_multipliers(  # validate
        trainer.completed_steps, total, trainer.recipe.lr_anneal_start,
        trainer.recipe.lr_floor, network_lr_horizon_cap,
        network_lr_floor=network_lr_floor,
    )
    if (network_lr_horizon_cap >= total
            and (network_lr_floor is None or network_lr_floor == trainer.recipe.lr_floor)):
        return trainer.step(real, **step_kwargs)
    if len(trainer.opt_g.param_groups) != 2 or len(trainer.opt_d.param_groups) != 1:
        raise RuntimeError("expected G, prior, and D optimizer groups")
    policy = policy_recipe(trainer.recipe, network_lr_horizon_cap, network_lr_floor=network_lr_floor)
    schedules = [trainer.opt_g.lr_schedule, trainer.opt_d.lr_schedule]
    originals = [schedule.recipe for schedule in schedules]
    try:
        for schedule in schedules:
            schedule.recipe = policy
        return trainer.step(real, **step_kwargs)
    finally:
        for schedule, original in zip(schedules, originals):
            schedule.recipe = original


def policy_rate_action(trainer, completed_step: int, *,
                       network_lr_horizon_cap: int | None = None,
                       network_lr_floor: float | None = None) -> dict:
    """Record and verify the rates actually left on optimizer groups."""
    if type(completed_step) is not int or completed_step != trainer.completed_steps or completed_step < 1:
        raise ValueError("completed_step must equal the trainer's completed update count")
    network, prior = policy_multipliers(
        completed_step - 1, trainer.recipe.total_steps,
        trainer.recipe.lr_anneal_start, trainer.recipe.lr_floor,
        network_lr_horizon_cap,
        network_lr_floor=network_lr_floor,
    )
    if len(trainer.opt_g.param_groups) != 2 or len(trainer.opt_d.param_groups) != 1:
        raise RuntimeError("expected G, prior, and D optimizer groups")
    rates = {
        "lr_g": trainer.opt_g.param_groups[0]["lr"],
        "lr_prior": trainer.opt_g.param_groups[1]["lr"],
        "lr_d": trainer.opt_d.param_groups[0]["lr"],
    }
    expected = {
        "lr_g": trainer.initial_lrs[0][0] * network,
        "lr_prior": trainer.initial_lrs[0][1] * prior,
        "lr_d": trainer.initial_lrs[1][0] * network,
    }
    for role, actual in rates.items():
        if not math.isclose(actual, expected[role], rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError(f"{role} diverged from declared network LR policy")
    receipt = {**rates, "network_multiplier": network,
               "prior_multiplier": prior,
               "network_lr_horizon_cap": network_lr_horizon_cap}
    if network_lr_floor is not None:
        receipt["network_lr_floor"] = float(network_lr_floor)
    return receipt

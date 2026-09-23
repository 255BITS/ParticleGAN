"""Benchmark-local optimizer policy for a shared network LR horizon.

The public Recipe and GANTrainer keep their original full-budget schedule.
Only the dense generator and discriminator rates are adjusted here; the
particle prior always follows the ordinary full-budget cosine.
"""

from __future__ import annotations

import math

from particlegan.recipes import learning_rate_scale


def policy_multipliers(
    completed_step: int,
    total_steps: int,
    anneal_start: float,
    floor: float,
    network_lr_horizon_cap: int | None = None,
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
    horizon = min(total_steps, network_lr_horizon_cap or total_steps)
    network = learning_rate_scale(completed_step, horizon, anneal_start, floor)
    prior = learning_rate_scale(completed_step, total_steps, anneal_start, floor)
    return network, prior


def step_with_policy(trainer, real, *, network_lr_horizon_cap: int | None = None,
                     **step_kwargs):
    """Run one ordinary GANTrainer update with a capped dense-network horizon.

    GANTrainer sets ordinary full-budget group rates inside ``step``. Local
    optimizer pre-step hooks replace the G and D group rates by their exact
    capped values immediately before the updates. The prior remains on the
    ordinary full-budget rate. Hooks are removed even when an update raises;
    the trainer's base rates are never modified, including in checkpoints.
    """
    total = trainer.recipe.total_steps
    if network_lr_horizon_cap is None or network_lr_horizon_cap >= total:
        if network_lr_horizon_cap is not None:
            policy_multipliers(trainer.completed_steps, total,
                               trainer.recipe.lr_anneal_start, trainer.recipe.lr_floor,
                               network_lr_horizon_cap)
        return trainer.step(real, **step_kwargs)
    network, prior = policy_multipliers(
        trainer.completed_steps, total, trainer.recipe.lr_anneal_start,
        trainer.recipe.lr_floor, network_lr_horizon_cap,
    )
    if len(trainer.opt_g.param_groups) != 2 or len(trainer.opt_d.param_groups) != 1:
        raise RuntimeError("expected G, prior, and D optimizer groups")

    def set_g_rates(optimizer, args, kwargs):
        optimizer.param_groups[0]["lr"] = trainer.initial_lrs[0][0] * network
        optimizer.param_groups[1]["lr"] = trainer.initial_lrs[0][1] * prior

    def set_d_rate(optimizer, args, kwargs):
        optimizer.param_groups[0]["lr"] = trainer.initial_lrs[1][0] * network

    g_hook = trainer.opt_g.register_step_pre_hook(set_g_rates)
    try:
        d_hook = trainer.opt_d.register_step_pre_hook(set_d_rate)
        try:
            return trainer.step(real, **step_kwargs)
        finally:
            d_hook.remove()
    finally:
        g_hook.remove()


def policy_rate_action(trainer, completed_step: int, *,
                       network_lr_horizon_cap: int | None = None) -> dict:
    """Record and verify the rates actually left on optimizer groups."""
    if type(completed_step) is not int or completed_step != trainer.completed_steps or completed_step < 1:
        raise ValueError("completed_step must equal the trainer's completed update count")
    network, prior = policy_multipliers(
        completed_step - 1, trainer.recipe.total_steps,
        trainer.recipe.lr_anneal_start, trainer.recipe.lr_floor,
        network_lr_horizon_cap,
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
    return {**rates, "network_multiplier": network,
            "prior_multiplier": prior,
            "network_lr_horizon_cap": network_lr_horizon_cap}

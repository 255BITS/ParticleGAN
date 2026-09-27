"""Lazy application (``reg_every`` / ``lazy_k``) keeps the default technique.

Scripts that used to pin ``GradientPenalty(arm="b_cap", lazy_k=4)`` now use
``recipe.make_critic_penalty(opt_d)`` with ``reg_every=4``. Laziness must only
change *when* the default penalty is applied and scale its coefficient by k on
the applied steps; the formulation (KA2, its EMA anchor and blend) is unchanged.
"""
import copy

import pytest
import torch
from torch import nn

from particlegan import get_recipe


def _critic(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()


def _batch(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(32, 2, generator=g, dtype=torch.float64),
            torch.randn(32, 2, generator=g, dtype=torch.float64) + 0.5)


def _penalty(recipe, lazy_override=None):
    D = _critic()
    opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D))
    extra = {} if lazy_override is None else {"lazy_k": lazy_override}
    return D, opt, recipe.make_critic_penalty(opt, **extra)


@pytest.mark.parametrize("via", ["recipe", "override"])
def test_lazy_keeps_default_technique_and_only_changes_frequency(via):
    base = get_recipe("gan", total_steps=40, network_lr_horizon_cap=40)
    k = 4
    lazy_recipe = base.replace(reg_every=k) if via == "recipe" else base
    D1, opt1, eager = _penalty(base)
    Dk, optk, lazy = _penalty(lazy_recipe, None if via == "recipe" else k)
    eager_k = base.make_critic_penalty(optk)  # drives the lazy run's updates
    assert lazy.regularizer.lazy_k == k and eager.regularizer.lazy_k == 1
    for key in ("coeff", "kappa", "anchor_weight"):
        assert getattr(lazy.regularizer, key) == getattr(eager.regularizer, key)
    # Same critic state, same batch: the lazy penalty is 0 off-schedule and
    # k times the eager penalty on-schedule (the time-averaged pressure matches).
    for step in range(1, 13):
        real, fake = _batch(step)
        assert optk.record.observed_steps + 1 == step
        pe, pk = eager(D1, real, fake), lazy(Dk, real, fake)
        if step % k:
            assert float(pk) == 0.0 and not pk.requires_grad
        else:
            assert torch.allclose(pk, k * pe, rtol=1e-12, atol=0)
        # Advance both critics identically: each takes the eager penalty's
        # gradient so the two stay parameter-for-parameter equal.
        for D, opt, pen in ((D1, opt1, eager), (Dk, optk, eager_k)):
            opt.zero_grad()
            (D(real).mean() - D(fake).mean() + pen(D, real, fake)).backward()
            opt.step()
        for a, b in zip(D1.parameters(), Dk.parameters()):
            assert torch.equal(a, b)
    # Laziness never stops the step-time state from advancing every step.
    assert optk.record.observed_steps == opt1.record.observed_steps == 12


def test_lazy_blend_counts_applied_calls_independently_of_learning_rate():
    recipe = get_recipe("gan", total_steps=24, network_lr_horizon_cap=24, reg_every=4)
    D = _critic()
    opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D))
    penalty = recipe.make_critic_penalty(opt, collect_stats=True)
    # Start just before KA2's measured warmup boundary. Skipped calls must
    # neither start the anchor nor consume warmup, even at a tiny LR.
    opt.record.calls = 798
    opt.param_groups[0]["lr"] *= .01
    phases = []
    for step in range(1, 13):
        real, fake = _batch(step)
        calls_before = opt.record.calls
        pen = penalty(D, real, fake)
        if step % 4 == 0:
            phases.append(penalty.last_stats["phase"])
            assert opt.record.calls == calls_before + 1
        else:
            assert opt.record.calls == calls_before
        assert opt.record.anchor_started == (step >= 8)
        opt.zero_grad()
        (D(real).mean() - D(fake).mean() + pen).backward()
        opt.step()
    assert phases == ["a", "blend", "blend"]
    assert opt.record.observed_steps == 12 and opt.record.calls == 801

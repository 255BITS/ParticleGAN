"""Recipe ablation switches: defaults are the shipped formulation; off removes the effect."""
import copy

import pytest
import torch
from torch import nn

from particlegan import get_recipe


def _critic(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()


def _batch(step):
    g = torch.Generator().manual_seed(step)
    return (torch.randn(32, 2, generator=g, dtype=torch.float64),
            torch.randn(32, 2, generator=g, dtype=torch.float64) + 0.5)


def _critic_run(recipe, steps=8, ema=True):
    """Train one critic with the recipe's penalty; return penalties and stats."""
    D = _critic()
    opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D) if ema else None, lr=1e-2)
    penalty = recipe.make_critic_penalty(opt, generator=torch.Generator().manual_seed(0), collect_stats=True)
    rows = []
    for step in range(1, steps + 1):
        real, fake = _batch(step)
        pen = penalty(D, real, fake)
        rows.append((pen.detach().clone(), dict(penalty.last_stats)))
        loss = D(real).mean() - D(fake).mean() + pen
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rows, D


def test_switch_defaults_are_the_shipped_formulation():
    recipe = get_recipe()
    assert recipe.reg_anchor_weight == 1.0 and recipe.amsgrad is True
    explicit = recipe.replace(reg_anchor_weight=1.0, amsgrad=True)
    assert explicit == recipe
    rows, D = _critic_run(recipe)
    same, D2 = _critic_run(explicit)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(rows, same))
    assert all(torch.equal(p, q) for p, q in zip(D.parameters(), D2.parameters()))
    # The anchor term is live from the second critic step on (it starts at the first).
    assert rows[0][1]["prox"] == 0.0 and all(stats["prox"] > 0 for _, stats in rows[1:])


def test_anchor_weight_zero_removes_the_anchor_term():
    base = get_recipe()
    rows, _ = _critic_run(base)
    off, _ = _critic_run(base.replace(reg_anchor_weight=0.0))
    # Identical on the first call (the anchor starts with prox exactly 0).
    assert torch.equal(rows[0][0], off[0][0])
    assert all(stats["prox"] == 0.0 for _, stats in off)
    # With the same critic state, the default penalty exceeds weight 0 by exactly coeff/2 * prox.
    prox = rows[1][1]["prox"]
    assert prox > 0 and not torch.equal(rows[1][0], off[1][0])
    # Weight 0 needs no EMA critic, and the weighted term scales linearly.
    no_ema, _ = _critic_run(base.replace(reg_anchor_weight=0.0), ema=False)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(off, no_ema))
    with pytest.raises(ValueError, match="EMA"):
        base.make_critic_penalty(base.make_critic_optimizer(_critic()))
    half, _ = _critic_run(base.replace(reg_anchor_weight=0.5))
    assert half[1][1]["prox"] == pytest.approx(0.5 * prox, rel=1e-12)
    expected = off[1][0] + base.reg_coeff / 2 * prox
    torch.testing.assert_close(rows[1][0], expected, rtol=1e-12, atol=1e-15)


def test_amsgrad_switch_reaches_every_optimizer_group():
    from particlegan.particle_prior import ParticlePrior
    for amsgrad in (False, True):
        recipe = get_recipe(amsgrad=amsgrad)
        prior = ParticlePrior(num_particles=16, z_dim=2)
        opt_g, opt_d = recipe.make_optimizers(nn.Linear(2, 2), _critic(), prior, ema_critic=_critic())
        assert [g["amsgrad"] for opt in (opt_g, opt_d) for g in opt.param_groups] == [amsgrad] * 3
    assert get_recipe().make_critic_optimizer(_critic(), amsgrad=False).param_groups[0]["amsgrad"] is False


@pytest.mark.parametrize("bad", [dict(reg_anchor_weight=-1.0), dict(reg_anchor_weight=float("nan")),
                                 dict(amsgrad=1), dict(amsgrad=None)])
def test_switch_validation(bad):
    with pytest.raises(ValueError):
        get_recipe(**bad)


def test_removed_direct_particle_switches_are_gone():
    with pytest.raises(TypeError):
        get_recipe(direct_particle_gain=True)
    with pytest.raises(TypeError):
        get_recipe().make_generator_optimizer([nn.Parameter(torch.zeros(2))], direct_particles=[])

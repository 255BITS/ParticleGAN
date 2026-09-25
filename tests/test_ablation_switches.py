"""Recipe ablation switches: defaults are the shipped formulation; off removes the effect."""
import copy

import pytest
import torch
from torch import nn

from particlegan import get_recipe, learning_rate_scale


def _critic(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()


def _batch(step):
    g = torch.Generator().manual_seed(step)
    return (torch.randn(32, 2, generator=g, dtype=torch.float64),
            torch.randn(32, 2, generator=g, dtype=torch.float64) + 0.5)


def _critic_run(recipe, steps=14, ema=True):
    """Train one critic through the blended/late penalty phases; return penalties and stats."""
    D = _critic()
    opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D) if ema else None, lr=1e-2)
    penalty = recipe.make_critic_penalty(opt, collect_stats=True)
    rows = []
    for step in range(1, steps + 1):
        for group in opt.param_groups:
            group["lr"] = 1e-2 * learning_rate_scale(step - 1, 10, .3, .01)
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
    assert recipe.reg_anchor_weight == 1.0 and recipe.direct_particle_gain is True
    explicit = recipe.replace(reg_anchor_weight=1.0, direct_particle_gain=True)
    assert explicit == recipe
    rows, D = _critic_run(recipe)
    same, D2 = _critic_run(explicit)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(rows, same))
    assert all(torch.equal(p, q) for p, q in zip(D.parameters(), D2.parameters()))
    # The anchor term is live in the default: some late-phase step has prox > 0.
    assert any(stats["phase"] in ("blend", "b") and stats["prox"] > 0 for _, stats in rows)


def test_anchor_weight_zero_removes_the_anchor_term():
    base = get_recipe()
    rows, _ = _critic_run(base)
    off, _ = _critic_run(base.replace(reg_anchor_weight=0.0))
    late = [i for i, (_, stats) in enumerate(rows) if stats["phase"] in ("blend", "b")]
    assert late, "the run must reach the blended phase"
    first = late[0]
    # Identical until the anchor term first contributes (the step after the anchor starts).
    for i in range(first + 1):
        assert torch.equal(rows[i][0], off[i][0])
    assert all(stats["prox"] == 0.0 for _, stats in off)
    # With the same critic state, the default penalty exceeds weight 0 by exactly
    # coeff/2 * (1 - s) * prox.
    i = first + 1
    s, prox = rows[i][1]["s"], rows[i][1]["prox"]
    assert prox > 0
    assert not torch.equal(rows[i][0], off[i][0])
    # Weight 0 needs no EMA critic, and the weighted term scales linearly.
    no_ema, _ = _critic_run(base.replace(reg_anchor_weight=0.0), ema=False)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(off, no_ema))
    with pytest.raises(ValueError, match="EMA"):
        base.make_critic_penalty(base.make_critic_optimizer(_critic()))
    half, _ = _critic_run(base.replace(reg_anchor_weight=0.5))
    assert half[i][1]["prox"] == pytest.approx(0.5 * prox, rel=1e-12)
    expected = off[i][0] + base.reg_coeff / 2 * (1 - s) * prox
    torch.testing.assert_close(rows[i][0], expected, rtol=1e-12, atol=1e-15)


def _direct_run(recipe, steps=6):
    torch.manual_seed(0)
    particles = nn.Parameter(torch.randn(16, 2))
    opt = recipe.make_generator_optimizer([particles], direct_particles=[particles], lr=1e-2)
    gains, lrs = [], []
    for step in range(steps):
        opt.zero_grad()
        (particles - torch.tensor([1.0, -1.0]) * (step + 1)).square().sum().backward()
        opt.step()
        gains.append(opt.direct_response.last_gain)
        lrs.append(opt.param_groups[0]["lr"])
    return particles.detach().clone(), gains, lrs


def test_direct_particle_gain_false_keeps_the_scheduled_lr():
    recipe = get_recipe()
    on, gains_on, lrs_on = _direct_run(recipe)
    again, _, _ = _direct_run(recipe.replace(direct_particle_gain=True))
    assert torch.equal(on, again)
    assert gains_on[0] == 1.0 and max(gains_on[1:]) > 1.0  # aligned gradients raise the LR
    off, gains_off, lrs_off = _direct_run(recipe.replace(direct_particle_gain=False))
    assert gains_off == [1.0] * len(gains_off)
    assert lrs_on == lrs_off == [1e-2] * len(lrs_on)  # the gain never leaks into the stored LR
    assert not torch.equal(on, off)
    # Without the gain the step is Adam with the direct-particle betas.
    torch.manual_seed(0)
    particles = nn.Parameter(torch.randn(16, 2))
    adam = torch.optim.Adam([particles], lr=1e-2, betas=recipe.direct_particle_betas)
    for step in range(6):
        adam.zero_grad()
        (particles - torch.tensor([1.0, -1.0]) * (step + 1)).square().sum().backward()
        adam.step()
    assert torch.equal(particles.detach(), off)


@pytest.mark.parametrize("bad", [dict(reg_anchor_weight=-1.0), dict(reg_anchor_weight=float("nan")),
                                 dict(direct_particle_gain=1)])
def test_switch_validation(bad):
    with pytest.raises(ValueError):
        get_recipe(**bad)

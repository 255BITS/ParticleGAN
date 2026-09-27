"""Recipe ablation switch: the default is the shipped formulation; off removes the effect."""
import copy

import pytest
import torch
from torch import nn

from particlegan import get_recipe, grad_regularizers


@pytest.fixture(autouse=True)
def short_warmup(monkeypatch):
    """Reach KA2's blended phase after 5 applied calls instead of 799."""
    monkeypatch.setattr(grad_regularizers, "WARMUP_CALLS", 6)


def _critic(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).double()


def _batch(step):
    g = torch.Generator().manual_seed(step)
    return (torch.randn(32, 2, generator=g, dtype=torch.float64),
            torch.randn(32, 2, generator=g, dtype=torch.float64) + 0.5)


def _critic_run(recipe, steps=14, ema=True):
    """Train one critic through phase A into the blend; return penalties and stats."""
    D = _critic()
    opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D) if ema else None, lr=1e-2)
    penalty = recipe.make_critic_penalty(opt, collect_stats=True)
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


def test_switch_default_is_the_shipped_formulation():
    recipe = get_recipe()
    assert recipe.reg_anchor_weight == 1.0
    explicit = recipe.replace(reg_anchor_weight=1.0)
    assert explicit == recipe
    rows, D = _critic_run(recipe)
    same, D2 = _critic_run(explicit)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(rows, same))
    assert all(torch.equal(p, q) for p, q in zip(D.parameters(), D2.parameters()))
    # The anchor term is live in the default: some blended step has prox > 0.
    assert any(stats["phase"] == "blend" and stats["prox"] > 0 for _, stats in rows)


def test_anchor_weight_zero_removes_the_anchor_term():
    base = get_recipe()
    rows, _ = _critic_run(base)
    off, _ = _critic_run(base.replace(reg_anchor_weight=0.0))
    late = [i for i, (_, stats) in enumerate(rows) if stats["phase"] == "blend"]
    assert late, "the run must reach the blended phase"
    first = late[0]
    # Identical until the anchor term first contributes (the call after the anchor starts).
    for i in range(first + 1):
        assert torch.equal(rows[i][0], off[i][0])
    assert all(stats["prox"] == 0.0 for _, stats in off)
    i = first + 1
    prox, weight = rows[i][1]["prox"], rows[i][1]["w"]
    assert prox > 0 and weight == 1.0
    assert not torch.equal(rows[i][0], off[i][0])
    # Weight 0 needs no EMA critic, and the weighted term scales linearly.
    no_ema, _ = _critic_run(base.replace(reg_anchor_weight=0.0), ema=False)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(off, no_ema))
    with pytest.raises(ValueError, match="EMA"):
        base.make_critic_penalty(base.make_critic_optimizer(_critic()))
    half, _ = _critic_run(base.replace(reg_anchor_weight=0.5))
    assert half[i][1]["prox"] == pytest.approx(0.5 * prox, rel=1e-12)
    # With the same critic state, the default exceeds weight 0 by exactly coeff/2 * .5 * W * prox.
    expected = off[i][0] + base.reg_coeff / 2 * .5 * weight * prox
    torch.testing.assert_close(rows[i][0], expected, rtol=1e-12, atol=1e-15)


def test_removed_switches_are_gone():
    for removed in ("direct_particle_gain", "direct_particle_betas", "reg_anchor_decay"):
        with pytest.raises(TypeError):
            get_recipe(**{removed: 1})


@pytest.mark.parametrize("bad", [dict(reg_anchor_weight=-1.0), dict(reg_anchor_weight=float("nan")),
                                 dict(reg_anchor_min_decay=1.0), dict(amsgrad=1)])
def test_switch_validation(bad):
    with pytest.raises(ValueError):
        get_recipe(**bad)

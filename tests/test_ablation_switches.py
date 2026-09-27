"""Recipe ablation switches: defaults are the shipped formulation; off removes the effect."""
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


def _critic_run(recipe, steps=8):
    """Train one critic with the recipe's penalty; return penalties and stats."""
    D = _critic()
    opt = recipe.make_critic_optimizer(D, lr=1e-2)
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


def test_switch_defaults_are_the_shipped_formulation():
    recipe = get_recipe()
    assert recipe.amsgrad is True
    explicit = recipe.replace(amsgrad=True)
    assert explicit == recipe
    rows, D = _critic_run(recipe)
    same, D2 = _critic_run(explicit)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(rows, same))
    assert all(torch.equal(p, q) for p, q in zip(D.parameters(), D2.parameters()))
    # R1 on the reals is live on every step; the fake cap only above kappa.
    assert all(stats["r1"] > 0 for _, stats in rows)


def test_removed_anchor_switches_are_gone():
    for removed in (dict(reg_anchor_weight=0.0), dict(reg_anchor_decay=0.999)):
        with pytest.raises(TypeError):
            get_recipe(**removed)
    recipe = get_recipe()
    with pytest.raises(TypeError):
        recipe.make_critic_optimizer(_critic(), ema_critic=_critic())
    with pytest.raises(TypeError):
        recipe.make_optimizers(nn.Linear(2, 2), _critic(), ema_critic=_critic())
    with pytest.raises(TypeError):
        recipe.make_critic_penalty(recipe.make_critic_optimizer(_critic()), anchor_weight=1.0)


def test_amsgrad_switch_reaches_every_optimizer_group():
    from particlegan.particle_prior import ParticlePrior
    for amsgrad in (False, True):
        recipe = get_recipe(amsgrad=amsgrad)
        prior = ParticlePrior(num_particles=16, z_dim=2)
        opt_g, opt_d = recipe.make_optimizers(nn.Linear(2, 2), _critic(), prior)
        assert [g["amsgrad"] for opt in (opt_g, opt_d) for g in opt.param_groups] == [amsgrad] * 3
    assert get_recipe().make_critic_optimizer(_critic(), amsgrad=False).param_groups[0]["amsgrad"] is False


@pytest.mark.parametrize("bad", [dict(amsgrad=1), dict(amsgrad=None)])
def test_switch_validation(bad):
    with pytest.raises(ValueError):
        get_recipe(**bad)


def test_removed_direct_particle_switches_are_gone():
    with pytest.raises(TypeError):
        get_recipe(direct_particle_gain=True)
    with pytest.raises(TypeError):
        get_recipe().make_generator_optimizer([nn.Parameter(torch.zeros(2))], direct_particles=[])

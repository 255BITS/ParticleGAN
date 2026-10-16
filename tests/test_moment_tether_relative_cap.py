"""Software checks for the opt-in relative-force cap on the moment tether."""
import math

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.moment_tether import MomentTetherRegularizer
from experiments.forge.techniques import recipe_field_active


def bind(rows):
    prior = nn.Module()
    prior.z = nn.Parameter(rows.clone())
    get_recipe("bcap", prior_regularizer="moment_tether", prior_reg=.1).apply_prior_policy(prior)
    return prior


def drifted(seed=0, n=12, d=3):
    g = torch.Generator().manual_seed(seed)
    initial = torch.randn(n, d, generator=g, dtype=torch.float64)
    prior = bind(initial)
    with torch.no_grad():
        prior.z.mul_(1.7).add_(.4)
    target = torch.randn(n, d, generator=g, dtype=torch.float64)
    return prior, target


def adversarial(z, target, scale):
    return scale * (z * target).sum()


def gradients(cap, scale, coefficient=1.):
    prior, target = drifted()
    tether = MomentTetherRegularizer(prior, eps=1e-4, weight=.5, relative_cap=cap)
    loss = adversarial(prior.z, target, scale) + coefficient * tether(prior.z)
    loss.backward()
    return prior, target, prior.z.grad.detach().clone()


def components(scale, coefficient=1.):
    prior, target = drifted()
    adv = torch.autograd.grad(adversarial(prior.z, target, scale), prior.z)[0]
    tether = MomentTetherRegularizer(prior, eps=1e-4, weight=.5)
    teth = torch.autograd.grad(coefficient * tether(prior.z), prior.z)[0]
    return adv, teth


def test_zero_cap_is_exact_fixed_tether():
    _, _, fixed = gradients(0., 1e-3)
    adv, teth = components(1e-3)
    assert torch.equal(fixed, adv + teth)


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.])
@pytest.mark.parametrize("coefficient", [1., .01])
def test_cap_limits_tether_relative_force(scale, coefficient):
    cap = .05
    prior, _, capped = gradients(cap, scale, coefficient)
    adv, teth = components(scale, coefficient)
    expected_scale = min(1., cap * float(adv.norm()) / float(teth.norm()))
    torch.testing.assert_close(capped, adv + expected_scale * teth, rtol=1e-10, atol=1e-14)
    applied = capped - adv
    assert float(applied.norm()) <= cap * float(adv.norm()) * (1 + 1e-9) or expected_scale == 1.


def test_unrelated_backward_does_not_consume_cap_and_stats_record():
    prior, target = drifted()
    tether = MomentTetherRegularizer(prior, eps=1e-4, weight=.5, relative_cap=.1)
    adv_loss = adversarial(prior.z, target, 1e-3)
    reg = tether(prior.z)
    # A protected-loss probe (as in constraint geometry) differentiates only the
    # adversarial objective first; it must be untouched by the cap.
    probe = torch.autograd.grad(adv_loss, prior.z, retain_graph=True)[0]
    adv, teth = components(1e-3)
    assert torch.equal(probe, adv)
    (adv_loss + reg).backward()
    expected = min(1., .1 * float(adv.norm()) / float(teth.norm()))
    torch.testing.assert_close(prior.z.grad, adv + expected * teth, rtol=1e-10, atol=1e-14)
    stats = tether.relative_cap_stats
    assert stats["applications"] == 1
    assert stats["capped"] == int(expected < 1)
    assert math.isclose(stats["last_scale"], expected, rel_tol=1e-9)


def test_concatenated_tables_are_capped_per_table():
    torch.manual_seed(0)
    a = bind(torch.randn(6, 2, dtype=torch.float64))
    b = bind(torch.randn(5, 2, dtype=torch.float64) + 1)
    with torch.no_grad():
        a.z.mul_(2.)
        b.z.add_(.5)
    tether = MomentTetherRegularizer((a, b), weight=.3, relative_cap=.2)
    t1, t2 = torch.randn(6, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    loss = 1e-4 * ((a.z * t1).sum() + (b.z * t2).sum()) + tether(torch.cat([a.z, b.z]))
    loss.backward()
    plain = MomentTetherRegularizer((a, b), weight=.3)
    za, zb = a.z.detach().clone().requires_grad_(), b.z.detach().clone().requires_grad_()
    ga, gb = torch.autograd.grad(plain(torch.cat([za, zb])), (za, zb))
    for grad, adv, teth in ((a.z.grad, 1e-4 * t1, ga), (b.z.grad, 1e-4 * t2, gb)):
        s = min(1., .2 * float(adv.norm()) / float(teth.norm()))
        torch.testing.assert_close(grad, adv + s * teth, rtol=1e-9, atol=1e-14)


def test_recipe_plumbing_and_validation():
    recipe = get_recipe("bcap", prior_regularizer="moment_tether", prior_reg=.01,
                        prior_tether_relative_cap=.05)
    prior = bind(torch.randn(4, 2))
    assert recipe.make_prior_regularizer(prior=prior).relative_cap == .05
    assert recipe_field_active("prior_tether_relative_cap", recipe)
    assert not recipe_field_active("prior_tether_relative_cap", recipe.replace(prior_reg=0.))
    assert "prior_tether_relative_cap" not in get_recipe("bcap").to_dict()
    with pytest.raises(ValueError, match="prior_tether_relative_cap"):
        get_recipe("bcap", prior_tether_relative_cap=.05)
    with pytest.raises(ValueError, match="prior_tether_relative_cap"):
        get_recipe("bcap", prior_regularizer="moment_tether", prior_reg=.01,
                   prior_tether_relative_cap=-1.)


def test_zero_default_preserves_signature_and_identity_fields():
    from dataclasses import asdict
    from experiments.forge import configuration_search as search
    from experiments.forge.techniques import technique_signature
    recipe = asdict(get_recipe("bcap", prior_regularizer="moment_tether", prior_reg=.01))
    historical = {k: v for k, v in recipe.items() if k != "prior_tether_relative_cap"}
    assert technique_signature(recipe) == technique_signature(historical)
    assert search.recipe_identity_fields(recipe) == search.recipe_identity_fields(historical)
    capped = dict(recipe, prior_tether_relative_cap=.05)
    assert technique_signature(capped) != technique_signature(recipe)

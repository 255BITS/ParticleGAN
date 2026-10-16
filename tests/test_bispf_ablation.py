"""Opt-in BISPF component ablation (Recipe.bispf_ablation): default is exact BISPF."""
import pytest
import torch

from particlegan import Recipe
from particlegan.bispf import BISPF_ABLATION_PARTS, SBNPPriorRegularizer, _BISPFPenaltyKernel, parse_bispf_ablation


def _critic():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def _data():
    torch.manual_seed(1)
    return torch.randn(64, 2) * 3, torch.randn(64, 2)


def _penalty(**kwargs):
    real, fake = _data()
    critic = _critic()
    kernel = _BISPFPenaltyKernel(**kwargs)
    pen, _ = kernel.penalty(critic, real, fake, step=1)
    pen.backward()
    return pen.detach(), [p.grad.clone() for p in critic.parameters()]


def test_parse():
    assert parse_bispf_ablation("none") == frozenset()
    assert parse_bispf_ablation("capture,chord") == {"capture", "chord"}
    for bad in ("chord,capture", "chord,chord", "bogus", ","):
        with pytest.raises(ValueError):
            parse_bispf_ablation(bad)
    assert parse_bispf_ablation(",".join(BISPF_ABLATION_PARTS)) == set(BISPF_ABLATION_PARTS)


def test_recipe_field_requires_bispf_and_default_is_none():
    assert Recipe().bispf_ablation == "none"
    with pytest.raises(ValueError):
        Recipe(bispf_ablation="chord")
    with pytest.raises(ValueError):
        Recipe(critic_formulation="bispf", reg_anchor_weight=0.0, bispf_ablation="nope")
    recipe = Recipe(critic_formulation="bispf", reg_anchor_weight=0.0, bispf_ablation="capture,gauge")
    assert recipe._penalty_options()["ablation"] == "capture,gauge"
    assert "ablation" not in Recipe(critic_formulation="bispf", reg_anchor_weight=0.0)._penalty_options()


def test_default_kernel_unchanged_and_reduced_kernel_is_b_cap():
    default, g_default = _penalty()
    explicit, g_explicit = _penalty(ablation="none")
    assert torch.equal(default, explicit)
    assert all(torch.equal(a, b) for a, b in zip(g_default, g_explicit))
    # All critic-side parts off: 1/2 (E relu(|g_r|-k)^2 + E relu(|g_f|-k)^2), BCAP's b_cap.
    reduced, _ = _penalty(ablation="capture,chord,floor,gauge,softplus")
    real, fake = _data()
    critic = _critic()
    norms = []
    for x in (real, fake):
        x = x.clone().requires_grad_(True)
        g = torch.autograd.grad(critic(x).sum(), x)[0]
        norms.append((g.pow(2).sum(1) + 1e-12).sqrt())
    expected = 0.5 * sum(torch.relu(n - 1.0).pow(2).mean() for n in norms)
    assert torch.allclose(reduced, expected, rtol=1e-6, atol=1e-9)
    assert not torch.equal(reduced, default)


def test_tether_thermal_coupling_flag():
    with pytest.raises(ValueError):
        SBNPPriorRegularizer(thermal_coupling=1)
    recipe = Recipe(critic_formulation="bispf", reg_anchor_weight=0.0, prior_regularizer="sbnp", prior_reg=0.01,
                    bispf_ablation="tether_thermal")
    assert recipe.make_prior_regularizer().thermal_coupling is False
    base = Recipe(critic_formulation="bispf", reg_anchor_weight=0.0, prior_regularizer="sbnp", prior_reg=0.01)
    assert base.make_prior_regularizer().thermal_coupling is True

"""reg_real_weight / reg_real_mode / reg_real_kappa: K3P phase A's reals term (R1 by default, or the fakes' cap)."""
import copy

import pytest
import torch
from torch import nn

from particlegan import Recipe, get_recipe
from particlegan.grad_regularizers import GradientPenalty
from particlegan.training import _upgrade_recipe_fields


def _penalty(real_weight=None, coeff=1.0, **options):
    torch.manual_seed(0)
    D = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    real, fake = torch.randn(64, 2), 3 * torch.randn(64, 2)
    for p in D.parameters():  # steep enough that the fake cap (kappa 1) is active
        p.data.mul_(4)
    kw = ({} if real_weight is None else {"real_weight": real_weight}) | options
    pen, _ = GradientPenalty(coeff=coeff, **kw).penalty(D, real, fake, 1, False)
    return pen, D, real, fake


def test_default_is_bit_identical_and_zero_removes_r1():
    base, D, real, fake = _penalty()
    assert torch.equal(_penalty(1.0)[0], base)
    d = real[0].numel()
    r1 = (GradientPenalty._grad_norm(D, real, squared=True) / d).mean()
    cap = ((GradientPenalty._grad_norm(D, fake) / d ** 0.5 - 1.0).relu().square()).mean()
    assert cap > 0 and r1 > 0
    torch.testing.assert_close(_penalty(0.0)[0], 0.5 * cap)
    torch.testing.assert_close(_penalty(0.3)[0], 0.5 * (0.3 * r1 + cap))


def test_recipe_field_default_validation_and_upgrade():
    assert get_recipe().reg_real_weight == 1.0
    assert get_recipe(reg_real_weight=0.1)._penalty_options()["real_weight"] == 0.1
    with pytest.raises(ValueError, match="reg_real_weight"):
        get_recipe(reg_real_weight=-1.0)
    saved = get_recipe().to_dict()
    del saved["reg_real_weight"]
    assert Recipe(**_upgrade_recipe_fields(saved)).reg_real_weight == 1.0


def test_recipe_penalty_reaches_the_kernel():
    D = nn.Linear(2, 1)
    recipe = get_recipe(reg_real_weight=0.25)
    opt = recipe.make_critic_optimizer(D, ema_critic=copy.deepcopy(D))
    assert recipe.make_critic_penalty(opt).regularizer.real_weight == 0.25


def _rms(D, x):
    return GradientPenalty._grad_norm(D, x) / x[0].numel() ** 0.5


def test_real_cap_is_the_fakes_cap_on_reals():
    base, D, real, fake = _penalty()
    assert torch.equal(_penalty(real_mode="r1")[0], base)
    rms_r, rms_f = _rms(D, real), _rms(D, fake)
    cap = lambda rms, k: (rms - k).relu().square().mean()  # noqa: E731
    assert cap(rms_r, 1.0) > 0
    torch.testing.assert_close(_penalty(real_mode="cap")[0], 0.5 * (cap(rms_r, 1.0) + cap(rms_f, 1.0)))
    torch.testing.assert_close(_penalty(real_mode="cap", real_kappa=2.0)[0],
                               0.5 * (cap(rms_r, 2.0) + cap(rms_f, 1.0)))
    # a cap above every real slope leaves only the fake cap (no pull toward zero slope at the reals)
    big = float(rms_r.detach().max()) + 1.0
    torch.testing.assert_close(_penalty(real_mode="cap", real_kappa=big)[0], _penalty(0.0)[0])


def test_real_mode_recipe_fields():
    recipe = get_recipe()
    assert (recipe.reg_real_mode, recipe.reg_real_kappa) == ("r1", None)
    opts = get_recipe(reg_real_mode="cap", reg_real_kappa=0.5)._penalty_options()
    assert (opts["real_mode"], opts["real_kappa"]) == ("cap", 0.5)
    with pytest.raises(ValueError, match="real_mode"):
        get_recipe(reg_real_mode="r2")
    with pytest.raises(ValueError, match="real_kappa"):
        get_recipe(reg_real_mode="cap", reg_real_kappa=-1.0)
    saved = get_recipe().to_dict()
    for key in ("reg_real_mode", "reg_real_kappa"):
        del saved[key]
    upgraded = Recipe(**_upgrade_recipe_fields(saved))
    assert (upgraded.reg_real_mode, upgraded.reg_real_kappa) == ("r1", None)

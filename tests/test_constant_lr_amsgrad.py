"""amsgrad: AMSGrad for every recipe optimizer group (opt-in; recommended at constant LR)."""
import copy

import pytest
import torch
from torch import nn

from particlegan import Recipe, get_recipe
from particlegan.particle_prior import ParticlePrior
from particlegan.training import _upgrade_recipe_fields


def _optimizers(recipe):
    torch.manual_seed(0)
    G, D = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    prior = ParticlePrior(num_particles=16, z_dim=2)
    return recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))


@pytest.mark.parametrize("amsgrad", [False, True])
def test_amsgrad_reaches_every_optimizer_group(amsgrad):
    recipe = get_recipe(total_steps=10, lr_floor=1.0, network_lr_floor=1.0, amsgrad=amsgrad)
    opt_g, opt_d = _optimizers(recipe)
    assert [g["amsgrad"] for opt in (opt_g, opt_d) for g in opt.param_groups] == [amsgrad] * 3


def test_default_is_plain_adam_and_amsgrad_keeps_the_max_second_moment():
    assert get_recipe().amsgrad is False
    _, opt_d = _optimizers(get_recipe(total_steps=10, amsgrad=True))
    params = opt_d.param_groups[0]["params"]
    for p in params:
        p.grad = torch.ones_like(p)
    opt_d.step()
    assert all("max_exp_avg_sq" in opt_d.state[p] for p in params)


def test_adam_kwargs_override_and_validation():
    opt_d = get_recipe(amsgrad=True).make_critic_optimizer(nn.Linear(2, 1), amsgrad=False)
    assert opt_d.param_groups[0]["amsgrad"] is False
    with pytest.raises(ValueError, match="amsgrad"):
        get_recipe(amsgrad=None)


def test_old_checkpoint_recipes_keep_plain_adam():
    saved = get_recipe().to_dict()
    del saved["amsgrad"]
    assert Recipe(**_upgrade_recipe_fields(saved)).amsgrad is False

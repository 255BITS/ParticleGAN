"""The pure BCAP preset is fixed caps plus native Adam at constant rates."""
import pytest
import torch
from torch import nn

from particlegan import Recipe, get_recipe, learning_rate_scales


def test_bcap_public_preset_contains_only_requested_training_mechanisms():
    recipe = get_recipe("bcap")
    assert recipe.critic_formulation == recipe.effective_critic_formulation == "bcap"
    assert recipe.optimizer_family == "adam"
    assert (recipe.reg_arm, recipe.reg_coeff, recipe.reg_kappa, recipe.reg_every) == ("b_cap", 1., 1., 1)
    assert (recipe.lr, recipe.d_lr_mult, recipe.prior_lr_mult, recipe.betas) == (.00425, 1., 2., (0., .999))
    assert recipe.lr_floor == recipe.network_lr_floor == 1.
    assert recipe.network_lr_horizon_cap is recipe.beta2_end is recipe.reg_coeff_end is recipe.continuous_policy is None
    assert recipe.d_guard_ratio == recipe.reg_anchor_weight == recipe.latent_damping_max_rate == 0.
    assert recipe.direct_particle_gain is recipe.amsgrad is False
    assert recipe.prior_reg == recipe.ema_decay == recipe.input_noise_std == recipe.output_noise_std == recipe.output_noise_warmup == 0.
    assert Recipe(**recipe.to_dict()) == recipe


def test_bcap_factories_return_native_adam_and_exact_fixed_cap_gradient():
    recipe = get_recipe("bcap", loss="hinge", lr=.0010625, num_particles=8)
    generator, critic = nn.Linear(2, 2), nn.Linear(2, 1, bias=False)
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    assert type(opt_g) is type(opt_d) is torch.optim.Adam
    assert opt_g.latent_damping is opt_g.direct_response is opt_d.guard is opt_d.anchor is opt_d.ema_critic is None
    assert opt_g.prior_mechanisms["a2"]["enabled"] is False
    assert [group["lr"] for group in opt_g.param_groups] == [.0010625, .002125]
    assert opt_d.param_groups[0]["lr"] == .0010625
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3., 4.]]))
    penalty = recipe.make_critic_penalty(opt_d)
    real, fake = torch.zeros(4, 2), torch.ones(4, 2)
    value = penalty(critic, real, fake)
    assert penalty.regularizer.arm == "b_cap"
    assert value.item() == pytest.approx(16.)  # (norm([3,4])-1)^2
    gradient, = torch.autograd.grad(value, critic.weight)
    assert gradient.tolist()[0] == pytest.approx([4.8, 6.4])
    assert opt_d.record.observed_steps == 0


def test_constant_rate_preset_never_computes_cosine(monkeypatch):
    def cosine_forbidden(_):
        raise AssertionError("constant rates must not evaluate cosine")
    monkeypatch.setattr("particlegan.recipes.math.cos", cosine_forbidden)
    recipe = get_recipe("bcap")
    for step in (0, 1, 200, 4200, 6999, 7000, 100000):
        assert learning_rate_scales(step, recipe) == (1., 1.)


def test_new_bcap_label_does_not_reinterpret_historical_fixed_arm_recipes():
    old = Recipe(reg_arm="b_cap")
    assert old.critic_formulation == old.effective_critic_formulation == "k3p"
    assert old.optimizer_family == "formulation"
    assert old.d_guard_ratio == 5. and old.direct_particle_gain is True
    with pytest.raises(ValueError, match="plain Adam"):
        Recipe(critic_formulation="bcap", reg_arm="b_cap")
    with pytest.raises(ValueError, match="reg_arm='b_cap'"):
        get_recipe("bcap", reg_arm="a_r1r2")

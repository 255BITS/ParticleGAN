"""Public BCAP optimizer presets and bounded checkpoint software controls.

Full-budget acquisition evidence and actual-training GIFs are retained in the
registered pacing study; the tiny updates here check API continuation only.
"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe, learning_rate_scales
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.dualnorm import NormalizedOptimizer


def test_bcap_public_preset_contains_only_requested_training_mechanisms():
    recipe = get_recipe("bcap")
    assert recipe.critic_formulation == recipe.effective_critic_formulation == "bcap"
    assert recipe.optimizer_family == "dualnorm"
    assert recipe.optimizer_momentum == 0. and recipe.optimizer_adam_lr is None
    assert (recipe.reg_arm, recipe.reg_coeff, recipe.reg_kappa, recipe.reg_every) == ("b_cap", 1., 1., 1)
    assert (recipe.lr, recipe.d_lr_mult, recipe.prior_lr_mult, recipe.betas) == (.012, 1.5, 2.5, (0., .999))
    assert recipe.lr_floor == recipe.network_lr_floor == 1.
    assert recipe.network_lr_horizon_cap is recipe.beta2_end is recipe.reg_coeff_end is recipe.continuous_policy is None
    assert recipe.d_guard_ratio == recipe.reg_anchor_weight == recipe.latent_damping_max_rate == 0.
    assert recipe.direct_particle_gain is recipe.amsgrad is False
    assert recipe.prior_reg == recipe.ema_decay == recipe.input_noise_std == recipe.output_noise_std == recipe.output_noise_warmup == 0.
    assert Recipe(**recipe.to_dict()) == recipe


def test_bcap_adam_retains_the_exact_historical_preset_under_an_explicit_name():
    recipe = get_recipe("bcap_adam")
    assert (recipe.optimizer_family, recipe.lr, recipe.d_lr_mult, recipe.prior_lr_mult) == ("adam", .00425, 1., 2.)
    # Apart from the public name and four optimizer choices, every resolved
    # field retains the old law. Checkpoints can reconstruct their saved name.
    assert recipe.replace(name="bcap", optimizer_family="dualnorm", lr=.012,
                          d_lr_mult=1.5, prior_lr_mult=2.5) == get_recipe("bcap")
    assert Recipe(**recipe.to_dict()) == recipe


@pytest.mark.parametrize("preset,optimizer_type", [("bcap", NormalizedOptimizer), ("bcap_adam", torch.optim.Adam)])
def test_bcap_factories_return_declared_optimizer_and_exact_fixed_cap_gradient(preset, optimizer_type):
    recipe = get_recipe(preset, loss="hinge", num_particles=8)
    generator, critic = nn.Linear(2, 2), nn.Linear(2, 1, bias=False)
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    assert type(opt_g) is type(opt_d) is optimizer_type
    assert opt_g.latent_damping is opt_g.direct_response is opt_d.guard is opt_d.anchor is opt_d.ema_critic is None
    assert opt_g.prior_mechanisms["a2"]["enabled"] is False
    assert [group["lr"] for group in opt_g.param_groups] == pytest.approx([recipe.lr, recipe.lr * recipe.prior_lr_mult])
    assert opt_d.param_groups[0]["lr"] == pytest.approx(recipe.lr * recipe.d_lr_mult)
    if preset == "bcap":
        assert opt_g._adam is opt_d._adam is None
        assert [group["role"] for group in opt_g.param_groups] == ["generator", "prior"]
        assert [group["algorithm"] for group in opt_g.param_groups] == ["dualnorm", "rownorm"]
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


@pytest.mark.parametrize("preset", ["bcap", "bcap_adam"])
def test_constant_rate_preset_never_computes_cosine(monkeypatch, preset):
    def cosine_forbidden(_):
        raise AssertionError("constant rates must not evaluate cosine")
    monkeypatch.setattr("particlegan.recipes.math.cos", cosine_forbidden)
    recipe = get_recipe(preset)
    for step in (0, 1, 200, 4200, 6999, 7000, 100000):
        assert learning_rate_scales(step, recipe) == (1., 1.)


def test_new_bcap_label_does_not_reinterpret_historical_fixed_arm_recipes():
    old = Recipe(reg_arm="b_cap")
    assert old.critic_formulation == old.effective_critic_formulation == "k3p"
    assert old.optimizer_family == "formulation"
    assert old.d_guard_ratio == 5. and old.direct_particle_gain is True
    with pytest.raises(ValueError, match="plain optimizer"):
        Recipe(critic_formulation="bcap", reg_arm="b_cap")
    with pytest.raises(ValueError, match="reg_arm='b_cap'"):
        get_recipe("bcap", reg_arm="a_r1r2")


def test_public_dualnorm_default_gives_encoder_the_generator_step_size():
    recipe = get_recipe("bcap", num_particles=8)
    generator, encoder, critic = (nn.Linear(2, 2) for _ in range(3))
    opt_g, opt_d = recipe.make_optimizers(generator, critic, recipe.make_prior(), encoder=encoder)
    assert [group["role"] for group in opt_g.param_groups] == ["generator", "encoder", "prior"]
    assert [group["lr"] for group in opt_g.param_groups] == pytest.approx([.012, .012, .03])
    assert opt_d.param_groups[0]["lr"] == pytest.approx(.018)


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
    else:
        assert left == right


def _trainer(recipe=None):
    from experiments.forge.api import TRAINER_STREAM_BINDINGS
    from experiments.forge.rng import NamedStreams
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = (get_recipe("bcap") if recipe is None else recipe).replace(
            z_dim=2, num_particles=8, batch_size=4, total_steps=4,
            prior_kind="mog", sigma_rel=.025, standardize=False)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for component in (generator, critic, prior):
            deterministic_orthogonal_(component, seed=0)
        streams = NamedStreams(0)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          **{name: streams.generator(family, component=component, purpose=purpose)
                             for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()})


def _batch():
    return torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])


def test_public_dualnorm_default_restores_exact_next_update_and_every_rng():
    trainer = _trainer()
    trainer.step(_batch())
    saved = trainer.state_dict()
    trainer.step(_batch())
    expected = trainer.state_dict()
    restored = _trainer()
    restored.load_state_dict(saved)
    restored.step(_batch())
    _assert_state_equal(expected, restored.state_dict())


@pytest.mark.parametrize("corruption", ["native_adam", "prior_role"])
def test_public_dualnorm_default_rejects_incompatible_optimizer_state_atomically(corruption):
    trainer = _trainer()
    trainer.step(_batch())
    before = trainer.state_dict()
    saved = deepcopy(before)
    if corruption == "native_adam":
        saved["optimizers"][0] = torch.optim.Adam(trainer.G.parameters()).state_dict()
    else:
        saved["optimizers"][0]["param_groups"][0]["role"] = "prior"
    with pytest.raises(ValueError, match="optimizer"):
        trainer.load_state_dict(saved)
    _assert_state_equal(before, trainer.state_dict())


def test_historical_adam_checkpoint_reconstructs_saved_recipe_and_cannot_load_as_new_default():
    historical = _trainer(get_recipe("bcap_adam").replace(name="bcap"))
    historical.step(_batch())
    saved = historical.state_dict()
    current = _trainer()
    before = current.state_dict()
    with pytest.raises(ValueError, match="recipe"):
        current.load_state_dict(saved)
    _assert_state_equal(before, current.state_dict())
    restored = _trainer(Recipe(**saved["recipe"]))
    restored.load_state_dict(saved)
    historical.step(_batch())
    restored.step(_batch())
    _assert_state_equal(historical.state_dict(), restored.state_dict())

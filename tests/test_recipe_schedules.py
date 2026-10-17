"""Public configurable plain-Adam schedules, formulas and exact continuation."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from benchmarks.toy100.schedule import step_with_policy
from particlegan import GANTrainer, Recipe, get_recipe, scale_learning_rates
from particlegan.recipe_schedules import apply_training_schedules, cosine_value


def plain_recipe(**changes):
    return Recipe(**{
        "name": "independently-named-baseline", "optimizer_family": "adam", "reg_arm": "a_r1r2",
        "num_particles": 8, "z_dim": 2, "batch_size": 4, "total_steps": 20,
        "standardize": False, "lr": .0002, "d_lr_mult": 1., "prior_lr_mult": 1.,
        "betas": (0., .9), "beta2_end": .99, "beta2_anneal_end": .2,
        "reg_coeff": 1., "reg_coeff_end": .1, "reg_coeff_anneal_end": .2,
        "reg_every": 1, "reg_anchor_weight": 0., "d_guard_ratio": 0.,
        "latent_damping_max_rate": 0., "direct_particle_gain": False,
        "input_noise_std": 0., "output_noise_std": 0., "amsgrad": False,
        "network_lr_floor": 1., "network_lr_horizon_cap": None, "lr_floor": 1., **changes})


def make_trainer(recipe=None, **options):
    torch.manual_seed(17)
    return GANTrainer(recipe or plain_recipe(), nn.Linear(2, 2), nn.Linear(2, 1), seed=7, **options)


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def test_plain_adam_factories_are_native_and_match_manual_adam_updates():
    recipe = plain_recipe(eps=1e-7)
    left, right = nn.Linear(2, 1), nn.Linear(2, 1)
    right.load_state_dict(left.state_dict())
    actual = recipe.make_critic_optimizer(left, ema_critic=deepcopy(left))
    expected = torch.optim.Adam(right.parameters(), lr=.0002, betas=(0., .9), eps=1e-7)
    assert type(actual) is torch.optim.Adam
    assert actual.anchor is actual.guard is actual.ema_critic is None
    for step in range(6):
        apply_training_schedules(step, recipe, [actual])
        expected.param_groups[0]["betas"] = actual.param_groups[0]["betas"]
        for module in (left, right):
            for parameter in module.parameters():
                parameter.grad = torch.full_like(parameter, .25 + step)
        actual.step()
        expected.step()
        equal_state(left.state_dict(), right.state_dict())
        assert actual.record.observed_steps == step + 1
    generator = recipe.make_generator_optimizer(left.parameters(), direct_particles=left.weight)
    assert type(generator) is torch.optim.Adam
    assert generator.latent_damping is generator.latent_history is generator.direct_response is generator.direct_history is None


@pytest.mark.parametrize("step,beta2,gamma", [(0, .9, 1.), (2, .945, .55), (4, .99, .1), (9, .99, .1)])
def test_cosine_endpoints_midpoint_and_flat_lr_cover_every_parameter_role(step, beta2, gamma):
    trainer = make_trainer()
    scale_learning_rates(step, trainer.recipe, (trainer.opt_g, trainer.opt_d), trainer.initial_lrs, trainer.prior)
    apply_training_schedules(step, trainer.recipe, (trainer.opt_g, trainer.opt_d), trainer.penalty)
    for optimizer in (trainer.opt_g, trainer.opt_d):
        for group in optimizer.param_groups:
            assert group["betas"] == pytest.approx((0., beta2))
            assert group["lr"] == .0002
    assert trainer.penalty.regularizer.coeff == pytest.approx(gamma)
    assert trainer.prior_mechanisms["a2"]["enabled"] is False


def test_fixed_r1_r2_has_paper_half_gamma_l2_units_and_counts_updates_not_calls():
    recipe = plain_recipe()
    critic = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1., 2.]]))
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(optimizer)
    batch = torch.ones(4, 2)
    assert penalty(critic, batch, batch).item() == pytest.approx(5.)
    assert penalty(critic, batch, batch).item() == pytest.approx(5.)
    assert optimizer.record.observed_steps == 0 and optimizer.record.calls == 2
    for _ in range(2):
        critic.weight.grad = torch.zeros_like(critic.weight)
        optimizer.step()
    assert penalty(critic, batch, batch).item() == pytest.approx(2.75)
    assert optimizer.record.observed_steps == 2
    assert penalty.regularizer.arm == "a_r1r2"


def test_training_schedule_checkpoint_restores_exactly_across_burn_in():
    trainer = make_trainer()
    batch = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    for _ in range(3):
        trainer.step(batch)
    checkpoint = trainer.state_dict()
    for _ in range(4):
        trainer.step(batch)
    expected = trainer.state_dict()
    restored = make_trainer()
    restored.load_state_dict(checkpoint)
    for _ in range(4):
        restored.step(batch)
    equal_state(expected, restored.state_dict())
    assert restored.opt_d.record.observed_steps == 7
    assert restored.penalty.regularizer.coeff == .1


def test_checkpoint_preflight_rejects_bad_observer_and_schedule_before_live_mutation():
    trainer = make_trainer()
    trainer.step(torch.ones(4, 2))
    before = trainer.state_dict()
    bad = deepcopy(before)
    bad["optimizers"][1]["regularizer"]["record"]["observed_steps"] = -1
    with pytest.raises(ValueError, match="optimizer state"):
        trainer.load_state_dict(bad)
    equal_state(before, trainer.state_dict())
    bad = deepcopy(before)
    bad["optimizers"][0]["param_groups"][0]["_recipe_initial_betas"] = (0., .8)
    with pytest.raises(ValueError, match="optimizer state"):
        trainer.load_state_dict(bad)
    equal_state(before, trainer.state_dict())
    changed = make_trainer(plain_recipe(beta2_end=.98))
    unchanged = changed.state_dict()
    with pytest.raises(ValueError, match="recipe"):
        changed.load_state_dict(before)
    equal_state(unchanged, changed.state_dict())


def test_invalid_policy_begin_does_not_mutate_schedule():
    trainer = make_trainer()
    trainer.completed_steps = 2
    betas = [group["betas"] for optimizer in (trainer.opt_g, trainer.opt_d) for group in optimizer.param_groups]
    coefficient = trainer.penalty.regularizer.coeff
    with pytest.raises(ValueError, match="real"):
        trainer.policy.begin_step(torch.ones(4))
    assert betas == [group["betas"] for optimizer in (trainer.opt_g, trainer.opt_d) for group in optimizer.param_groups]
    assert trainer.penalty.regularizer.coeff == coefficient


def test_uncapped_native_fixed_lr_path_keeps_beta2_and_original_schedule_horizon():
    trainer = make_trainer(max_steps=3)
    batch = torch.ones(4, 2)
    for step in range(3):
        step_with_policy(trainer, batch, network_lr_horizon_cap=None, network_lr_floor=1.)
        expected = cosine_value(step, .9, .99, .2, 20)
        assert all(group["lr"] == .0002 and group["betas"][1] == expected
                   for optimizer in (trainer.opt_g, trainer.opt_d) for group in optimizer.param_groups)
    assert trainer.opt_d.record.observed_steps == 3
    assert trainer.penalty.regularizer.coeff == pytest.approx(.55)


def test_default_recipe_dicts_and_components_remain_legacy_compatible():
    fields = {"optimizer_family", "eps", "beta2_end", "beta2_anneal_end", "reg_coeff_end", "reg_coeff_anneal_end"}
    for name in ("ka2", "k3p", "e22"):
        recipe = get_recipe(name)
        assert not fields & recipe.to_dict().keys()
        assert Recipe(**recipe.to_dict()) == recipe
    renamed = plain_recipe(name="anything-the-caller-chooses")
    assert renamed.to_dict()["optimizer_family"] == "adam"
    assert renamed.beta2_end == .99 and renamed.reg_coeff_end == .1


@pytest.mark.parametrize("fractions", [
    {"beta2_anneal_end": .4}, {"reg_coeff_anneal_end": .6},
    {"beta2_anneal_end": .4, "reg_coeff_anneal_end": .6},
])
def test_inactive_nondefault_schedule_fractions_survive_roundtrip_and_later_activation(fractions):
    recipe = plain_recipe(beta2_end=None, reg_coeff_end=None, **fractions)
    packet = recipe.to_dict()
    assert all(packet[key] == value for key, value in fractions.items())
    restored = Recipe(**packet)
    assert restored == recipe
    activated = restored.replace(beta2_end=.99, reg_coeff_end=.1)
    trainer = make_trainer(activated)
    apply_training_schedules(4, activated, (trainer.opt_g, trainer.opt_d), trainer.penalty)
    assert trainer.opt_g.param_groups[0]["betas"][1] == cosine_value(
        4, .9, .99, fractions.get("beta2_anneal_end", .2), 20)
    assert trainer.penalty.regularizer.coeff == cosine_value(
        4, 1., .1, fractions.get("reg_coeff_anneal_end", .2), 20)


@pytest.mark.parametrize("changes", [
    {"optimizer_family": "unknown"}, {"eps": 0}, {"eps": float("nan")},
    {"beta2_end": 1}, {"beta2_anneal_end": 0}, {"beta2_anneal_end": 1.1},
    {"reg_coeff_end": -1}, {"reg_coeff_anneal_end": 0}, {"d_guard_ratio": 1},
    {"reg_anchor_weight": 1}, {"latent_damping_max_rate": .5}, {"direct_particle_gain": True},
    {"reg_arm": "k3p"}, {"optimizer_family": "formulation"},
])
def test_invalid_or_incompatible_recipe_schedules_are_rejected(changes):
    with pytest.raises(ValueError):
        plain_recipe(**changes)

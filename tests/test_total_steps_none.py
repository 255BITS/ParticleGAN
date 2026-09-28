"""total_steps=None (the default) means no horizon: training runs indefinitely."""
import json

import pytest
import torch
from torch import nn

from benchmarks.legacy.recipe import LegacyRecipe
from particlegan import (GANTrainer, Recipe, get_recipe, learning_rate_scale, learning_rate_scales,
                         scale_learning_rates)
from particlegan.training import input_noise_std, output_noise_std


def make_trainer(**overrides):
    recipe = get_recipe(num_particles=8, z_dim=2, batch_size=4, **overrides)
    torch.manual_seed(0)
    generator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1))
    return GANTrainer(recipe, generator, critic)


def test_default_recipe_has_no_horizon():
    assert Recipe().total_steps is None and get_recipe().total_steps is None
    for name in ("gan", "mog", "ddgan", "ddgan_mog", "ae_gan", "vae_gan", "ae_ddgan"):
        assert get_recipe(name).total_steps is None


@pytest.mark.parametrize("value", [0, -1, 7000.0, "7000", True])
def test_total_steps_must_be_none_or_a_positive_integer(value):
    with pytest.raises(ValueError, match="total_steps must be a positive integer or None"):
        get_recipe(total_steps=value)


def test_default_trainer_trains_past_the_old_7000_step_budget():
    trainer = make_trainer()
    batch = lambda: torch.randn(4, 2)  # noqa: E731
    for _ in range(3):
        trainer.step(batch())
    state = trainer.state_dict()
    state["completed_steps"] = 7000
    trainer.load_state_dict(state)
    for expected in range(7001, 7006):
        stats = trainer.step(batch())
        assert stats["step"] == expected and torch.isfinite(stats["loss_d"])
    state = trainer.state_dict()
    state["completed_steps"] = 10**9
    trainer.load_state_dict(state)  # any nonnegative count resumes without a budget
    assert trainer.step(batch())["step"] == 10**9 + 1
    state["completed_steps"] = -1
    with pytest.raises(ValueError, match="invalid checkpoint step count"):
        trainer.load_state_dict(state)


def test_integer_budget_still_stops_and_bounds_checkpoints():
    trainer = make_trainer(total_steps=2)
    trainer.step(torch.randn(4, 2))
    trainer.step(torch.randn(4, 2))
    with pytest.raises(RuntimeError, match="budget exhausted"):
        trainer.step(torch.randn(4, 2))
    state = trainer.state_dict()
    state["completed_steps"] = 3
    with pytest.raises(ValueError, match="invalid checkpoint step count"):
        trainer.load_state_dict(state)


@pytest.mark.parametrize("overrides,needs", [
    (dict(lr_floor=.05), "lr_floor"),
    (dict(network_lr_floor=.01), "network LR floor"),
    (dict(lr_floor=.05, network_lr_horizon_cap=100), "lr_floor"),
    (dict(input_noise_std=.1), "input_noise_std"),
    (dict(output_noise_warmup=.2), "output_noise_warmup"),
])
def test_horizon_features_require_total_steps(overrides, needs):
    with pytest.raises(ValueError, match=f"total_steps=None.*{needs}.*set total_steps"):
        get_recipe(**overrides)
    get_recipe(total_steps=100, **overrides)  # an explicit horizon enables each of them


def test_inert_horizon_settings_construct_without_total_steps():
    # A floor of 1, zero input noise, or warmup of zero output noise needs no horizon.
    get_recipe(network_lr_floor=1.0, input_noise_anneal_end=.5, lr_anneal_start=.2)
    get_recipe(output_noise_warmup=.2, output_noise_std=0.0)
    # The G/D schedule can run over its own cap while the prior (floor 1) holds.
    recipe = get_recipe(network_lr_floor=.01, network_lr_horizon_cap=100)
    assert learning_rate_scales(0, recipe) == (1.0, 1.0)
    network, prior = learning_rate_scales(100, recipe)
    assert network == pytest.approx(.01) and prior == 1.0
    assert learning_rate_scales(100, recipe) == learning_rate_scales(
        100, recipe.replace(total_steps=10**6))


def test_helpers_return_constants_without_reading_the_missing_horizon():
    recipe = get_recipe()
    for step in (0, 1, 7000, 7001, 10**9):
        assert input_noise_std(recipe, step) == 0.0
        assert output_noise_std(recipe, step) == recipe.output_noise_std == .029
        assert learning_rate_scales(step, recipe) == (1.0, 1.0)
        assert learning_rate_scale(step, None, floor=1.0) == 1.0
    assert output_noise_std(get_recipe(output_noise_warmup=.2, output_noise_std=0.0), 5) == 0.0
    with pytest.raises(ValueError, match="invalid learning-rate schedule"):
        learning_rate_scale(5, None, floor=.05)
    with pytest.raises(ValueError, match="invalid learning-rate schedule"):
        learning_rate_scale(5, 0, floor=1.0)
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=.25)
    assert scale_learning_rates(10**6, recipe, [optimizer], [[.5]]) == (1.0, 1.0)
    assert optimizer.param_groups[0]["lr"] == .5


def test_integer_horizon_keeps_the_schedules():
    recipe = get_recipe(total_steps=100, lr_floor=.05, input_noise_std=.2, output_noise_warmup=.5)
    assert input_noise_std(recipe, 0) == pytest.approx(.2)
    assert input_noise_std(recipe, 5) == pytest.approx(.1)
    assert input_noise_std(recipe, 10) == 0.0
    assert output_noise_std(recipe, 25) == pytest.approx(.029 / 2)
    assert learning_rate_scales(100, recipe) == (pytest.approx(.05), pytest.approx(.05))


@pytest.mark.parametrize("total_steps", [None, 7000])
def test_recipe_and_trainer_checkpoints_round_trip(total_steps):
    recipe = get_recipe(total_steps=total_steps)
    assert Recipe(**json.loads(json.dumps(recipe.to_dict()))) == recipe
    trainer = make_trainer(total_steps=total_steps)
    trainer.step(torch.randn(4, 2))
    state = trainer.state_dict()
    assert state["recipe"]["total_steps"] == total_steps
    restored = make_trainer(total_steps=total_steps)
    restored.load_state_dict(state)
    assert restored.completed_steps == 1
    other = make_trainer(total_steps=5 if total_steps is None else None)
    with pytest.raises(ValueError, match="checkpoint recipe does not match trainer"):
        other.load_state_dict(state)


def test_legacy_recipe_keeps_its_recorded_budget():
    assert LegacyRecipe().total_steps == 7000

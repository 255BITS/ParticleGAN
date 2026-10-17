"""Opt-in progress-driven (horizon-free) K3P schedules."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.recipe_schedules import ProgressClock, progress_fraction_scale


def k3p(**changes):
    return get_recipe("k3p", num_particles=8, z_dim=2, batch_size=4, total_steps=40,
                      standardize=False, **changes)


def trainer(recipe):
    torch.manual_seed(3)
    return GANTrainer(recipe, nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)),
                      nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)), seed=5)


def run(model, steps, start=0):
    data = torch.Generator().manual_seed(11 + start)
    for _ in range(steps):
        model.step(torch.randn(4, 2, generator=data) + 2)


def test_default_recipe_serialization_is_unchanged():
    assert "progress_schedule" not in k3p().to_dict()
    assert "progress_rate" not in k3p().to_dict()
    assert k3p(progress_schedule="critic_order").to_dict()["progress_schedule"] == "critic_order"


def test_validation_rejects_conflicting_schedules():
    with pytest.raises(ValueError):
        k3p(progress_schedule="bogus")
    with pytest.raises(ValueError):
        k3p(progress_schedule="updates", progress_rate=0.)
    with pytest.raises(ValueError):
        k3p(progress_schedule="updates", lr_schedule="constant", network_lr_horizon_cap=None)


def test_fraction_scale_matches_cosine_shape():
    assert progress_fraction_scale(0., .6, .01) == 1.
    assert progress_fraction_scale(.6, .6, .01) == 1.
    assert progress_fraction_scale(1., .6, .01) == pytest.approx(.01)
    assert progress_fraction_scale(.8, .6, .05) == pytest.approx(.05 + .95 * .5)


def test_ungated_progress_is_an_internal_counter_and_schedules_follow_it():
    model = trainer(k3p(progress_schedule="updates", progress_rate=.1))
    run(model, 3)
    assert model.policy.progress.progress == pytest.approx(.3)
    assert model.policy.progress.input_noise(model.recipe) == 0.
    assert model.policy.progress.output_noise(model.recipe) == pytest.approx(model.recipe.output_noise_std)


def test_progress_ignores_step_label_and_horizon_and_restores_exactly():
    recipe = k3p(progress_schedule="critic_order", progress_rate=.05)
    reference = trainer(recipe)
    run(reference, 4)
    saved = reference.state_dict()
    run(reference, 4, start=4)
    expected = reference.state_dict()

    shifted = trainer(recipe)
    shifted.load_state_dict(deepcopy(saved))
    shifted.completed_steps += 1000
    shifted.max_steps = 5000
    shifted.recipe = shifted.policy.recipe = recipe.replace(total_steps=4096)
    run(shifted, 4, start=4)
    actual = shifted.state_dict()
    assert actual["progress"] == expected["progress"]
    for name in ("G", "D", "prior"):
        for key, value in expected["models"][name].items():
            assert torch.equal(value, actual["models"][name][key])
    assert 0 < expected["progress"]["progress"] <= .4


def test_clock_load_rejects_bad_state():
    clock = ProgressClock(k3p(progress_schedule="updates"))
    with pytest.raises(ValueError):
        clock.load_state_dict({"progress": 2.0, "order_ema": None, "observed": 0})

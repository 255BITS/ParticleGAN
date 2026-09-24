"""Numerical and replay checks for the research-only Optimistic Adam adapter."""

from copy import deepcopy
import math

import pytest
import torch

from benchmarks.toy100.schedule import policy_multipliers, step_with_policy
from lib.toy_models import SimpleMLPDiscriminator
from particlegan import GANTrainer, Recipe
from reports.toy100.optimistic_adam_scratch import (
    PREVIOUS_DIRECTION_KEY, UPDATE_COUNT_KEY, optimistic_adam,
)


def test_formula_uses_current_rate_for_both_directions():
    parameter = torch.nn.Parameter(torch.tensor([0.75, -0.5], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.03, betas=(0.2, 0.8), eps=1e-7)
    gradients = ([0.3, -0.4], [-0.2, 0.1], [0.8, -0.5], [0.1, 0.2])
    rates = (0.03, 0.01, 0.002, 0.004)
    expected = torch.tensor([0.75, -0.5], dtype=torch.float64)
    first = torch.zeros_like(expected)
    second = torch.zeros_like(expected)
    previous = torch.zeros_like(expected)
    alpha = 0.5
    with optimistic_adam(alpha) as recorder:
        for t, (gradient, rate) in enumerate(zip(gradients, rates), start=1):
            grad = torch.tensor(gradient, dtype=torch.float64)
            first = 0.2 * first + 0.8 * grad
            second = 0.8 * second + 0.2 * grad.square()
            current = (first / (1 - 0.2 ** t)) / (
                (second / (1 - 0.8 ** t)).sqrt() + 1e-7
            )
            expected -= rate * ((1 + alpha) * current - alpha * previous)
            previous = current.clone()
            optimizer.param_groups[0]["lr"] = rate
            parameter.grad = grad
            optimizer.step()
            assert torch.allclose(parameter, expected, atol=2e-15, rtol=0)
    receipt = recorder.receipt()
    assert receipt["optimizer_count"] == 1
    assert receipt["optimizer_step_calls"] == len(rates)
    assert receipt["parameter_updates"] == len(rates)
    assert receipt["optimizers"][0]["group_lrs"] == [[rate] for rate in rates]
    assert receipt["optimizers"][0]["optimistic_parameter_update_count"] == len(rates)
    assert torch.allclose(optimizer.state[parameter][PREVIOUS_DIRECTION_KEY], previous,
                          atol=1e-15, rtol=0)


def test_zero_alpha_is_bitwise_ordinary_adam_with_original_state_schema():
    a = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    b = torch.nn.Parameter(a.detach().clone())
    base = torch.optim.Adam([a], lr=0.01, betas=(0.0, 0.99))
    trial = torch.optim.Adam([b], lr=0.01, betas=(0.0, 0.99))
    schedule = [(0.01, [0.3, -0.7]), (0.004, [-0.1, 0.2]),
                (0.001, [0.5, 0.3])]
    baseline = []
    for rate, values in schedule:
        base.param_groups[0]["lr"] = rate
        a.grad = torch.tensor(values, dtype=torch.float64)
        base.step()
        baseline.append((a.detach().clone(), deepcopy(base.state_dict())))
    with optimistic_adam(0) as recorder:
        for (rate, values), (expected_parameter, expected_state) in zip(schedule, baseline):
            grad = torch.tensor(values, dtype=torch.float64)
            trial.param_groups[0]["lr"] = rate
            b.grad = grad.clone()
            trial.step()
            assert torch.equal(expected_parameter, b)
            actual_state = trial.state_dict()
            assert actual_state["param_groups"] == expected_state["param_groups"]
            assert actual_state["state"].keys() == expected_state["state"].keys()
            for key in actual_state["state"]:
                assert actual_state["state"][key].keys() == expected_state["state"][key].keys()
                for name in actual_state["state"][key]:
                    assert torch.equal(actual_state["state"][key][name],
                                       expected_state["state"][key][name])
    assert recorder.receipt()["optimizers"][0]["previous_direction_state_count"] == 0


def test_checkpoint_restores_unscaled_previous_direction_at_new_rate():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -0.25], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.02, betas=(0.3, 0.9))
    with optimistic_adam(1):
        for values in ([0.2, -0.1], [-0.4, 0.3]):
            parameter.grad = torch.tensor(values, dtype=torch.float64)
            optimizer.step()
        saved_parameter = parameter.detach().clone()
        saved_optimizer = deepcopy(optimizer.state_dict())
        restored = torch.nn.Parameter(saved_parameter.clone())
        replay = torch.optim.Adam([restored], lr=0.02, betas=(0.3, 0.9))
        replay.load_state_dict(saved_optimizer)
        assert torch.equal(optimizer.state[parameter][PREVIOUS_DIRECTION_KEY],
                           replay.state[restored][PREVIOUS_DIRECTION_KEY])
        for rate, values in [(0.002, [0.7, -0.6]), (0.007, [-0.2, 0.5])]:
            grad = torch.tensor(values, dtype=torch.float64)
            optimizer.param_groups[0]["lr"] = replay.param_groups[0]["lr"] = rate
            parameter.grad = grad.clone()
            restored.grad = grad.clone()
            optimizer.step()
            replay.step()
            assert torch.equal(parameter, restored)
            assert optimizer.state[parameter][UPDATE_COUNT_KEY] == replay.state[restored][UPDATE_COUNT_KEY]
            assert torch.equal(optimizer.state[parameter][PREVIOUS_DIRECTION_KEY],
                               replay.state[restored][PREVIOUS_DIRECTION_KEY])


def test_native_g_d_and_prior_groups_use_actual_capped_rates():
    recipe = Recipe(
        total_steps=4, num_particles=32, z_dim=2, batch_size=8,
        lr=0.01, prior_lr_mult=2.0, d_lr_mult=1.0,
        lr_anneal_start=0.5, lr_floor=0.05,
    )
    generator = torch.nn.Linear(2, 2)
    discriminator = SimpleMLPDiscriminator(in_dim=2, hidden_dim=8, n_hidden=1, fourier=1)
    prior = recipe.make_prior(learnable=True)
    real = torch.linspace(-1, 1, 16).reshape(8, 2)
    with optimistic_adam(0.5) as recorder:
        trainer = GANTrainer(recipe, generator, discriminator, prior=prior, seed=17)
        for _ in range(4):
            step_with_policy(trainer, real, generator_real=real,
                             network_lr_horizon_cap=2, network_lr_floor=0.01)
    records = recorder.receipt()["optimizers"]
    d = next(row for row in records if len(row["group_parameter_counts"]) == 1)
    g = next(row for row in records if len(row["group_parameter_counts"]) == 2)
    assert d["step_calls"] == g["step_calls"] == 4
    assert all(count > 0 for count in d["group_parameter_updates"] + g["group_parameter_updates"])
    assert d["previous_direction_state_count"] > 0
    assert g["previous_direction_state_count"] > 0
    for t in range(4):
        network, prior_scale = policy_multipliers(
            t, 4, recipe.lr_anneal_start, recipe.lr_floor, 2,
            network_lr_floor=0.01,
        )
        assert g["group_lrs"][t] == [recipe.lr * network,
                                      recipe.lr * recipe.prior_lr_mult * prior_scale]
        assert d["group_lrs"][t] == [recipe.lr * recipe.d_lr_mult * network]


def _bilinear_radius(alpha: float) -> float:
    """A deterministic min-x/max-y xy game, with simultaneous gradients."""
    x = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    # A large epsilon makes the adaptive denominator nearly constant, so
    # ordinary Adam exhibits the bilinear rotation of near-raw GD.
    opt_x = torch.optim.Adam([x], lr=1.0, betas=(0.0, 0.0), eps=10.0)
    opt_y = torch.optim.Adam([y], lr=1.0, betas=(0.0, 0.0), eps=10.0)
    with optimistic_adam(alpha):
        for _ in range(400):
            # Compute both gradients at the same (x_t,y_t); stepping remains
            # one call per player and introduces no extra loss evaluation.
            gx, gy = y.detach().clone(), -x.detach().clone()
            x.grad, y.grad = gx, gy
            opt_x.step()
            opt_y.step()
    return math.hypot(float(x.detach()), float(y.detach()))


def test_optimism_damps_a_bilinear_cycle_with_one_gradient_per_player():
    assert _bilinear_radius(0.0) > math.sqrt(2)
    assert _bilinear_radius(1.0) < math.sqrt(2) / 2


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True])
def test_invalid_alpha_rejected(bad):
    with pytest.raises(ValueError, match="optimism alpha"):
        with optimistic_adam(bad):
            pass

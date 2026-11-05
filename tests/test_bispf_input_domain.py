"""Domain declarations exercise the public penalty/trainer, without quality claims."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import GANTrainer, get_recipe, init


def penalty_case(shape, domain=None, **recipe_options):
    width = math.prod(shape)
    critic = nn.Sequential(nn.Flatten(), nn.Linear(width, 1)).double()
    with torch.no_grad():
        critic[1].weight.fill_(0.2)
        critic[1].bias.fill_(0.3)
    recipe = get_recipe("bispf_optimized", batch_size=4, total_steps=4, **recipe_options)
    optimizer = recipe.make_critic_optimizer(critic)
    options = {} if domain is None else {"input_domain": domain}
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True, **options)
    real = torch.linspace(-0.5, 0.8, 4 * width, dtype=torch.float64).reshape(4, *shape)
    fake = real * 0.7 - 0.2
    return critic, penalty, real, fake


@pytest.mark.parametrize("domain", [None, True, False, 1, [], "", "vector", "CONTINUOUS"])
def test_public_factory_refuses_unknown_domain(domain):
    recipe = get_recipe("bispf_optimized")
    critic = nn.Linear(2, 1)
    optimizer = recipe.make_critic_optimizer(critic)
    with pytest.raises(ValueError, match="input_domain"):
        recipe.make_critic_penalty(optimizer, input_domain=domain)


@pytest.mark.parametrize("shape", [(2,), (64,), (169,), (170,), (1024,), (2, 3, 4)])
def test_auto_preserves_values_gradients_and_records(shape):
    results = []
    corresponding_domain = "categorical" if len(shape) == 1 and shape[0] >= 170 else "continuous"
    for domain in (None, "auto", corresponding_domain):
        critic, penalty, real, fake = penalty_case(shape, domain)
        value = penalty(critic, real, fake)
        gradients = torch.autograd.grad(value, tuple(critic.parameters()), allow_unused=True)
        results.append((value.detach(), gradients, deepcopy(penalty.last_stats)))
    for value, gradients, stats in results[1:]:
        assert torch.equal(value, results[0][0])
        assert stats == results[0][2]
        for actual, expected in zip(gradients, results[0][1]):
            assert actual is None and expected is None or torch.equal(actual, expected)


@pytest.mark.parametrize("width", [169, 170, 1024, 2048, 4096])
def test_wide_continuous_uses_existing_rms_floor_cap_and_gauge(width):
    critic, penalty, real, fake = penalty_case((width,), "continuous")
    value = penalty(critic, real, fake)
    stats = penalty.last_stats
    kernel = penalty.regularizer
    weight = critic[1].weight
    gradient_norm = (weight.square().sum() + 1e-12).sqrt() / math.sqrt(width)
    expected_floor = 0.5 * kernel.lambda_floor * weight.square().mean()
    expected_cap = kernel.lambda_cap * F.softplus(gradient_norm - kernel.kappa,
                                                 beta=kernel.ebi_beta).square()
    mean_logit = 0.5 * (critic(real).mean() + critic(fake).mean())
    expected_gauge = 0.5 * min(kernel.gamma_gauge, 0.020) * mean_logit.square()
    assert float(value.detach()) == pytest.approx(float((expected_floor + expected_cap + expected_gauge).detach()))
    assert stats["mean_grad_norm"] == pytest.approx(float(gradient_norm.detach()))
    assert stats["floor_pen"] > 0 and stats["gauge_loss"] > 0
    first = torch.autograd.grad(value, weight, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), weight)[0]
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    assert torch.isfinite(second).all() and second.abs().sum() > 0


def test_explicit_small_categorical_uses_categorical_law_and_refuses_image_geometry():
    critic, penalty, real, fake = penalty_case((16,), "categorical")
    penalty(critic, real, fake)
    assert penalty.last_stats["floor_pen"] == 0
    assert penalty.last_stats["gauge_loss"] == 0
    assert penalty.last_stats["s_thermal"] == 1
    assert penalty.last_stats["mean_grad_norm"] == pytest.approx(0.2 * math.sqrt(16))
    with pytest.raises(ValueError, match="categories"):
        penalty(critic, real.reshape(4, 1, 4, 4), fake.reshape(4, 1, 4, 4))


@pytest.mark.parametrize("options", [
    {"bispf_shape_gates": False},
    {"bispf_gate_overrides": {"simplex_enabled": False}},
    {"bispf_gate_overrides": {"simplex_min_dim": 2}},
])
@pytest.mark.parametrize("domain", ["continuous", "categorical"])
def test_explicit_domain_takes_precedence_over_develop_shape_gates(options, domain):
    critic, penalty, real, fake = penalty_case((16,), domain, **options)
    penalty(critic, real, fake)
    if domain == "categorical":
        assert penalty.last_stats["floor_pen"] == penalty.last_stats["gauge_loss"] == 0
    else:
        assert penalty.last_stats["floor_pen"] > 0
        assert penalty.last_stats["gauge_loss"] > 0


@pytest.mark.parametrize("options,categorical", [
    ({"bispf_shape_gates": False}, False),
    ({"bispf_gate_overrides": {"simplex_enabled": False}}, False),
    ({"bispf_gate_overrides": {"simplex_min_dim": 2}}, True),
])
def test_auto_keeps_develop_simplex_gate_contract(options, categorical):
    critic, penalty, real, fake = penalty_case((170,), "auto", **options)
    penalty(critic, real, fake)
    assert (penalty.last_stats["floor_pen"] == 0) == categorical


def test_two_category_inputs_do_not_enter_late_continuous_thermal_freeze():
    critic, penalty, _, _ = penalty_case((2,), "categorical")
    rows = torch.eye(2, dtype=torch.float64).repeat(64, 1)
    # A matched distribution at this call count triggers the 2-D continuous
    # freeze. It must not cool an explicitly categorical two-token codebook.
    penalty.regularizer.record.calls = 1649
    penalty(critic, rows, rows)
    assert penalty.last_stats["s_thermal"] == 1
    assert penalty.last_stats["floor_pen"] == penalty.last_stats["gauge_loss"] == 0


def wide_trainer(domain):
    recipe = get_recipe("bispf_optimized", num_particles=8, z_dim=2,
                        batch_size=4, total_steps=4)
    generator = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 256))
    critic = nn.Sequential(nn.Linear(256, 8), nn.SiLU(), nn.Linear(8, 1))
    for index, module in enumerate((generator, critic)):
        init.deterministic_orthogonal_(module, seed=index)
    return GANTrainer(recipe, generator, critic, seed=0,
                      penalty_options={"input_domain": domain})


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_public_wide_trainer_continuation_and_domain_identity():
    trainer = wide_trainer("continuous")
    rows = torch.linspace(-1, 1, 4 * 256).reshape(4, 256)
    trainer.step(rows)
    checkpoint = deepcopy(trainer.state_dict())
    assert checkpoint["penalty_options"] == {"input_domain": "continuous"}
    expected = trainer.step(rows * 0.9, collect_stats=True)
    expected_state = deepcopy(trainer.state_dict())
    restored = wide_trainer("continuous")
    restored.load_state_dict(checkpoint)
    actual = restored.step(rows * 0.9, collect_stats=True)
    assert_same(expected, actual)
    assert_same(expected_state, restored.state_dict())
    incompatible = wide_trainer("auto")
    with pytest.raises(ValueError, match="penalty_options"):
        incompatible.load_state_dict(checkpoint)

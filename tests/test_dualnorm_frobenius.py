"""Software contracts for the explicit large-CUDA Frobenius optimizer rule.

These checks establish its declared law and compatibility, not GAN quality.
Separate Forge studies measure trained gates through the public API.
"""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.configuration_search import recipe_identity_fields
from experiments.forge.state import state_digest
from experiments.forge.techniques import recipe_field_active, validate_same_technique
from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.optim import dualnorm
from particlegan.optim import convolution_quintic
from particlegan.optim.convolution_frobenius import frobenius_direction


@pytest.fixture(autouse=True)
def cuda_contract():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for optimizer dispatch verification")
    with torch.device("cuda:0"), torch.autograd.set_multithreading_enabled(False):
        yield


def configuration(**overrides):
    return get_recipe("bcap", **{
        "constraint_geometry_mode": "none", "optimizer_convolution": "per_offset",
        "optimizer_convolution_rule": "frobenius", "optimizer_smoothing": .001,
        **overrides,
    })


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("smoothing", [0., .001])
def test_matrix_law_zero_rank_deficiency_and_operator_bound(dtype, smoothing):
    matrices = torch.stack([torch.diag(torch.tensor([.003, .004, 0.])), torch.zeros(3, 3),
                            torch.diag(torch.tensor([1., 1e-9, 0.]))]).to(dtype)
    actual = frobenius_direction(matrices, smoothing=smoothing)
    oracle = matrices.double()
    denominator = torch.hypot(oracle.norm(dim=(-2, -1), keepdim=True),
                              oracle.new_tensor(smoothing))
    expected = oracle / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    torch.testing.assert_close(actual, expected.to(dtype), rtol=.01 if dtype == torch.bfloat16 else .001,
                               atol=1e-7)
    assert actual.dtype == dtype and actual.device == matrices.device
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.linalg.matrix_norm(actual.double(), ord=2).max() <= 1.004


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_finite_extreme_scales_do_not_overflow_normalization(scale):
    matrices = torch.tensor([[[scale, -2 * scale], [3 * scale, 0.]]])
    actual = frobenius_direction(matrices, smoothing=.001)
    reference = matrices.double()
    reference /= torch.hypot(reference.norm(dim=(-2, -1), keepdim=True), reference.new_tensor(.001))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference.float(), rtol=1e-6, atol=1e-35)


def test_smoothing_outside_float32_range_is_scaled_before_conversion():
    matrices = torch.full((2, 3, 4), 1e30)
    actual = frobenius_direction(matrices, smoothing=1e40)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, matrices.new_full(matrices.shape, 1e-10))


def test_frobenius_is_explicitly_smaller_than_polar_on_multiple_singular_directions():
    matrix = torch.eye(64)
    actual = frobenius_direction(matrix[None], smoothing=.001)[0]
    polar = dualnorm.polar_factor(matrix, smoothing=.001)
    assert actual.norm() < polar.norm() / 7.
    torch.testing.assert_close(actual.diag(), actual.new_full((64,), 1 / math.hypot(8., .001)))


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
@pytest.mark.parametrize("momentum", [0., .5])
def test_grouped_large_kernel_matches_independent_matrix_reference(kind, momentum):
    module = kind(66, 68, (2, 1), groups=2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = configuration(lr=.2, optimizer_momentum=momentum).make_generator_optimizer(module)
    history = torch.zeros_like(module.weight)
    expected = torch.zeros_like(module.weight)
    inputs, outputs = 33, 34
    for offset in (0., .9):
        values = torch.arange(module.weight.numel(), dtype=torch.float64).reshape_as(module.weight)
        gradient = (values + offset).sin() * .001
        history = history * momentum + gradient
        for group in range(2):
            for row in range(2):
                if kind is nn.ConvTranspose2d:
                    matrix = history[group * inputs:(group + 1) * inputs, :, row, 0].T
                else:
                    matrix = history[group * outputs:(group + 1) * outputs, :, row, 0]
                direction = matrix / torch.hypot(matrix.norm(), matrix.new_tensor(.001))
                change = -.2 * math.sqrt(outputs / inputs) / 2 * direction
                if kind is nn.ConvTranspose2d:
                    expected[group * inputs:(group + 1) * inputs, :, row, 0] += change.T
                else:
                    expected[group * outputs:(group + 1) * outputs, :, row, 0] += change
        module.weight.grad = gradient
        optimizer.step()
        torch.testing.assert_close(module.weight, expected, rtol=1e-12, atol=1e-14)
    assert optimizer.param_groups[0]["dualnorm_convolution"]["update_version"] == "per_offset_frobenius_hybrid_v1"


@pytest.mark.parametrize("rule", ["frobenius", "quintic"])
@pytest.mark.parametrize("device,channels,groups", [("cpu", 34, 1), ("cuda", 32, 1), ("cuda", 64, 2)])
def test_cpu_and_per_group_channel_boundary_preserve_bitwise_native(rule, device, channels, groups, monkeypatch):
    native = nn.Conv2d(channels, channels, 1, groups=groups, device=device)
    fast = deepcopy(native)
    old = configuration(optimizer_convolution_rule="polar").make_generator_optimizer(native)
    new = configuration(optimizer_convolution_rule=rule).make_generator_optimizer(fast)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU and per-group matrices <=32 must preserve native SVD")
    monkeypatch.setattr(dualnorm, "frobenius_direction", forbidden)
    monkeypatch.setattr(convolution_quintic, "quintic_direction", forbidden)
    for a, b in zip(native.parameters(), fast.parameters()):
        a.grad = torch.arange(a.numel(), device=device).reshape_as(a).sin()
        b.grad = a.grad.clone()
    old.step()
    new.step()
    assert all(torch.equal(a, b) for a, b in zip(native.parameters(), fast.parameters()))


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
def test_quintic_large_grouped_dispatch_restores_channel_layout(kind, monkeypatch):
    module = kind(66, 68, (2, 1), groups=2, bias=False)
    optimizer = configuration(optimizer_convolution_rule="quintic").make_generator_optimizer(module)
    before = module.weight.detach().clone()
    gradient = torch.arange(module.weight.numel()).reshape_as(module.weight).sin()
    module.weight.grad = gradient
    calls = []
    def observed(matrices, *, smoothing):
        calls.append((tuple(matrices.shape), smoothing))
        return matrices
    monkeypatch.setattr(convolution_quintic, "quintic_direction", observed)
    optimizer.step()
    assert calls == [((2, 2, 1, 34, 33), .001)]
    expected = before.add(gradient, alpha=-optimizer.param_groups[0]["lr"] * math.sqrt(68 / 66) / 2)
    assert torch.equal(module.weight, expected)
    assert optimizer.param_groups[0]["dualnorm_convolution"]["update_version"] == "per_offset_quintic_hybrid_v1"


def test_large_dispatch_never_calls_svd_and_zero_current_slice_stays_fixed(monkeypatch):
    module = nn.Conv2d(33, 33, (1, 2), bias=False)
    optimizer = configuration(optimizer_momentum=.5, optimizer_smoothing=1e-300).make_generator_optimizer(module)
    def forbidden(*args, **kwargs):
        raise AssertionError("large Frobenius kernels must not call SVD")
    monkeypatch.setattr(dualnorm, "polar_factor", forbidden)
    module.weight.grad = torch.ones_like(module.weight)
    module.weight.grad[:, :, 0, 0] = 0
    before = module.weight.detach().clone()
    optimizer.step()
    assert torch.equal(module.weight[:, :, 0, 0], before[:, :, 0, 0])
    assert torch.isfinite(module.weight).all()
    before = module.weight.detach().clone()
    module.weight.grad.zero_()
    optimizer.step()
    assert torch.equal(module.weight, before)


@pytest.mark.parametrize("rule", ["frobenius", "quintic"])
def test_dense_vector_sampled_prior_and_history_remain_bitwise_native(rule):
    native = nn.Linear(4, 3, dtype=torch.float64)
    fast = deepcopy(native)
    prior_a = nn.Parameter(torch.arange(20, dtype=torch.float64).reshape(5, 4))
    prior_b = nn.Parameter(prior_a.detach().clone())
    def build(module, prior, rule):
        return dualnorm.NormalizedOptimizer([
            dict(params=list(module.parameters()), role="generator"), dict(params=[prior], role="prior"),
        ], smoothing=.001, momentum=.5, convolution="per_offset", convolution_rule=rule)
    optimizers = [build(native, prior_a, "polar"), build(fast, prior_b, rule)]
    for offset in (0., .9):
        for a, b in zip(list(native.parameters()) + [prior_a], list(fast.parameters()) + [prior_b]):
            a.grad = (torch.arange(a.numel(), dtype=a.dtype).reshape_as(a) + offset).sin()
            b.grad = a.grad.clone()
        for optimizer, prior in zip(optimizers, (prior_a, prior_b)):
            optimizer.set_sampled_rows(prior, torch.tensor([0, 2], dtype=torch.long))
            optimizer.step()
        for a, b in zip(list(native.parameters()) + [prior_a], list(fast.parameters()) + [prior_b]):
            assert torch.equal(a, b)
            assert state_digest(optimizers[0].state[a]) == state_digest(optimizers[1].state[b])


@pytest.mark.parametrize("rule", ["frobenius", "quintic"])
@pytest.mark.parametrize("options", [dict(optimizer_family="dualnorm_D_only", optimizer_smoothing=0.),
                                     dict(optimizer_convolution="none"), dict(optimizer_svd_backend="cpu"),
                                     dict(optimizer_svd_backend="newton_schulz")])
def test_nonpolar_unsupported_recipe_and_optimizer_contracts_reject(rule, options):
    with pytest.raises(ValueError):
        configuration(optimizer_convolution_rule=rule, **options)
    mapped = {key.removeprefix("optimizer_"): value for key, value in options.items()}
    with pytest.raises(ValueError):
        dualnorm.NormalizedOptimizer([nn.Parameter(torch.eye(2))], **{
            "smoothing": .001, "convolution": "per_offset", "convolution_rule": rule, **mapped,
        })


def test_default_recipe_checkpoint_and_configuration_identity_stay_unchanged():
    recipe = configuration(optimizer_convolution_rule="polar")
    assert "optimizer_convolution_rule" not in recipe.to_dict()
    legacy = recipe.to_dict()
    expanded = {**legacy, "optimizer_convolution_rule": "polar"}
    assert recipe_identity_fields(legacy) == recipe_identity_fields(expanded)
    assert type(recipe)(**legacy).to_dict() == legacy
    module = nn.Conv2d(3, 4, 1)
    optimizer = recipe.make_generator_optimizer(module)
    saved = optimizer.state_dict()
    assert "convolution_rule" not in saved["dualnorm"]
    assert saved["param_groups"][0]["dualnorm_convolution"]["update_version"] == "per_offset_polar_v1"
    assert recipe_field_owner("optimizer_convolution_rule") == "technique"
    assert recipe_field_active("optimizer_convolution_rule", configuration())
    with pytest.raises(ValueError):
        validate_same_technique(recipe, configuration())


@pytest.mark.parametrize("rule", ["frobenius", "quintic"])
def test_public_checkpoint_restores_next_update_and_rejects_changed_rule_atomically(rule):
    def build():
        recipe = configuration(optimizer_convolution_rule=rule, num_particles=8, z_dim=2, batch_size=4, total_steps=8,
                               standardize=False, prior_kind="mog", sigma_rel=.1)
        generator = nn.Sequential(nn.Unflatten(1, (2, 1, 1)), nn.ConvTranspose2d(2, 33, 3),
                                  nn.Tanh(), nn.Conv2d(33, 1, 1))
        critic = nn.Sequential(nn.Conv2d(1, 33, 3), nn.Tanh(), nn.Flatten(), nn.Linear(33, 1))
        prior = recipe.make_prior()
        for module in (generator, critic, prior):
            deterministic_orthogonal_(module, seed=0)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator(device="cuda:0").manual_seed(0))
    trainer = build()
    batch = torch.arange(36, dtype=torch.float32).reshape(4, 1, 3, 3).sin()
    trainer.step(batch)
    saved = deepcopy(trainer.state_dict())
    trainer.step(batch)
    expected = state_digest(trainer.state_dict())
    restored = build()
    restored.load_state_dict(saved)
    restored.step(batch)
    assert state_digest(restored.state_dict()) == expected
    for mutation in ("rule", "version"):
        invalid = deepcopy(restored.state_dict())
        if mutation == "rule":
            invalid["optimizers"][0]["dualnorm"]["convolution_rule"] = "quintic" if rule == "frobenius" else "frobenius"
        else:
            group = next(g for g in invalid["optimizers"][0]["param_groups"] if "dualnorm_convolution" in g)
            group["dualnorm_convolution"]["update_version"] = "per_offset_polar_v1"
        next(iter(invalid["models"]["G"].values())).add_(1.)
        before = state_digest(restored.state_dict())
        with pytest.raises(ValueError):
            restored.load_state_dict(invalid)
        assert state_digest(restored.state_dict()) == before

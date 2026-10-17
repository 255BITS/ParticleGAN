"""Numerical software checks for the opt-in convolution polar approximation.

Quality evidence belongs to separately registered public-API Forge tasks.
These fixtures check its bounded solver, layouts and checkpoint semantics.
"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.forge.state import state_digest
from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.optim import dualnorm


@pytest.fixture(autouse=True)
def cuda_contract():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for convolution solver verification")
    with torch.device("cuda:0"), torch.autograd.set_multithreading_enabled(False):
        yield


def configuration(**overrides):
    return get_recipe("bcap", **{
        "constraint_geometry_mode": "none", "optimizer_convolution": "per_offset",
        "optimizer_svd_backend": "newton_schulz", "optimizer_smoothing": .001,
        **overrides,
    })


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_diagonal_smoothing_zero_and_rank_deficiency(dtype):
    matrices = torch.stack([
        torch.diag(torch.tensor([.003, .004, 0.])), torch.zeros(3, 3),
        torch.diag(torch.tensor([1e-8, .002, 0.])),
    ]).to(dtype)
    actual = dualnorm._newton_schulz_polar(matrices, smoothing=.001)
    expected = matrices.float() / torch.hypot(matrices.float(), matrices.new_tensor(.001).float())
    torch.testing.assert_close(actual, expected.to(dtype), rtol=5e-3, atol=2e-5)
    assert actual.dtype == dtype and actual.device == matrices.device
    assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize("shape", [(2, 6, 4), (3, 4, 6), (2, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_structured_smoothed_operator_stays_within_declared_numerical_envelope(shape, dtype):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
    matrices = (values.sin() + (values * .37).cos()) * .001
    actual = dualnorm._newton_schulz_polar(matrices, smoothing=.001)
    expected = torch.stack([dualnorm.polar_factor(m, smoothing=.001) for m in matrices])
    # The sinusoidal fixture is low rank. The native hard cutoff and rounded
    # Gram matrix deliberately have distinct null-space treatment.
    assert (actual - expected).norm() / expected.norm() < 1e-3
    assert (actual - expected).abs().amax() < 1e-4


def test_successful_fast_branch_omits_native_hard_rank_mask_explicitly():
    matrix = torch.diag(torch.tensor([1., 1e-9]))
    actual = dualnorm._newton_schulz_polar(matrix[None], smoothing=.001)[0]
    native = dualnorm.polar_factor(matrix, smoothing=.001)
    assert native[1, 1] == 0
    torch.testing.assert_close(actual[1, 1], actual.new_tensor(1e-6), rtol=1e-4, atol=1e-10)


def test_unconverged_slice_falls_back_to_exact_svd_without_replacing_healthy_slice(monkeypatch):
    healthy = torch.diag(torch.tensor([.001, .002, .003], dtype=torch.float64))
    difficult = torch.diag(torch.tensor([1e12, 0., 0.], dtype=torch.float64))
    calls = []
    original = dualnorm.polar_factor

    def observed(matrix, **options):
        calls.append(matrix.detach().clone())
        return original(matrix, **options)

    monkeypatch.setattr(dualnorm, "polar_factor", observed)
    actual = dualnorm._newton_schulz_polar(torch.stack([healthy, difficult]), smoothing=.001)
    assert len(calls) == 1 and torch.equal(calls[0], difficult)
    assert torch.equal(actual[1], original(difficult, smoothing=.001))
    torch.testing.assert_close(actual[0], original(healthy, smoothing=.001), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_extreme_finite_scales_do_not_overflow_gram(scale):
    matrix = torch.diag(torch.tensor([scale, 2 * scale, 0.]))
    actual = dualnorm._newton_schulz_polar(matrix[None], smoothing=.001)[0]
    expected = dualnorm.polar_factor(matrix, smoothing=.001)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
@pytest.mark.parametrize("channels,kernel,groups", [((6, 4), (2, 3), 2), ((4, 6), (1, 1), 2),
                                                   ((3, 2), (3, 2), 1)])
def test_grouped_kernel_layout_and_momentum_match_native(kind, channels, kernel, groups):
    native = kind(*channels, kernel, groups=groups, bias=False, dtype=torch.float64)
    fast = deepcopy(native)
    native_optimizer = configuration(optimizer_svd_backend="native", optimizer_momentum=.5).make_generator_optimizer(native)
    fast_optimizer = configuration(optimizer_momentum=.5).make_generator_optimizer(fast)
    values = torch.arange(native.weight.numel(), dtype=torch.float64).reshape(native.weight.shape)
    for offset in (0., .9):
        gradient = ((values + offset).sin() + (values * .37 + offset).cos()) * .001
        native.weight.grad, fast.weight.grad = gradient, gradient.clone()
        native_optimizer.step()
        fast_optimizer.step()
        torch.testing.assert_close(fast.weight, native.weight, rtol=1e-11, atol=1e-12)
        assert torch.equal(fast_optimizer.state[fast.weight]["momentum_buffer"],
                           native_optimizer.state[native.weight]["momentum_buffer"])


def test_zero_current_slice_never_moves_on_stale_momentum():
    module = nn.ConvTranspose2d(1, 1, (1, 3), bias=False, dtype=torch.float64)
    optimizer = configuration(optimizer_momentum=.5).make_generator_optimizer(module)
    module.weight.grad = module.weight.new_tensor([.002, 5e-9, 0.]).reshape_as(module.weight)
    optimizer.step()
    before = module.weight.detach().clone()
    module.weight.grad = torch.zeros_like(module.weight)
    optimizer.step()
    assert torch.equal(module.weight, before)


def test_all_dense_vector_and_sampled_prior_paths_are_bitwise_native():
    native = nn.Linear(4, 3, dtype=torch.float64)
    fast = deepcopy(native)
    prior_a = nn.Parameter(torch.arange(20, dtype=torch.float64).reshape(5, 4))
    prior_b = nn.Parameter(prior_a.detach().clone())
    def build(module, prior, backend):
        return dualnorm.NormalizedOptimizer([
            dict(params=list(module.parameters()), role="generator"),
            dict(params=[prior], role="prior"),
        ], smoothing=.001, momentum=.5, convolution="per_offset", svd_backend=backend)
    optimizers = [build(native, prior_a, "native"), build(fast, prior_b, "newton_schulz")]
    for offset in (0., .9, 1.7):
        for a, b in zip(list(native.parameters()) + [prior_a], list(fast.parameters()) + [prior_b]):
            values = torch.arange(a.numel(), dtype=a.dtype).reshape_as(a)
            a.grad = (values + offset).sin()
            b.grad = a.grad.clone()
        for optimizer, prior in zip(optimizers, [prior_a, prior_b]):
            optimizer.set_sampled_rows(prior, torch.tensor([0, 2], dtype=torch.long))
            optimizer.step()
        for a, b in zip(list(native.parameters()) + [prior_a], list(fast.parameters()) + [prior_b]):
            assert torch.equal(a, b)
            assert state_digest(optimizers[0].state[a]) == state_digest(optimizers[1].state[b])


@pytest.mark.parametrize("options", [dict(smoothing=0.), dict(convolution="none"),
                                     dict(family="dualnorm_D_only")])
def test_unsupported_solver_contracts_fail_closed(options):
    with pytest.raises(ValueError):
        dualnorm.NormalizedOptimizer([nn.Parameter(torch.eye(2))], **{
            "smoothing": .001, "convolution": "per_offset", "svd_backend": "newton_schulz", **options,
        })


def test_public_checkpoint_restores_exact_next_update_and_rejects_backend_switch_atomically():
    def build():
        recipe = configuration(num_particles=8, z_dim=2, batch_size=4, total_steps=8,
                               standardize=False, prior_kind="mog", sigma_rel=.1)
        generator = nn.Sequential(nn.Unflatten(1, (2, 1, 1)), nn.ConvTranspose2d(2, 3, 3),
                                  nn.Tanh(), nn.Conv2d(3, 1, 1))
        critic = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Tanh(), nn.Flatten(), nn.Linear(2, 1))
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
    invalid = deepcopy(restored.state_dict())
    invalid["optimizers"][0]["dualnorm"]["svd_backend"] = "native"
    next(iter(invalid["models"]["G"].values())).add_(1.)
    before = state_digest(restored.state_dict())
    with pytest.raises(ValueError):
        restored.load_state_dict(invalid)
    assert state_digest(restored.state_dict()) == before

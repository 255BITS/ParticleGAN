"""Numerical software checks for the opt-in convolution polar approximation.

Quality evidence belongs to separately registered public-API Forge tasks.
These fixtures check its bounded solver, layouts and checkpoint semantics.
"""
from copy import deepcopy
from io import BytesIO

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
    torch.testing.assert_close(actual[0], original(healthy, smoothing=.001), rtol=1e-5, atol=1e-6)


def test_augmented_iteration_handles_ill_conditioned_gradient_without_svd_fallback(monkeypatch):
    generator = torch.Generator(device="cuda:0").manual_seed(0)
    left = torch.linalg.qr(torch.randn(64, 32, generator=generator)).Q
    right = torch.linalg.qr(torch.randn(32, 32, generator=generator)).Q
    singular = torch.logspace(-.7, -8, 32)
    matrix = (left * singular) @ right.T
    expected = dualnorm.polar_factor(matrix, smoothing=.001)
    # A float64 SVD without the hard rank mask is an independent oracle for
    # the mathematical function used by the successful approximate branch.
    left64, singular64, right64 = torch.linalg.svd(matrix.double(), full_matrices=False)
    unmasked = ((left64 * (singular64 / torch.hypot(singular64, singular64.new_tensor(.001))))
                @ right64).float()
    calls = []
    original = dualnorm.polar_factor
    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(dualnorm, "polar_factor", observed)
    actual = dualnorm._newton_schulz_polar(matrix[None], smoothing=.001)[0]
    assert calls == []
    assert (actual - expected).norm() / expected.norm() < 1e-3
    assert (actual - unmasked).norm() / unmasked.norm() < 1e-4
    assert (actual - unmasked).abs().amax() < 1e-4


def test_extended_iteration_bound_converges_slow_regularized_null_block_without_fallback(monkeypatch):
    matrix = torch.zeros(64, 64)
    matrix[0, 0] = 10.
    def forbidden(*args, **kwargs):
        raise AssertionError("the extended 32-iteration bound should avoid an SVD fallback")
    monkeypatch.setattr(dualnorm, "polar_factor", forbidden)
    actual = dualnorm._newton_schulz_polar(matrix[None], smoothing=.001)[0]
    expected = torch.zeros_like(matrix)
    expected[0, 0] = 10 / (100 + .001**2)**.5
    torch.testing.assert_close(actual, expected)


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


def test_small_kernel_dispatch_is_bitwise_native(monkeypatch):
    native = nn.Conv2d(12, 12, 3)
    fast = deepcopy(native)
    old = configuration(optimizer_svd_backend="native").make_generator_optimizer(native)
    new = configuration().make_generator_optimizer(fast)
    def forbidden(*args, **kwargs):
        raise AssertionError("small convolutions must retain native SVD")
    monkeypatch.setattr(dualnorm, "_newton_schulz_polar", forbidden)
    native.weight.grad = torch.arange(native.weight.numel()).reshape_as(native.weight).sin()
    native.bias.grad = torch.ones_like(native.bias)
    fast.weight.grad, fast.bias.grad = native.weight.grad.clone(), native.bias.grad.clone()
    old.step()
    new.step()
    assert torch.equal(native.weight, fast.weight) and torch.equal(native.bias, fast.bias)


def test_cpu_kernel_dispatch_is_bitwise_native(monkeypatch):
    native = nn.Conv2d(64, 64, 1, bias=False, device="cpu")
    fast = deepcopy(native)
    old = configuration(optimizer_svd_backend="native").make_generator_optimizer(native)
    new = configuration().make_generator_optimizer(fast)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU convolutions must retain native SVD")
    monkeypatch.setattr(dualnorm, "_newton_schulz_polar", forbidden)
    native.weight.grad = torch.arange(native.weight.numel(), device="cpu").reshape_as(native.weight).sin()
    fast.weight.grad = native.weight.grad.clone()
    old.step()
    new.step()
    assert torch.equal(native.weight, fast.weight)


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
def test_large_grouped_kernel_uses_solver_with_correct_channel_layout(kind, monkeypatch):
    native = kind(64, 128, (2, 1), groups=2, bias=False)
    fast = deepcopy(native)
    old = configuration(optimizer_svd_backend="native").make_generator_optimizer(native)
    new = configuration().make_generator_optimizer(fast)
    generator = torch.Generator(device="cuda:0").manual_seed(0)
    native.weight.grad = torch.randn(native.weight.shape, generator=generator) * .001
    fast.weight.grad = native.weight.grad.clone()
    original = dualnorm._newton_schulz_polar
    calls = []
    def observed(matrices, **kwargs):
        calls.append(tuple(matrices.shape))
        return original(matrices, **kwargs)
    monkeypatch.setattr(dualnorm, "_newton_schulz_polar", observed)
    old.step()
    new.step()
    assert calls == [(2, 2, 1, 64, 32)]
    torch.testing.assert_close(native.weight, fast.weight, rtol=1e-4, atol=1e-6)


def test_zero_current_slice_never_moves_on_stale_momentum():
    module = nn.ConvTranspose2d(1, 1, (1, 3), bias=False, dtype=torch.float64)
    optimizer = configuration(optimizer_momentum=.5).make_generator_optimizer(module)
    module.weight.grad = module.weight.new_tensor([.002, 5e-9, 0.]).reshape_as(module.weight)
    optimizer.step()
    before = module.weight.detach().clone()
    module.weight.grad = torch.zeros_like(module.weight)
    optimizer.step()
    assert torch.equal(module.weight, before)


def test_inactive_slice_stays_fixed_when_smoothing_underflows_computation_dtype():
    native = nn.Conv2d(64, 64, (1, 2), bias=False)
    fast = deepcopy(native)
    old = configuration(optimizer_svd_backend="native", optimizer_smoothing=1e-300).make_generator_optimizer(native)
    new = configuration(optimizer_smoothing=1e-300).make_generator_optimizer(fast)
    native.weight.grad = torch.zeros_like(native.weight)
    native.weight.grad[:, :, 0, 1] = torch.eye(64)
    fast.weight.grad = native.weight.grad.clone()
    inactive_before = fast.weight[:, :, 0, 0].detach().clone()
    old.step()
    new.step()
    assert torch.equal(fast.weight[:, :, 0, 0], inactive_before)
    assert torch.isfinite(fast.weight).all()
    torch.testing.assert_close(fast.weight, native.weight, rtol=1e-5, atol=1e-6)


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
        generator = nn.Sequential(nn.Unflatten(1, (2, 1, 1)), nn.ConvTranspose2d(2, 64, 3),
                                  nn.Tanh(), nn.Conv2d(64, 1, 1))
        critic = nn.Sequential(nn.Conv2d(1, 64, 3), nn.Tanh(), nn.Flatten(), nn.Linear(64, 1))
        prior = recipe.make_prior()
        for module in (generator, critic, prior):
            deterministic_orthogonal_(module, seed=0)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator(device="cuda:0").manual_seed(0))
    trainer = build()
    batch = torch.arange(36, dtype=torch.float32).reshape(4, 1, 3, 3).sin()
    trainer.step(batch)
    assert len(trainer.opt_g._newton_graphs.entries) == 2
    assert len(trainer.opt_g._newton_graphs.resources) == 1
    copied = deepcopy(trainer.opt_g)
    assert not hasattr(copied, "_newton_graphs")
    assert state_digest(copied.state_dict()) == state_digest(trainer.opt_g.state_dict())
    serialized = BytesIO()
    torch.save(trainer.opt_g, serialized)
    serialized.seek(0)
    unpickled = torch.load(serialized, weights_only=False)
    assert not hasattr(unpickled, "_newton_graphs")
    assert state_digest(unpickled.state_dict()) == state_digest(trainer.opt_g.state_dict())
    saved = deepcopy(trainer.state_dict())
    trainer.step(batch)
    expected = state_digest(trainer.state_dict())
    restored = build()
    restored.load_state_dict(saved)
    assert not hasattr(restored.opt_g, "_newton_graphs")
    restored.step(batch)
    assert state_digest(restored.state_dict()) == expected
    invalid = deepcopy(restored.state_dict())
    invalid["optimizers"][0]["dualnorm"]["svd_backend"] = "native"
    next(iter(invalid["models"]["G"].values())).add_(1.)
    before = state_digest(restored.state_dict())
    with pytest.raises(ValueError):
        restored.load_state_dict(invalid)
    assert state_digest(restored.state_dict()) == before


@pytest.mark.parametrize("version", [None, 1, 2, 4])
def test_previous_or_unknown_solver_checkpoint_version_is_rejected_atomically(version):
    module = nn.Conv2d(2, 64, 1)
    optimizer = configuration().make_generator_optimizer(module)
    saved = deepcopy(optimizer.state_dict())
    assert saved["dualnorm"]["solver_version"] == 3
    if version is None:
        saved["dualnorm"].pop("solver_version")
    else:
        saved["dualnorm"]["solver_version"] = version
    saved["param_groups"][0]["lr"] = 123.
    before = state_digest(optimizer.state_dict())
    with pytest.raises(ValueError, match="solver version"):
        optimizer.load_state_dict(saved)
    assert state_digest(optimizer.state_dict()) == before


def test_graph_capture_preserves_rng_caller_stream_device_and_eager_result():
    values = torch.arange(2 * 64 * 3, dtype=torch.float32).reshape(2, 64, 3)
    matrices = values.sin() * .001
    expected = dualnorm._newton_schulz_polar(matrices, smoothing=.001)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    device_before = torch.cuda.current_device()
    caller = torch.cuda.Stream()
    caller.wait_stream(torch.cuda.current_stream())
    cache = dualnorm._PolarGraphCache()
    with torch.cuda.stream(caller):
        actual = dualnorm._newton_schulz_polar(matrices, smoothing=.001, cache=cache)
        assert torch.cuda.current_stream() == caller
        assert torch.cuda.current_device() == device_before
    torch.cuda.current_stream().wait_stream(caller)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


def test_mixed_shape_shared_graph_pool_never_aliases_live_results():
    cache = dualnorm._PolarGraphCache()
    first_values = torch.arange(2 * 64 * 3, dtype=torch.float32).reshape(2, 64, 3)
    second_values = torch.arange(2 * 128 * 64, dtype=torch.float32).reshape(2, 128, 64)
    first, second = first_values.sin() * .001, second_values.cos() * .001
    actual_first = dualnorm._newton_schulz_polar(first, smoothing=.001, cache=cache)
    saved_first = actual_first.clone()
    actual_second = dualnorm._newton_schulz_polar(second, smoothing=.001, cache=cache)
    saved_second = actual_second.clone()
    assert torch.equal(actual_first, saved_first)
    expected_first = dualnorm._newton_schulz_polar(first * 2, smoothing=.001)
    replayed_first = dualnorm._newton_schulz_polar(first * 2, smoothing=.001, cache=cache)
    assert torch.equal(replayed_first, expected_first)
    assert torch.equal(actual_second, saved_second)
    assert len(cache.entries) == 2 and len(cache.resources) == 1

"""CPU numerical-kernel/API contracts, not GAN acquisition experiments.

The oracle uses FP64 SVD of the stored FP32 input, original shape/epsilon32,
strict s>tau and absolute damping .001. Accuracy is not default-SVD bit parity.
All fixtures are fixed analytic tensors or use a private seed0 CPU generator;
no model training, CUDA queries, hardware calls, payloads or GIFs are involved.
"""
from copy import deepcopy
import math

import pytest
import torch

from particlegan import get_recipe
from particlegan.optim import GramGuardError, guarded_gram_polar
from particlegan.optim import dualnorm
from particlegan.optim.dualnorm import NormalizedOptimizer, make_normalized_optimizer


PROFILE = "gram_fp64_guarded_v1"


def oracle(matrix):
    value = matrix.double()
    if not value.numel():
        return torch.zeros_like(value)
    u, singular, vh = torch.linalg.svd(value, full_matrices=False)
    cutoff = max(matrix.shape) * torch.finfo(torch.float32).eps * singular[0]
    weights = (singular > cutoff) * singular / torch.hypot(
        singular, singular.new_tensor(.001))
    return (u * weights) @ vh


def assert_accuracy(actual, expected):
    # Explicit CPU operator metric; neither native FP32 SVD nor bit parity.
    torch.testing.assert_close(actual.double(), expected.double(), rtol=2e-6, atol=2e-7)


def same_tree(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.detach().contiguous().view(torch.uint8),
                           right.detach().contiguous().view(torch.uint8))
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            same_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_tree(a, b)
    else:
        assert left == right


def optimizer(parameter, *, profile=PROFILE, **kwargs):
    options = dict(family="dualnorm", smoothing=.001, momentum=0.,
                   svd_backend="native", lr=.01, dense_profile=profile)
    options.update(kwargs)
    return NormalizedOptimizer([parameter], **options)


@pytest.mark.parametrize("values", [[3., .004, 0.], [1., 1e-4, 0.], [-2., .03, 0.]])
def test_public_diagonal_damping_and_rank_are_analytic(values):
    matrix = torch.diag(torch.tensor(values, dtype=torch.float32, device="cpu"))
    result = guarded_gram_polar(matrix, smoothing=.001)
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert result.shape == matrix.shape and not result.requires_grad
    assert_accuracy(result, oracle(matrix))
    assert result[-1, -1] == 0  # No invented null-space polar direction.


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (4, 4)])
@pytest.mark.parametrize("scale", [1e-4, 1., 1e3])
def test_rectangular_and_scaled_inputs_match_original_fp32_cutoff_oracle(shape, scale):
    generator = torch.Generator(device="cpu").manual_seed(0)
    rows, columns = shape
    size = min(shape)
    left, _ = torch.linalg.qr(torch.randn(rows, size, generator=generator, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(columns, size, generator=generator, dtype=torch.float64))
    spectrum = torch.linspace(2., .02, size, dtype=torch.float64)
    matrix = ((left * spectrum) @ right.T).float() * scale
    result = guarded_gram_polar(matrix)
    assert_accuracy(result, oracle(matrix))


def test_noncontiguous_strided_input_and_sign_are_supported():
    matrix = torch.tensor([[1., .2, 0.], [-.3, .7, .1]], device="cpu").T
    assert not matrix.is_contiguous()
    assert_accuracy(guarded_gram_polar(matrix), oracle(matrix))
    assert_accuracy(guarded_gram_polar(-matrix), -oracle(matrix))


def test_absolute_damping_is_not_lost_by_input_scaling():
    matrix = torch.diag(torch.tensor([.001, .0001], device="cpu"))
    one = guarded_gram_polar(matrix)
    ten = guarded_gram_polar(matrix * 10)
    assert_accuracy(one, oracle(matrix))
    assert_accuracy(ten, oracle(matrix * 10))
    assert float((one - ten).norm()) > .5
    # A common output renormalization cannot restore both fixed-lambda gains.
    assert abs(float(one[0, 0] / ten[0, 0] - one[1, 1] / ten[1, 1])) > .5


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (0, 3), (3, 0), (0, 0)])
def test_zero_and_empty_fp32_inputs_return_zero(shape):
    matrix = torch.zeros(shape, dtype=torch.float32, device="cpu")
    result = guarded_gram_polar(matrix)
    assert result.shape == matrix.shape and result.dtype == matrix.dtype
    assert result.device == matrix.device and torch.equal(result, matrix)


@pytest.mark.parametrize("adjacent", ["below", "tie", "above"])
def test_exact_and_adjacent_fp32_cutoff_are_terminal_ambiguity(adjacent, monkeypatch):
    cutoff = torch.tensor(2 * torch.finfo(torch.float32).eps, device="cpu")
    tail = cutoff
    if adjacent == "below":
        tail = torch.nextafter(cutoff, cutoff.new_tensor(0.))
    elif adjacent == "above":
        tail = torch.nextafter(cutoff, cutoff.new_tensor(1.))
    matrix = torch.diag(torch.stack((cutoff.new_tensor(1.), tail)))
    target = oracle(matrix)
    assert (target[1, 1] > 0) == (adjacent == "above")
    before = matrix.clone()

    def no_fallback(*args, **kwargs):
        raise AssertionError("guarded Gram must not fall back to SVD")

    monkeypatch.setattr(torch.linalg, "svd", no_fallback)
    with pytest.raises(GramGuardError):
        guarded_gram_polar(matrix)
    same_tree(matrix, before)


def test_original_eps32_drops_a_mode_retained_by_eps64():
    matrix = torch.diag(torch.tensor([1., 1e-8], device="cpu"))
    result = guarded_gram_polar(matrix)
    assert result[1, 1] == 0
    assert matrix[1, 1] > 2 * torch.finfo(torch.float64).eps
    assert_accuracy(result, oracle(matrix))


def test_unreliable_eigensolve_is_terminal_without_native_fallback(monkeypatch):
    original_eigh = torch.linalg.eigh

    def corrupt_eigh(*args, **kwargs):
        eigen, vectors = original_eigh(*args, **kwargs)
        eigen = eigen.clone()
        eigen[0] = -1.  # Deliberately inconsistent decomposition/negative mode.
        return eigen, vectors

    def no_fallback(*args, **kwargs):
        raise AssertionError("numerical guard must not invoke SVD")

    monkeypatch.setattr(torch.linalg, "eigh", corrupt_eigh)
    monkeypatch.setattr(torch.linalg, "svd", no_fallback)
    with pytest.raises(GramGuardError):
        guarded_gram_polar(torch.diag(torch.tensor([1., .2], device="cpu")))


def test_backend_failure_is_terminal_without_native_fallback(monkeypatch):
    def failed_eigh(*args, **kwargs):
        raise RuntimeError("injected numerical backend failure")

    def no_fallback(*args, **kwargs):
        raise AssertionError("backend failure must not invoke SVD")

    monkeypatch.setattr(torch.linalg, "eigh", failed_eigh)
    monkeypatch.setattr(torch.linalg, "svd", no_fallback)
    with pytest.raises(RuntimeError, match="injected numerical backend failure"):
        guarded_gram_polar(torch.eye(2, device="cpu"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64,
                                  torch.int64, torch.bool, torch.complex64])
def test_helper_rejects_non_fp32_dtype(dtype):
    with pytest.raises(ValueError):
        guarded_gram_polar(torch.ones(2, 2, dtype=dtype, device="cpu"))


@pytest.mark.parametrize("matrix", [None, [[1., 0.], [0., 1.]],
    torch.tensor(1., device="cpu"), torch.ones(2, device="cpu"), torch.ones(2, 2, 2, device="cpu"),
    torch.ones(2, 2, device="cpu", requires_grad=True),
    torch.tensor([[float("nan"), 0.], [0., 1.]], device="cpu"),
    torch.tensor([[float("inf"), 0.], [0., 1.]], device="cpu"),
    torch.eye(2, device="cpu").to_sparse(), torch.empty(2, 2, device="meta")])
def test_helper_rejects_invalid_shape_layout_device_or_values(matrix):
    with pytest.raises(ValueError):
        guarded_gram_polar(matrix)


@pytest.mark.parametrize("smoothing", [True, False, 0, 1, 0., -.001, .01,
    float("nan"), float("inf"), ".001", torch.tensor(.001, device="cpu"),
    math.nextafter(.001, math.inf)])
def test_helper_requires_exact_typed_fixed_smoothing(smoothing):
    with pytest.raises(ValueError):
        guarded_gram_polar(torch.eye(2, device="cpu"), smoothing=smoothing)


def test_helper_preserves_input_rng_and_cpu_runtime_settings():
    matrix = torch.tensor([[1., .2], [-.3, .7]], device="cpu")
    before = matrix.clone()
    version = matrix._version
    rng = torch.get_rng_state().clone()
    settings = (torch.get_default_dtype(), torch.get_default_device(), torch.get_num_threads(),
                torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
                torch.is_autocast_enabled("cpu"), torch.is_grad_enabled())
    assert_accuracy(guarded_gram_polar(matrix), oracle(matrix))
    same_tree(matrix, before)
    assert matrix._version == version and torch.equal(torch.get_rng_state(), rng)
    assert settings == (torch.get_default_dtype(), torch.get_default_device(), torch.get_num_threads(),
                        torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
                        torch.is_autocast_enabled("cpu"), torch.is_grad_enabled())


def test_selected_optimizer_routes_only_eligible_dense_and_keeps_eps_clock(monkeypatch):
    calls = []

    def fixed_direction(matrix, *, smoothing):
        calls.append((matrix.clone(), smoothing))
        return torch.eye(2, device="cpu")

    monkeypatch.setattr(dualnorm, "guarded_gram_polar", fixed_direction)
    dense = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    tiny = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    vector = torch.nn.Parameter(torch.zeros(2, device="cpu"))
    opt = NormalizedOptimizer([dense, tiny, vector], family="dualnorm", momentum=0.,
        smoothing=.001, dense_profile=PROFILE, lr=.01)
    dense.grad = torch.diag(torch.tensor([1., .2], device="cpu"))
    tiny.grad = torch.full_like(tiny, 1e-12)
    vector.grad = torch.tensor([.3, -.4], device="cpu")
    opt.step()
    assert len(calls) == 1 and calls[0][1] == .001
    torch.testing.assert_close(dense, -.01 * torch.eye(2, device="cpu"), rtol=0, atol=0)
    assert torch.count_nonzero(tiny) == 0
    expected = -.01 * vector.grad / torch.hypot(vector.grad.norm(), vector.grad.new_tensor(.001))
    torch.testing.assert_close(vector, expected)
    assert all(opt.state[p]["step"] == 1 for p in (dense, tiny, vector))


def test_default_dispatch_never_calls_guarded_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default profile changed dispatch")

    monkeypatch.setattr(dualnorm, "guarded_gram_polar", forbidden)
    parameter = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    opt = optimizer(parameter, profile="default")
    parameter.grad = torch.diag(torch.tensor([1., .2], device="cpu"))
    opt.step()
    assert "dense_profile" not in opt.state_dict()["dualnorm"]


def test_selected_optimizer_does_not_hide_an_active_numerical_hold(monkeypatch):
    cutoff = 2 * torch.finfo(torch.float32).eps
    parameter = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    opt = optimizer(parameter)
    parameter.grad = torch.diag(torch.tensor([1., cutoff], device="cpu"))
    before = parameter.detach().clone()

    def no_fallback(*args, **kwargs):
        raise AssertionError("selected optimizer must not hide a Gram HOLD with SVD")

    monkeypatch.setattr(torch.linalg, "svd", no_fallback)
    with pytest.raises(GramGuardError):
        opt.step()
    same_tree(parameter.detach(), before)
    # The public optimizer clock may have advanced before the terminal error;
    # this disposable object must not be continued as if rollback were promised.


def test_vector_prior_and_annotated_convolution_keep_existing_public_rule(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("profile escaped ordinary dense matrix scope")

    monkeypatch.setattr(dualnorm, "guarded_gram_polar", forbidden)
    metadata = dict(update_version="per_offset_polar_v1", layout="conv2d_out_in",
                    groups=1, in_channels=2, out_channels=2, kernel_size=[1, 1])
    outputs = []
    for profile in ("default", PROFILE):
        vector = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64, device="cpu"))
        prior = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float64, device="cpu"))
        convolution = torch.nn.Parameter(torch.zeros(2, 2, 1, 1, dtype=torch.float64, device="cpu"))
        groups = [dict(params=[vector], role="generator"), dict(params=[prior], role="prior"),
                  dict(params=[convolution], role="generator", dualnorm_convolution=deepcopy(metadata))]
        opt = NormalizedOptimizer(groups, family="dualnorm", smoothing=.001, momentum=0.,
                                  convolution="per_offset", dense_profile=profile, lr=.01)
        vector.grad = torch.tensor([1., -2.], dtype=torch.float64, device="cpu")
        prior.grad = torch.tensor([[1., 2.], [3., 4.], [0., -1.]], dtype=torch.float64, device="cpu")
        convolution.grad = torch.diag(torch.tensor([1., .2], dtype=torch.float64, device="cpu")).reshape_as(convolution)
        opt.set_sampled_rows(prior, torch.tensor([0, 2], dtype=torch.long, device="cpu"))
        opt.step()
        outputs.append([p.detach().clone() for p in (vector, prior, convolution)])
    same_tree(outputs[0], outputs[1])


@pytest.mark.parametrize("options", [dict(family="sgda"), dict(family="dualnorm_D_only"),
    dict(momentum=.5), dict(smoothing=0.), dict(smoothing=.01), dict(svd_backend="cpu"),
    dict(svd_backend="newton_schulz"), dict(dense_profile="unknown"), dict(dense_profile=True)])
def test_constructor_rejects_incompatible_or_unknown_profile_configuration(options):
    with pytest.raises(ValueError):
        optimizer(torch.nn.Parameter(torch.zeros(2, 2, device="cpu")), **options)


def test_selected_eligible_dense_requires_fp32_at_construction():
    with pytest.raises(ValueError):
        optimizer(torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64, device="cpu")))


@pytest.mark.parametrize("backend", ["native", "batched"])
def test_same_profile_public_checkpoint_restores_exact_next_cpu_update(backend):
    parameter = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    convolution = "per_offset" if backend == "batched" else "none"
    opt = optimizer(parameter, svd_backend=backend, convolution=convolution)
    parameter.grad = torch.diag(torch.tensor([1., .2], device="cpu"))
    opt.step()
    saved = deepcopy(opt.state_dict())
    assert saved["dualnorm"]["dense_profile"] == PROFILE
    other_parameter = torch.nn.Parameter(parameter.detach().clone())
    other = optimizer(other_parameter, svd_backend=backend, convolution=convolution)
    other.load_state_dict(saved)
    same_tree(other.state_dict(), saved)
    for p in (parameter, other_parameter):
        p.grad = torch.tensor([[.4, .1], [-.2, .7]], device="cpu")
    opt.step()
    other.step()
    same_tree(parameter.detach(), other_parameter.detach())
    same_tree(opt.state_dict(), other.state_dict())


@pytest.mark.parametrize("source_profile,target_profile", [("default", PROFILE), (PROFILE, "default")])
def test_cross_profile_checkpoint_rejection_is_atomic(source_profile, target_profile):
    source_p = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    source = optimizer(source_p, profile=source_profile)
    source_p.grad = torch.diag(torch.tensor([1., .2], device="cpu"))
    source.step()
    target_p = torch.nn.Parameter(torch.full((2, 2), .5, device="cpu"))
    target = optimizer(target_p, profile=target_profile)
    before_state, before_parameter = deepcopy(target.state_dict()), target_p.detach().clone()
    with pytest.raises(ValueError):
        target.load_state_dict(deepcopy(source.state_dict()))
    same_tree(target.state_dict(), before_state)
    same_tree(target_p.detach(), before_parameter)


def test_recipe_profile_is_explicit_default_omitted_and_factory_consumes_selection():
    base = get_recipe("bcap", optimizer_family="dualnorm", optimizer_smoothing=.001,
                      optimizer_svd_backend="batched")
    assert base.optimizer_dense_profile == "default"
    assert "optimizer_dense_profile" not in base.to_dict()
    selected = base.replace(optimizer_dense_profile=PROFILE)
    assert selected.to_dict()["optimizer_dense_profile"] == PROFILE
    parameter = torch.nn.Parameter(torch.zeros(2, 2, device="cpu"))
    opt = make_normalized_optimizer(selected, [dict(params=[parameter], role="generator")])
    assert opt.dense_profile == PROFILE and opt.state_dict()["dualnorm"]["dense_profile"] == PROFILE
    assert "optimizer_dense_profile" not in selected.replace(optimizer_dense_profile="default").to_dict()


@pytest.mark.parametrize("options", [dict(optimizer_family="sgda"), dict(optimizer_family="dualnorm_D_only"),
    dict(optimizer_momentum=.5), dict(optimizer_smoothing=0.), dict(optimizer_svd_backend="cpu"),
    dict(optimizer_svd_backend="newton_schulz"), dict(optimizer_dense_profile="unknown")])
def test_recipe_rejects_incompatible_dense_profile(options):
    configured = dict(optimizer_family="dualnorm", optimizer_smoothing=.001,
                      optimizer_dense_profile=PROFILE, optimizer_svd_backend="native")
    configured.update(options)
    with pytest.raises(ValueError):
        get_recipe("bcap", **configured)

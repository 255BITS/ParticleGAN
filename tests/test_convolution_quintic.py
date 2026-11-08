"""Software oracles for the fixed bounded filter; these are not quality gates."""
import math

import pytest
import torch

from particlegan.optim.convolution_quintic import (
    COEFFICIENTS, ITERATIONS, OUTPUT_SCALE, SEMANTIC_VERSION, quintic_direction,
)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda:0" if request.param == "cuda" else "cpu")


def spectral_oracle(matrix, smoothing):
    """Evaluate the scalar filter through independent float64 SVD coordinates."""
    value = matrix.double()
    left, singular, right = torch.linalg.svd(value, full_matrices=False)
    scale = (singular.square().sum() + smoothing**2).sqrt()
    if not bool(scale):
        return torch.zeros_like(matrix)
    augmented = torch.hypot(singular, singular.new_tensor(smoothing))
    filtered = augmented / scale
    for _ in range(5):
        filtered = 3.4445 * filtered - 4.775 * filtered**3 + 2.0315 * filtered**5
    ratio = torch.where(augmented > 0, singular / augmented, 0.)
    return ((left * (ratio * filtered / 1.21)) @ right).to(matrix)


def test_declared_law_and_scalar_interval_bound():
    assert SEMANTIC_VERSION == "per_offset_quintic_hybrid_v1"
    assert COEFFICIENTS == (3.4445, -4.775, 2.0315)
    assert ITERATIONS == 5 and OUTPUT_SCALE == 1.21
    # All extrema of the odd quintic on this positive interval occur at its
    # endpoints or the two positive roots of p'(s)=a+3bs²+5cs⁴.
    a, b, c = 3.4445, -4.775, 2.0315
    upper = 1.202369
    discriminant = (3 * b)**2 - 20 * c * a
    critical = [math.sqrt((-3 * b + sign * math.sqrt(discriminant)) / (10 * c))
                for sign in (-1, 1)]
    values = [a * s + b * s**3 + c * s**5 for s in [0., upper, *critical]]
    assert min(values) >= 0 and max(values) <= upper
    assert upper / OUTPUT_SCALE < 1


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (8, 8)])
@pytest.mark.parametrize("smoothing", [0., .001, .3])
def test_svd_scalar_filter_oracle_and_operator_bound(device, shape, smoothing):
    values = torch.arange(math.prod(shape), dtype=torch.float32, device=device).reshape(shape)
    matrix = (values.sin() + (.37 * values).cos()) * .01
    actual = quintic_direction(matrix, smoothing=smoothing)
    expected = spectral_oracle(matrix, smoothing)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-6)
    assert torch.linalg.svdvals(actual.double())[0] <= 1.0001


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_batched_rank_deficient_and_zero_matrices_keep_shape_dtype_and_device(device, dtype):
    matrices = torch.stack([
        torch.diag(torch.tensor([.003, .004, 0.], device=device)),
        torch.zeros(3, 3, device=device),
        torch.diag(torch.tensor([1., 1e-9, 0.], device=device)),
    ]).to(dtype)
    actual = quintic_direction(matrices, smoothing=.001)
    expected = torch.stack([spectral_oracle(matrix.float(), .001) for matrix in matrices]).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-5)
    assert actual.shape == matrices.shape and actual.dtype == dtype and actual.device == device
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.isfinite(actual).all()


def test_kernel_batch_dimensions_and_noncontiguous_layout(device):
    values = torch.arange(2 * 3 * 7 * 4, device=device, dtype=torch.float32).reshape(2, 3, 7, 4)
    matrices = values.sin().transpose(-1, -2)
    actual = quintic_direction(matrices, smoothing=.001)
    expected = torch.stack([spectral_oracle(matrix, .001)
                            for matrix in matrices.reshape(-1, 4, 7)]).reshape_as(matrices)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-6)


@pytest.mark.parametrize("dtype,scale,smoothing", [
    (torch.float32, 1e30, .001), (torch.float32, 1e-30, .001),
    (torch.float32, 1e30, 1e300), (torch.float32, 0., 1e-300),
    (torch.float64, 1e300, 1e299), (torch.float64, 1e-300, 1e-301),
])
def test_extreme_finite_input_scaling_remains_finite(device, dtype, scale, smoothing):
    matrix = torch.diag(torch.tensor([scale, -scale / 2, 0.], device=device, dtype=dtype))
    actual = quintic_direction(matrix, smoothing=smoothing)
    assert torch.isfinite(actual).all()
    assert torch.linalg.svdvals(actual.double())[0] <= 1.0001
    if scale == 0:
        assert torch.count_nonzero(actual) == 0


def test_zero_smoothing_zero_matrix_and_empty_batches(device):
    for shape in [(3, 3), (0, 3, 3), (2, 0, 3)]:
        matrix = torch.zeros(shape, device=device)
        actual = quintic_direction(matrix, smoothing=0.)
        assert actual.shape == shape and torch.count_nonzero(actual) == 0
        assert torch.isfinite(actual).all()


def test_absolute_smoothing_and_joint_scale_invariance(device):
    matrix = torch.diag(torch.tensor([.001, .003, .01], device=device))
    original = quintic_direction(matrix, smoothing=.001)
    assert not torch.allclose(original, quintic_direction(matrix * 2, smoothing=.001))
    for scale in (1e-20, 1e20):
        actual = quintic_direction(matrix * scale, smoothing=.001 * scale)
        torch.testing.assert_close(actual, original, rtol=5e-4, atol=2e-6)


def test_no_rng_consumption_and_no_caller_autocast_change(device):
    values = torch.arange(28, dtype=torch.float32, device=device).reshape(7, 4).sin()
    expected = quintic_direction(values, smoothing=.001)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        actual = quintic_direction(values, smoothing=.001)
        assert torch.is_autocast_enabled(device.type)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


@pytest.mark.parametrize("smoothing", [True, -1., float("nan"), float("inf")])
def test_invalid_smoothing_rejected(smoothing):
    with pytest.raises(ValueError, match="smoothing"):
        quintic_direction(torch.eye(2), smoothing=smoothing)


@pytest.mark.parametrize("matrix", [torch.ones(2, dtype=torch.float32), torch.ones(2, 2, dtype=torch.int64),
                                    torch.ones(2, 2, dtype=torch.complex64)])
def test_invalid_matrix_type_or_rank_rejected(matrix):
    with pytest.raises(ValueError, match="floating-point matrices"):
        quintic_direction(matrix, smoothing=.001)

"""Independent math checks for the one-bank noise-aware MMD diagnostic."""

import math

import torch

from reports.toy100.sample_anchor_mmd_filter import (
    gaussian_mmd_emitted,
    same_law_rest,
)


def test_single_point_gaussian_convolution_matches_closed_form():
    real = torch.zeros(1, 2, dtype=torch.float64)
    model = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    h = 1.0
    sigma = .029
    s2 = sigma * sigma
    expected = 1 + 1 / (1 + 2 * s2) - 2 / (1 + s2) * math.exp(-1 / (2 * (1 + s2)))
    actual = float(gaussian_mmd_emitted(real, model, h, sigma))
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-14)


def test_equal_full_gaussian_mixture_laws_are_stationary():
    support = torch.tensor([[-2.0, 0.5], [1.0, -1.0], [2.0, 0.5]], dtype=torch.float64)
    check = same_law_rest(support, 1.7)
    assert check['mmd2'] == 0.0
    assert check['gradient_max_abs'] < 1e-14
    assert check['equality_implies_rest']
    assert not check['converse_claim']


def test_identical_empirical_measures_rest_without_output_noise():
    real = torch.tensor([[-2.0, 0.5], [1.0, -1.0], [2.0, 0.5]], dtype=torch.float64)
    model = real.detach().clone().requires_grad_(True)
    value = gaussian_mmd_emitted(real, model, 1.7, sigma=0.0)
    gradient = torch.autograd.grad(value, model)[0]
    assert abs(float(value.detach())) < 1e-14
    assert float(gradient.abs().max()) < 1e-14


def test_autograd_matches_finite_difference_of_emitted_mmd():
    real = torch.tensor([[-.7, .4], [.2, -.1], [1.1, .3]], dtype=torch.float64)
    model = torch.tensor([[.4, 1.0], [-1.3, -.2]], dtype=torch.float64,
                         requires_grad=True)
    loss = gaussian_mmd_emitted(real, model, 1.25)
    derivative = torch.autograd.grad(loss, model)[0]
    epsilon = 1e-6
    for row, column in ((0, 0), (0, 1), (1, 0), (1, 1)):
        plus = model.detach().clone()
        minus = model.detach().clone()
        plus[row, column] += epsilon
        minus[row, column] -= epsilon
        central = (gaussian_mmd_emitted(real, plus, 1.25)
                   - gaussian_mmd_emitted(real, minus, 1.25)) / (2 * epsilon)
        assert math.isclose(float(derivative[row, column]), float(central),
                            rel_tol=1e-7, abs_tol=1e-9)

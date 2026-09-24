"""Independent checks of the frozen local MMD filter's two central formulas."""

import math

import numpy as np
import pytest
import torch

from reports.toy100.sample_anchor_local_mmd_filter import replacement_deltas
from reports.toy100.sample_anchor_local_mmd_continuation import (
    candidate_cross_sums, deltas as cumulative_deltas, j_emitted)
from reports.toy100.sample_anchor_mmd_filter import gaussian_mmd_emitted


@pytest.mark.parametrize('sigma', [0., .23])
def test_every_vectorized_donor_delta_equals_a_full_objective_recalculation(sigma):
    real = torch.tensor([[-.8, .3], [.2, 1.4], [1.7, -.6]], dtype=torch.float64)
    generated = torch.tensor([[.6, -.9], [-.4, .7]], dtype=torch.float64)
    width = 1.13
    deltas = replacement_deltas(real, generated, width, sigma)
    before = gaussian_mmd_emitted(real, generated, width, sigma)
    assert deltas.shape == (len(real), len(generated))
    for i in range(len(real)):
        for j in range(len(generated)):
            changed = generated.clone()
            changed[j] = real[i]
            actual = gaussian_mmd_emitted(real, changed, width, sigma) - before
            assert math.isclose(float(deltas[i, j]), float(actual),
                                rel_tol=0, abs_tol=3e-15)


def test_analytic_output_noise_mmd_matches_explicit_gaussian_quadrature():
    real = torch.tensor([[-.7, .3], [1.1, -.2]], dtype=torch.float64)
    generated = torch.tensor([[.25, 1.0], [-.4, -.6]], dtype=torch.float64)
    width, sigma = 1.18, .27
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    grid = np.stack(np.meshgrid(nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 2)
    quadrature_weights = (weights[:, None] * weights[None, :]).reshape(-1) / math.pi

    def expected_kernel(a, b, noise_std):
        displacement = a.numpy() - b.numpy() - math.sqrt(2) * noise_std * grid
        values = np.exp(-(displacement ** 2).sum(axis=1)/(2 * width ** 2))
        return float(np.dot(quadrature_weights, values))

    pp = sum(expected_kernel(a, b, 0.) for a in real for b in real)/len(real)**2
    pq = sum(expected_kernel(a, b, sigma) for a in real for b in generated) / (
        len(real)*len(generated))
    qq = sum(expected_kernel(a, b, math.sqrt(2)*sigma)
             for a in generated for b in generated)/len(generated)**2
    quadrature = pp + qq - 2*pq
    analytic = float(gaussian_mmd_emitted(real, generated, width, sigma))
    assert math.isclose(analytic, quadrature, rel_tol=0, abs_tol=2e-13)


def test_cumulative_donor_delta_matches_both_j_and_full_mmd_differences():
    real = torch.tensor([[-1.2, .2], [.4, .1], [.9, -1.1], [-.3, 1.3]],
                        dtype=torch.float64)
    generated = torch.tensor([[.5, -.7], [-.8, .3], [.2, .9]], dtype=torch.float64)
    width, sigma = .71, .18
    cross = candidate_cross_sums(real, width, sigma)
    table = cumulative_deltas(real, generated, width, sigma, cross)
    original_j = j_emitted(real, generated, width, sigma)
    original_mmd = gaussian_mmd_emitted(real, generated, width, sigma)
    for i in range(len(real)):
        for j in range(len(generated)):
            changed = generated.clone()
            changed[j] = real[i]
            j_change = j_emitted(real, changed, width, sigma) - original_j
            mmd_change = gaussian_mmd_emitted(real, changed, width, sigma) - original_mmd
            assert math.isclose(float(table[i, j]), float(j_change),
                                rel_tol=0, abs_tol=4e-15)
            assert math.isclose(float(table[i, j]), float(mmd_change),
                                rel_tol=0, abs_tol=4e-15)

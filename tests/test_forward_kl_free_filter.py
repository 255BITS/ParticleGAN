"""Independent algebra and numerical checks for the pure forward-KL filter."""

import math

import torch

from reports.toy100.forward_kl_free_filter import (
    cross_entropy, em_centroids, global_donor, quadrature,
)


def test_gh_rule_integrates_normalized_gaussian_second_moments():
    real = torch.tensor([[.7, -1.2], [2.1, .4]], dtype=torch.float64)
    width = .031286240422040236
    for order in (5, 9):
        locations, weights = quadrature(real, width, order)
        expectation = weights @ locations
        expected = real.mean(0)
        second = weights @ locations.square()
        expected_second = real.square().mean(0) + width**2
        torch.testing.assert_close(weights.sum(), torch.tensor(1., dtype=torch.float64),
                                   atol=1e-14, rtol=0)
        torch.testing.assert_close(expectation, expected, atol=1e-13, rtol=0)
        torch.testing.assert_close(second, expected_second, atol=1e-13, rtol=0)


def test_global_donor_is_exact_argmin_of_all_first_replacements():
    real = torch.tensor([[-2., 0.], [0., 1.], [1.7, 0.]], dtype=torch.float64)
    initial = torch.tensor([[-.4, 0.], [.2, -.1]], dtype=torch.float64)
    update = quadrature(real, .2, 5)
    audit = quadrature(real, .2, 9)
    candidates = []
    for donor in range(len(initial)):
        for index in range(len(real)):
            proposal = initial.clone()
            proposal[donor] = real[index]
            candidates.append((float(cross_entropy(*update, proposal, .09)),
                               index, donor, proposal))
    minimum = min(candidates, key=lambda item: item[0])
    selected, rows = global_donor(real, initial, *update, .09, limit=1, audit=audit)
    assert len(rows) == 1
    assert (rows[0]['real_index'], rows[0]['donor_index']) == minimum[1:3]
    torch.testing.assert_close(selected, minimum[3], atol=0, rtol=0)
    assert abs(rows[0]['predicted']-minimum[0]) < 1e-12


def test_em_monotone_and_far_atom_has_finite_logspace_centroid():
    real = torch.tensor([[-3., 0.], [-2.9, .1], [3., 0.]], dtype=torch.float64)
    initial = torch.tensor([[-3.1, .2], [100., 100.]], dtype=torch.float64)
    update = quadrature(real, .1, 5)
    audit = quadrature(real, .1, 9)
    selected, rows = em_centroids(initial, *update, .01, limit=20, audit=audit)
    assert rows
    assert bool(torch.isfinite(selected).all())
    assert all(row['after'] <= row['before']+1e-12 for row in rows)
    assert float(cross_entropy(*update, selected, .01)) < float(
        cross_entropy(*update, initial, .01))
    # The second component's naive exp(log responsibility) is all zero here.
    assert math.isfinite(rows[0]['minimum_log_component_mass'])

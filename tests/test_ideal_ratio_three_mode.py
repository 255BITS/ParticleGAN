"""Check the population-field algebra used by the three-mode diagnosis."""

import math

import torch

from reports.toy100.ideal_ratio_three_mode import (
    centered_field, exact_ring_centers, ratio_gradient, ratio_score,
)


def test_frozen_support_score_gradient_matches_autodiff():
    p = exact_ring_centers()
    q = p[torch.tensor([0, 1, 7, 0, 1, 7])].clone()
    query = torch.tensor([[2.83, .12], [1.93, 2.26]], dtype=torch.float64,
                         requires_grad=True)
    analytic = ratio_gradient(query, p, q, .11205467415507488,
                              .08831393664790402)
    automatic = torch.autograd.grad(ratio_score(query, p, q,
                                                .11205467415507488,
                                                .08831393664790402).sum(), query)[0]
    assert torch.allclose(analytic, automatic, atol=1e-10, rtol=1e-10)


def test_centered_cloud_has_tiny_missing_specific_local_signal():
    p = exact_ring_centers()
    q = p[torch.tensor([0]*4+[1]*4+[7]*4)]
    current = centered_field(q, p, math.hypot(.07, .0875),
                             math.hypot(.0119625, .0875))
    late = centered_field(q, p, .07, .029)
    assert current['max_clean_center_gradient_norm'] < 1e-80
    assert current['leading_log10_missing_specific_tangent_at_mode1'] < -140
    assert late['leading_log10_missing_specific_tangent_at_mode1'] < -1300
    assert math.isclose(current['log_ratio_empty_vs_occupied_at_midpoint'],
                        math.log(2), rel_tol=1e-5)
    assert current['local_isolated_component_curvature'] > 0
    assert current['one_sigma_fixed_d_score_ascent_toward_empty'] > 0
    assert current['one_sigma_retracked_d_partial_score_ascent_toward_empty'] < 0

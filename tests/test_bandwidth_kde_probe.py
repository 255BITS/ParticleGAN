"""Focused numerical and host-route checks for the scratch KDE rule."""


import numpy as np
import pytest

from reports.toy100.bandwidth_kde_probe import estimate_bandwidth


def test_two_distinct_points_have_analytic_loo_optimum():
    # With n=2 and d=1, LOO log likelihood is -log(h)-distance²/(2h²).
    actual = estimate_bandwidth(np.array([[0.], [2.]]))
    assert actual["width"] == pytest.approx(2., rel=1e-6)
    assert actual["duplicate_rows"] == 0


def test_duplicate_limit_requires_every_row_to_have_a_twin():
    atoms = estimate_bandwidth(np.array([[0.], [0.], [1.], [1.]]))
    assert atoms["width"] == 0.
    assert atoms["boundary"] == "all_observations_duplicated"
    mixed = estimate_bandwidth(np.array([[0.], [0.], [1.]]))
    assert mixed["width"] > 0.
    assert mixed["duplicate_rows"] == 2

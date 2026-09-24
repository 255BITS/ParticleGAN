"""Exact small counterexamples and noisy-center branch checks; no training."""

from fractions import Fraction as F
from itertools import product

from reports.toy100.anchor_coverage_geometry import majorization_step, objective, selected_branch
from reports.toy100.anchor_invariant_region import region_bound


def test_noisy_separated_centers_have_correct_assignment_and_exact_center_targets():
    true = (F(-1), F(1))
    checked = 0
    bound = region_bound(n=3, k=2, separation=2., radius=.1, epsilon=.02, delta=.001)
    assert bound["invariant_if_rest_on_ineligible_fit"] is True
    for shifts in product((F(-1, 50), F(0), F(1, 50)), repeat=2):
        centers = tuple(c+e for c, e in zip(true, shifts))
        for labels in product(range(2), repeat=3):
            if len(set(labels)) != 2:
                continue
            for errors in product((F(-1, 10), F(0), F(1, 10)), repeat=3):
                points = tuple(true[k]+e for k, e in zip(labels, errors))
                assignment, nearest = selected_branch(points, centers)
                assert all(labels[j] == k for k, j in enumerate(assignment))
                assert nearest == labels
                target, _ = majorization_step(points, centers)
                assert target == tuple(centers[k] for k in labels)
                checked += 1
    assert checked == 1458


def test_native_loss_descent_without_eligible_fit_can_leave_the_good_region():
    centers = (F(-1), F(1))
    pre = (F(-9, 10), F(11, 10), F(-9, 10))
    native = (F(-1), F(1), F(-17, 20))
    assert objective(native, centers) == F(3, 400) < objective(pre, centers) == F(1, 50)
    assert max(min(abs(y-c) for c in centers) for y in pre) == F(1, 10)
    assert max(min(abs(y-c) for c in centers) for y in native) == F(3, 20)
    # The current unconverged-fit fallback can select this lower-loss native
    # cloud. A proposed failure->rest guard would retain pre exactly instead.


def test_native_better_than_fit_requires_the_normalized_loss_radius_factor():
    centers = (F(-1), F(1))
    fit = (F(-99, 100), F(101, 100), F(-99, 100))
    native = (F(-1), F(1), F(-49, 50))
    delta = F(1, 100)
    assert objective(native, centers) == F(1, 7500) < objective(fit, centers) == 2*delta*delta
    native_error = max(min(abs(y-c) for c in centers) for y in native)
    assert native_error == 2*delta > delta
    assert native_error**2 <= 3*2*delta*delta


def test_comparison_tolerance_and_cost_error_are_explicit_in_the_bound():
    ideal = region_bound(n=12, k=8, separation=2.29, radius=.2, epsilon=.08, delta=3.1e-5)
    finite = region_bound(n=12, k=8, separation=2.29, radius=.2, epsilon=.08,
                          delta=3.1e-5, tolerance=1e-14, objective_error=1e-12)
    assert ideal["invariant_if_rest_on_ineligible_fit"] is True
    assert finite["precision_radius"] > ideal["precision_radius"]
    assert finite["distinct_anchor_radius"] < finite["precision_radius"]

"""Exact finite branch checks for the fixed-center geometry claim."""

from fractions import Fraction as F

from reports.toy100.anchor_coverage_geometry import (
    branch_is_active, branch_minimum, check_geometry, majorization_step, objective,
)


def test_all_strict_surplus_branch_stationary_points_cover_all_centers():
    result=check_geometry()
    assert result['enumerated_branches']==48
    active=[row for row in result['branches'] if row['active']]
    assert len(active)==12
    assert all(row['objective']=='0' for row in active)
    assert result['wrong_subset_majorization'][0]['point']==['-1','1/5','-1']
    assert result['wrong_subset_majorization'][1]['after']=='0'


def test_non_smooth_equal_count_stationarity_is_not_a_local_minimum():
    centers=(F(-1),F(1)); point=(F(-1),F(0))
    assert branch_is_active(point,centers,(0,1),(0,0))
    assert branch_minimum(2,centers,(0,1),(0,0))==point
    assert objective(point,centers)==1
    assert objective((F(-1),F(1,10)),centers)<1
    # Deterministic first-index tie breaking can stall at this nonminimum;
    # the strict N>K scope is substantive, not cosmetic.
    next_point,_=majorization_step(point,centers)
    assert next_point==point


def test_fixed_target_rest_and_response_need_no_clock_or_positive_step_floor():
    centers=(F(-1),F(1)); good=(F(-1),F(1),F(-1))
    assert majorization_step(good,centers)[0]==good
    perturbed=(F(-9,10),F(1),F(-1))
    recovered,receipt=majorization_step(perturbed,centers)
    assert recovered==good and receipt['after']==0 < receipt['before']

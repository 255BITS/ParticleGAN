import numpy as np

from experiments.diagnose_gym_sparse_actions import lateral_metrics


def test_lateral_missed_extra_and_wrong_direction_have_distinct_denominators():
    actions = np.array([[0, .5], [0, -.6], [0, .6], [0, .6], [0, 0]])
    expert = np.array([[0, -.6], [0, .6], [0, .6], [0, .5], [0, 0]])
    value = lateral_metrics(actions, expert, np.ones(5, bool))
    assert value["lateral_missed_count"] == 1
    assert value["lateral_extra_count"] == 1
    assert value["lateral_wrong_direction_count"] == 1
    assert value["lateral_missed_conditional_fraction"] == 1/3
    assert value["lateral_extra_conditional_fraction"] == 1/2
    assert value["lateral_wrong_direction_conditional_fraction"] == 1/3
    assert lateral_metrics(actions, expert, np.zeros(5, bool)) == {}

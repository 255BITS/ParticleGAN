import numpy as np

from experiments.diagnose_gym_control_actions import action_metrics


def test_false_on_off_denominators_and_dead_zone_boundaries():
    expert = np.array([[1, .5], [1,-.501], [-1, .501], [-1,0]], np.float32)
    learner = np.array([[0,.5], [1, -.5], [.1,.6], [-1,0]], np.float32)
    result = action_metrics(learner, expert, np.ones(2), np.ones(4, bool))
    assert result["main_false_off_count"] == result["main_false_on_count"] == 1
    assert result["main_false_off_given_expert_on"] == .5
    assert result["main_false_on_given_expert_off"] == .5
    assert result["main_false_off_fraction"] == .25
    assert result["lateral_regime_agreement"] == .75
    assert result["joint_regime_agreement"] == .25
    assert action_metrics(learner, expert, np.ones(2), np.zeros(4, bool)) == {"records":0}

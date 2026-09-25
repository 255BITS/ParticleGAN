import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("Box2D")

from experiments.collect_gym_transition import collect_episode
from experiments.probe_gym_simulator import choose_anchors, probe_anchor, verify_episodes
from lib.gym_data import make_env


def test_complete_episode_probe_and_coupled_noise():
    episode = collect_episode(91000, 200000, "test")
    verification = verify_episodes([episode], count=1)
    assert verification[0]["exact_replay"]
    assert verification[0]["contact_rows"] > 0
    env = make_env()
    try:
        report = probe_anchor(env, episode, 20, "flight", np.ones(6), noise_samples=4)
    finally:
        env.close()
    assert report["commands"]["main_boundary"]["standardized_squared_effect_vs_off"] == 0
    assert report["commands"]["main_above"]["standardized_squared_effect_vs_off"] > 0
    assert report["noise"]["mean_standardized_variance"] > 0


def test_selection_uses_only_stored_test_anchors():
    states = np.zeros((5, 8)); states[:, 1] = [.8, .8, .2, .2, .1]; states[4, 6] = 1
    episode = dict(episode_id=4, split="test", states=states, next_states=states)
    dataset = dict(episode_ids=np.array([4, 4, 4]), anchor_steps=np.array([1, 3, 4]))
    selected = choose_anchors(dataset, [episode])
    assert [(step, phase) for _, step, phase in selected] == [(1, "flight"), (3, "approach"), (4, "contact")]

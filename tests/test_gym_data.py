import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("Box2D")

from lib.gym_data import make_env, phases, replay_anchor, terrain_context


def test_terrain_matches_observation_frame():
    from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H
    env = make_env()
    try:
        env.reset(seed=19)
        terrain = terrain_context(env)
        assert terrain.shape == (11,)
        assert terrain.dtype == np.float32
        # Simulator smoothing uses .33 (not exactly 1/3), lowering the flat
        # polyline slightly below the nominal helipad_y rendering marker.
        expected_pad = (-.01 * env.unwrapped.helipad_y - LEG_DOWN / SCALE) / (VIEWPORT_H / SCALE / 2)
        np.testing.assert_allclose(terrain[4:7], expected_pad, atol=1e-6)
    finally:
        env.close()


def test_prefix_replay_restores_contacts_and_rng():
    from gymnasium.envs.box2d.lunar_lander import heuristic
    env = make_env()
    try:
        state, _ = env.reset(seed=91000)
        prefix = []
        while not np.any(state[6:]):
            action = np.asarray(heuristic(env, state), np.float32)
            state, _, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated), "Chosen test episode must reach live leg contact"
            prefix.append(action)
        action = np.array([.8, -.9], np.float32)
        expected = env.step(action)
        restored = replay_anchor(env, 91000, prefix)
        np.testing.assert_array_equal(restored, state)
        actual = env.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:4] == expected[1:4]
    finally:
        env.close()


def test_exact_engine_deadzone_boundaries_in_installed_simulator():
    env = make_env()
    try:
        def outcome(action):
            env.reset(seed=53)
            return env.step(np.asarray(action, np.float32))[0]
        off = outcome([-1, 0])
        for command in ([0, 0], [-.2, -.5], [-.1, .5]):
            np.testing.assert_array_equal(outcome(command), off)
        for command in ([1e-6, 0], [0, .500001], [0, -.500001]):
            assert np.max(np.abs(outcome(command) - off)) > 1e-5
    finally:
        env.close()


def test_phase_contact_in_successor_takes_precedence():
    states = np.zeros((3, 8), np.float32)
    states[:, 1] = [.8, .2, .8]
    following = states.copy()
    following[2, 6] = 1
    np.testing.assert_array_equal(phases(states, following), [0, 1, 2])


def test_counterfactual_siblings_share_anchor_and_behavior_is_exact():
    from experiments.collect_gym_transition import collect_episode
    from lib.gym_data import branch_episode
    episode = collect_episode(91000, 0, "train")
    data, steps, resets = branch_episode((episode, [25], 7))
    assert data["states"].shape == (4, 8)
    np.testing.assert_array_equal(data["states"], np.repeat(data["states"][:1], 4, axis=0))
    np.testing.assert_array_equal(data["next_states"][0], episode["next_states"][25])
    assert len(np.unique(data["actions"], axis=0)) == 4
    assert steps == 3 * 26 and resets == 3
    assert np.all(data["episode_ids"] == 0) and np.all(data["anchor_steps"] == 25)


def test_anchor_selection_is_without_replacement_even_when_phase_missing():
    from experiments.collect_gym_transition import select_anchors
    states = np.zeros((20, 8), np.float32)
    states[:, 1] = .8
    episodes = [dict(episode_id=8, states=states, next_states=states)]
    selection, natural = select_anchors(episodes, 15, np.random.default_rng(3))
    assert len(selection[8]) == len(set(selection[8])) == 15
    assert natural == {"flight": 20, "approach": 0, "contact": 0}

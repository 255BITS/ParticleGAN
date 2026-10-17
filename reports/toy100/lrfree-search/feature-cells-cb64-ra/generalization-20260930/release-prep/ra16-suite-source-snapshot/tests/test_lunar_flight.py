"""Physics and outcome checks for the real LunarLander Box2D variant."""
import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip("gymnasium")
pytest.importorskip("Box2D")

from lib.lunar_flight import (SCORING_VERSION, VARIANT, _outcome,
                              active_ground_contact_counts, collect_counterfactuals,
                              fast_expert, make_lunar_env, rollout_episode,
                              slow_expert, summarize_episodes, terminal_contact_diagnostic)


def test_negative_main_command_accelerates_down_and_stock_opt_out_is_off():
    def one_step(action, bidirectional):
        env = make_lunar_env(bidirectional=bidirectional)
        try:
            env.reset(seed=73)
            return env.step(np.asarray(action, dtype=np.float32))
        finally:
            env.close()

    off, _, _, _, _ = one_step([0., 0.], True)
    down, _, _, _, info = one_step([-.8, 0.], True)
    assert down[3] < off[3] - 1e-4
    assert info["downward_main_power"] == pytest.approx(.8)
    assert info["lunar_variant"] == VARIANT
    stock_off, _, _, _, _ = one_step([0., 0.], False)
    stock_negative, _, _, _, _ = one_step([-.8, 0.], False)
    np.testing.assert_array_equal(stock_off, stock_negative)


def test_expert_rollout_contains_real_transitions_and_outcome_metrics():
    slow = rollout_episode(19, slow_expert)
    fast = rollout_episode(19, fast_expert)
    for episode in (slow, fast):
        assert episode["states"].shape == (episode["steps"], 8)
        assert episode["actions"].shape == (episode["steps"], 2)
        assert episode["next_states"].shape == (episode["steps"], 8)
        assert episode["variant"] == VARIANT
        assert episode["scoring_version"] == SCORING_VERSION
        assert len(episode["terminal_contact_diagnostic"]["active_ground_contacts"]) == 2
        assert episode["outcome"] in ("successful_landing", "incomplete_landing",
                                      "off_pad_landing", "crash", "out_of_bounds", "time_limit")
        if episode["outcome"] == "successful_landing":
            assert episode["terminated"][-1]
            assert episode["contact_step"] is not None
    summary = summarize_episodes([slow, fast])
    assert summary["episodes"] == 2
    assert sum(summary["outcomes"].values()) == 2


def test_frame_stride_returns_real_rendered_frames():
    episode = rollout_episode(13, slow_expert, render=True, frame_stride=5, max_steps=11)
    assert episode["frames"].ndim == 4
    assert episode["frames"].shape[-1] == 3
    assert episode["frame_stride"] == 5
    assert len(episode["frames"]) == 2 + episode["steps"] // 5
    assert episode["truncated"][-1]


def test_fixed_paired_flights_land_safely_and_fast_uses_down_thruster():
    seeds = range(71000, 71008)
    slow = [rollout_episode(seed, slow_expert) for seed in seeds]
    fast = [rollout_episode(seed, fast_expert) for seed in seeds]
    assert all(e["outcome"] == "successful_landing" for e in slow + fast)
    assert all(np.any(e["downward_main_power"] > 0) for e in fast)
    assert sum(e["steps"] for e in fast) < 0.9 * sum(e["steps"] for e in slow)


def test_counterfactuals_replay_prefix_and_capture_main_ignition_jump():
    episode = rollout_episode(24053, slow_expert, max_steps=5)
    messages = []
    branches = collect_counterfactuals([episode], log=messages.append)
    assert len(branches["states"]) == 24  # first 3 and three-quarter anchor
    assert len(messages) == 1
    assert set(branches["action_kind"]) == {"down", "off", "up_ignition",
                                               "up_low", "up_medium", "up_full"}
    for step in range(4):
        off = (branches["anchor_steps"] == step) & (branches["action_kind"] == "off")
        ignition = (branches["anchor_steps"] == step) & (branches["action_kind"] == "up_ignition")
        np.testing.assert_array_equal(branches["states"][off], branches["states"][ignition])
        assert (branches["next_states"][ignition, 3] - branches["next_states"][off, 3])[0] > .01
    assert np.all(branches["episode_seeds"] == 24053)


def test_counterfactuals_reject_wrong_prefix_state():
    episode = rollout_episode(24053, slow_expert, max_steps=5)
    episode["states"][1, 0] += .01
    with pytest.raises(RuntimeError, match="Replay mismatch.*anchor=1"):
        collect_counterfactuals([episode])


def test_terminal_scoring_requires_live_enabled_touching_terrain_contacts():
    moon, other = object(), object()

    def edge(body=moon, *, enabled=True, touching=True):
        return SimpleNamespace(other=body, contact=SimpleNamespace(enabled=enabled, touching=touching))

    base = SimpleNamespace(game_over=False, moon=moon, helipad_x1=0., helipad_x2=1.,
                           lander=SimpleNamespace(awake=False, position=SimpleNamespace(x=.5)),
                           legs=[SimpleNamespace(ground_contact=False, contacts=[edge()]),
                                 SimpleNamespace(ground_contact=False, contacts=[edge()])])
    env = SimpleNamespace(unwrapped=base)
    state = np.zeros(8, dtype=np.float32)
    assert active_ground_contact_counts(env) == (1, 1)
    assert terminal_contact_diagnostic(env, state)["cached_leg_flags"] == [False, False]
    assert _outcome(env, state, True, False) == "successful_landing"
    base.legs[0].ground_contact = base.legs[1].ground_contact = True
    for inactive in (edge(enabled=False), edge(touching=False), edge(body=other)):
        base.legs[0].contacts = [inactive]
        assert active_ground_contact_counts(env) == (0, 1)
        assert _outcome(env, state, True, False) == "incomplete_landing"
    base.legs[0].contacts = []
    assert _outcome(env, state, True, False) == "incomplete_landing"
    base.legs[0].contacts = [edge()]
    base.lander.position.x = 2.
    assert _outcome(env, state, True, False) == "off_pad_landing"
    base.game_over = True
    assert _outcome(env, state, True, False) == "crash"


def test_fast_expert_seed_94020_scores_real_contacts_despite_stale_flag():
    episode = rollout_episode(94020, fast_expert)
    terminal = episode["terminal_contact_diagnostic"]
    assert terminal["cached_leg_flags"] == [False, True]
    assert terminal["active_ground_contacts"] == [1, 1]
    assert terminal["on_pad"] and terminal["asleep"] and not terminal["game_over"]
    assert episode["outcome"] == "successful_landing"

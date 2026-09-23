"""Physics and outcome checks for the real LunarLander Box2D variant."""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("Box2D")

from lib.lunar_flight import (VARIANT, fast_expert, make_lunar_env,
                              rollout_episode, slow_expert, summarize_episodes)


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
        assert episode["outcome"] in ("successful_landing", "crash", "out_of_bounds", "time_limit")
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

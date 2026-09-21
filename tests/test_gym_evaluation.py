import numpy as np
import pytest

from lib.gym_evaluation import action_metrics, engine_regime, fit_scales, prediction_metrics


def test_engine_exact_boundaries():
    actions = np.array([[0, .5], [1e-6, .50001], [-1, -.5], [1, -.50001]])
    assert engine_regime(actions).tolist() == [[0, 0], [1, 1], [0, 0], [1, -1]]


def test_contact_errors_are_separate_and_nonfinite_fails():
    real = np.zeros((2, 8))
    predicted = real.copy()
    predicted[:, 6:] = 1
    metrics = prediction_metrics(predicted, real, np.ones(6))
    assert metrics["continuous_mse"] == 0
    assert metrics["contact_brier"] > .99
    predicted[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        prediction_metrics(predicted, real, np.ones(6))


def test_action_metric_requires_matched_worlds_and_detects_missing_response():
    data = {"states": np.zeros((4, 8)), "next_states": np.zeros((4, 8)),
            "actions": np.array([[-1, 0], [1, 0], [-1, -1], [-1, 1]]),
            "episode_ids": np.zeros(4), "anchor_steps": np.zeros(4)}
    data["next_states"][:, 0] = np.arange(4)
    assert action_metrics(data["next_states"], data, np.ones(6))["all"]["effect_mse"] == 0
    score = action_metrics(data["states"], data, np.ones(6))["all"]
    assert score["effect_mse"] == score["zero_response_mse"] > 0
    data["states"][1, 0] = 1
    with pytest.raises(ValueError, match="identical"):
        action_metrics(data["states"], data, np.ones(6))


def test_scaler_uses_both_roles():
    mean, scale = fit_scales({"states": np.zeros((4, 8)), "next_states": np.ones((4, 8))*2})
    np.testing.assert_equal(mean, np.ones(6))
    np.testing.assert_equal(scale, np.ones(6))


def test_rollout_never_refreshes_and_stops_at_saved_end():
    from experiments.evaluate_gym_transition import rollout_metrics
    states = np.zeros((7, 8), dtype=np.float32)
    states[:, 1] = .8
    episode = {"episode_id": 2, "terrain": [0.]*11, "states": states.tolist(),
               "next_states": states.tolist(), "actions": np.zeros((7, 2)).tolist()}
    data = {"states": states[:4], "next_states": states[:4], "episode_ids": np.array([2]*4),
            "anchor_steps": np.zeros(4, dtype=int)}
    inputs = []
    def predict(s, a, c):
        inputs.append(s.copy())
        out = s.copy()
        out[:, 0] += 1
        out[:, 6:] = .6
        return out
    metrics, scenes = rollout_metrics(predict, data, [episode], np.ones(6))
    assert len(inputs) == 7
    assert inputs[4][0, 0] == 4  # recursively predicted, never the reference zero
    assert np.all(inputs[1][:, 6:] == 1)  # disclosed threshold decoding
    assert metrics["horizons"]["5"]["count"] == 1
    assert metrics["horizons"]["20"]["count"] == 0
    assert len(scenes[0]["predicted"]) == 7


def test_checkpoint_rejects_mismatched_dataset_before_scoring():
    from experiments.evaluate_gym_transition import verify_checkpoint
    with pytest.raises(ValueError, match="dataset hashes"):
        verify_checkpoint({"provenance": {"dataset": {"train.npz": "wrong"}}}, "missing.pt",
                          {"references": {"train.npz": "expected"}})

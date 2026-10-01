from types import SimpleNamespace
import json

import numpy as np
import pytest

from lib.gym_control_evaluation import (engine_regimes, evaluate_controller,
    paired_comparison, selection_key, terminal_reason, verify_protocol, wilson_interval)
from lib.gym_data import sha256


def test_terminal_reason_matches_simulator_precedence():
    base = SimpleNamespace(lander=SimpleNamespace(awake=False), game_over=True)
    env = SimpleNamespace(unwrapped=base)
    assert terminal_reason(env, [1.1], True, True) == "successful_landing"
    base.lander.awake = True
    assert terminal_reason(env, [1.1], True, False) == "crash"
    base.game_over = False
    assert terminal_reason(env, [1.1], True, False) == "out_of_bounds"
    assert terminal_reason(env, [0], False, True) == "time_limit"
    assert terminal_reason(env, [0], False, False) is None
    with pytest.raises(RuntimeError):
        terminal_reason(env, [0], True, False)


def test_rates_pairing_and_validation_selection():
    low, high = wilson_interval(0, 50)
    assert low == 0 and .07 < high < .08
    assert wilson_interval(50, 50)[1] == 1
    assert engine_regimes([[0, .5], [.001, -.501], [-1, .501]]).tolist() == [[0,0], [1,-1], [0,1]]
    left = [dict(seed=1, outcome="successful_landing", **{"return": 20}),
            dict(seed=2, outcome="crash", **{"return": -10})]
    right = [dict(seed=2, outcome="successful_landing", **{"return": 30}),
             dict(seed=1, outcome="crash", **{"return": 20})]
    paired = paired_comparison(left, right)
    assert paired["return_wins"] == 0 and paired["return_ties"] == 1
    assert paired["landing_wins"] == paired["landing_losses"] == 1
    with pytest.raises(ValueError):
        paired_comparison(left, right[:1])
    rows = [dict(step=step, summary=dict(landing_rate=rate, mean_return=ret))
            for step,rate,ret in [(250,.5,0), (1000,.4,100), (2500,.5,0)]]
    assert max(rows, key=selection_key)["step"] == 250


def test_rollout_feedback_and_no_step_after_end(tmp_path, monkeypatch):
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.lander = SimpleNamespace(awake=True)
            self.game_over = False
            self.calls = 0
        def reset(self, seed):
            self.calls = 0
            return np.zeros(8, np.float32), {}
        def step(self, action):
            assert self.calls < 2
            self.calls += 1
            return np.ones(8, np.float32)*self.calls, 1., False, self.calls == 2, {}
        def close(self):
            pass
    monkeypatch.setattr("lib.gym_control_evaluation.make_env", Env)
    monkeypatch.setattr("lib.gym_control_evaluation.terrain_context", lambda e: np.zeros(11, np.float32))
    previous_seen = []
    def action(env, s, previous, terrain):
        previous_seen.append(previous.copy())
        return np.array([.2,.7], np.float32), {"component_id": 3}
    row = evaluate_controller(action, [1,2], tmp_path/"trace.npz", "test")
    assert row["summary"]["outcomes"]["time_limit"] == 2
    assert row["summary"]["simulator_steps"] == 4
    assert row["summary"]["main_engine_fraction"] == 1
    np.testing.assert_array_equal(previous_seen[0], [-1,0])
    np.testing.assert_allclose(previous_seen[1], [.2,.7])
    np.testing.assert_array_equal(previous_seen[2], [-1,0])
    trace = np.load(tmp_path/"trace.npz")
    np.testing.assert_array_equal(trace["seeds"], [1,1,2,2])


def test_provenance_and_cached_trace_drift_rejected(tmp_path, monkeypatch):
    from experiments.evaluate_gym_control import score
    source, episodes, trace = [tmp_path/name for name in ("source.py", "episodes.json", "trace.npz")]
    for path in (source, episodes, trace):
        path.write_text("initial")
    protocol = dict(simulator={"version":1}, source_episodes=str(episodes),
        source_episodes_sha256=sha256(episodes), sources={str(source):sha256(source)})
    protocol_path = tmp_path/"protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr("lib.gym_control_evaluation.simulator_provenance", lambda: {"version":1})
    assert verify_protocol(protocol_path) == protocol
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="source changed"):
        verify_protocol(protocol_path)
    (tmp_path/"evaluations").mkdir()
    old = dict(checkpoint_sha256=None, protocol_sha256=sha256(protocol_path),
               traces=str(trace), traces_sha256=sha256(trace))
    (tmp_path/"evaluations/expert_test.json").write_text(json.dumps(old))
    trace.write_text("changed")
    with pytest.raises(RuntimeError, match="traces changed"):
        score("expert", None, "test", "expert", tmp_path, protocol, "cpu")

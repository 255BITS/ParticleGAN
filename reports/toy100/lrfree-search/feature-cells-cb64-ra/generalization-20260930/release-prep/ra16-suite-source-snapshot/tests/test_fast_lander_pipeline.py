import numpy as np
import pytest

pytest.importorskip("rich")
from examples.fast_lander import extract_pairs, paired_speed, passes_gate


def episode(seed, steps, outcome="successful_landing"):
    return dict(seed=seed, steps=steps, outcome=outcome, return_=200., contact_step=steps-5,
                states=np.zeros((steps, 8), np.float32), actions=np.zeros((steps, 2), np.float32),
                next_states=np.zeros((steps, 8), np.float32))


def test_extraction_requires_same_reset_success_and_faster_flight():
    slow = [episode(1, 100), episode(2, 100), episode(3, 100), episode(4, 100)]
    fast = [episode(1, 80), episode(2, 30, "crash"), episode(3, 120), episode(9, 70)]
    pairs, data, count = extract_pairs(slow, fast)
    assert count == 1 and len(data["states"]) == 80
    assert len(pairs["slow_states"]) == 100
    np.testing.assert_array_equal(pairs["slow_seed"], pairs["fast_seed"])
    assert set(pairs["slow_seed"]) == {1}


def test_gate_rejects_crash_speed_and_lost_successes():
    slow = [episode(i, 100) for i in range(10)]
    fast = [episode(i, 80) for i in range(10)]
    assert passes_gate(slow, fast)
    fast[0] = episode(0, 10, "crash")
    assert paired_speed(slow, fast)["paired_speedup"] == 1.25
    assert not passes_gate(slow, fast)
    assert not passes_gate(slow, [episode(i, 95) for i in range(10)])
    assert not passes_gate(slow, [episode(i, 10, "crash") for i in range(10)])

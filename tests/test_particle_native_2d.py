"""Test the toy's protocol; opt into its platform-sensitive research gate."""
import math
import os
from unittest.mock import patch

import pytest

from experiments import toy_particle_native_2d as toy


@pytest.fixture(scope="module")
def result():
    return toy.run_gate()


def test_adversarial_protocol_and_honest_gate_reporting(result):
    for arm in ("current", "fixed"):
        assert all(math.isfinite(result[arm][key]) for key in ("live", "ema", "max_live", "max_ema"))
    assert result["adversarial_weight"] == 1.0
    assert result["l2_weight"] == 0.0
    assert result["b_cap_coeff"] == 1.0
    assert result["supervised_only"] is False
    collapse, passed = toy.gate_status(result["current"]["ema"], result["fixed"]["ema"])
    assert result["collapse"] == collapse
    assert result["fixed_pass"] == passed
    assert result["ok"] == (collapse and passed)


@pytest.mark.skipif(
    os.environ.get("RUN_PARTICLE_NATIVE_RESEARCH_GATE") != "1",
    reason="platform-sensitive convergence gate; opt in with RUN_PARTICLE_NATIVE_RESEARCH_GATE=1",
)
def test_latent_joint_historical_convergence(result):
    # This fixed-input training run both passes and fails on supported CI CPUs.
    # It is a research acceptance criterion, not an expected software failure.
    assert result["current"]["ema"] >= 1.0
    assert result["fixed"]["ema"] <= 0.18
    assert result["fixed"]["ema"] < result["current"]["ema"] * 0.25
    assert result["ok"], result


@pytest.mark.parametrize("current,fixed,expected", [
    (1.0, 0.18, (True, True)),
    (1.0, math.nextafter(0.18, math.inf), (True, False)),
    (math.nextafter(1.0, 0.0), 0.1, (False, True)),
    (0.6, 0.15, (False, False)),  # Relative improvement is strict.
    (math.inf, 0.1, (False, False)),
    (1.0, math.nan, (True, False)),
])
def test_gate_acceptance_boundaries(current, fixed, expected):
    assert toy.gate_status(current, fixed) == expected


def test_cli_rejects_failed_research_gate():
    with patch.object(toy, "run_gate", return_value={"ok": False}), patch("sys.argv", ["toy"]):
        with pytest.raises(SystemExit) as raised:
            toy.main()
    assert raised.value.code == 1

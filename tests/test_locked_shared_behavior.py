"""Measured behavior and numerical builder parity; no configuration gates."""

from copy import deepcopy

import pytest
import torch

from benchmarks.locked_shared import mode_hold, trajectory, two_pole
from benchmarks.locked_shared.run import compare, markdown, thinned_cap, no_cap
from particlegan import GANLoss, GradientPenalty


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def original_gan():
    return GANLoss("logistic", "rp")


def original_cap():
    return GradientPenalty("b_cap", coeff=1.0, kappa=1.0, norm="l2")


@pytest.mark.parametrize("train", [two_pole.train, trajectory.train, mode_hold.train_mode_hold])
def test_public_builders_reproduce_reference_constructor_training(train):
    actual = train()
    reference = train(gan_factory=original_gan, cap_factory=original_cap)
    # Includes complete measured outcomes, even a failing behavioral verdict.
    assert actual == reference


def test_two_pole_travel_and_slope_separate_real_negatives():
    good = two_pole.train()
    stranger = two_pole.train(pairing="stranger")
    thin = two_pole.train(cap_factory=thinned_cap)
    assert good["mean_abs"] >= two_pole.TRAVEL_MIN
    assert good["grad_med"] <= two_pole.GRAD_MED_MAX
    assert stranger["mean_abs"] < two_pole.TRAVEL_MIN
    assert thin["grad_med"] > two_pole.GRAD_MED_MAX
    assert (good["verdict"], stranger["verdict"], thin["verdict"]) == ("PASS", "FAIL", "FAIL")


def test_cap_off_is_trained_and_measured_as_collapsed():
    result = mode_hold.train_mode_hold(cap_factory=no_cap)
    assert result["modes"] <= mode_hold.COLLAPSE_MODES
    assert result["verdict"] == "FAIL"


def test_ring_gate_preserves_inconclusive_and_accepts_measured_success():
    assert mode_hold.verdict({"modes": 7, "hq": 0.95}) == "PASS"
    assert mode_hold.verdict({"modes": 3, "hq": 0.99}) == "INCONCLUSIVE"
    assert mode_hold.verdict({"modes": 8, "hq": 0.89}) == "INCONCLUSIVE"
    assert mode_hold.verdict({"modes": 2, "hq": 1.0}) == "FAIL"
    dead = mode_hold.diversity(torch.zeros(256, 2), mode_hold.ring_means())
    assert mode_hold.verdict(dead) == "FAIL"


def test_comparison_checks_metrics_not_just_matching_verdicts():
    actual = [{"toy": "trajectory", "arm": "locked_shared", "identity_mse": 0.1, "verdict": "FAIL"}]
    reference = deepcopy(actual)
    assert compare(actual, reference)[0]["match"]
    reference[0]["identity_mse"] = 0.2
    assert not compare(actual, reference)[0]["match"]
    reference[0]["identity_mse"] = float("nan")
    assert not compare(actual, reference)[0]["match"]
    with pytest.raises(ValueError, match="same unique rows"):
        compare(actual, [])


def test_report_does_not_turn_parity_into_behavioral_pass():
    rows = [{"toy": "trajectory", "arm": "locked_shared", "identity_mse": 0.1, "verdict": "FAIL"}]
    text = markdown({"rows": rows, "parity": compare(rows, rows), "python": "test", "torch": "test"})
    assert "0/1 behavioral targets" in text
    assert "1/1 rows match" in text
    assert "**FAIL** | MATCH" in text
    assert "does **not** reproduce" in text

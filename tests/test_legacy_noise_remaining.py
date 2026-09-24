"""The five nonstandard custom hosts apply shared training noise explicitly."""

import pytest

from benchmarks.locked_shared import two_pole
from benchmarks.locked_shared.hosts import (
    cover_leftover, mid_scale_identity, unipolar, unused_token_hold,
)
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy


def _run(host, policy, monkeypatch):
    if host == "two_pole":
        monkeypatch.setattr(two_pole, "TOY_STEPS", 2)
        return two_pole.train(noise_policy=policy)
    if host == "unipolar":
        return unipolar.run_arm("locked_rpgan", steps=2, noise_policy=policy)
    if host == "cover_leftover":
        return cover_leftover.fit_cover_leftover(
            cover_leftover.CoverRecipe(steps=2), noise_policy=policy,
        )
    if host == "unused_token_hold":
        return unused_token_hold.train(
            unused_token_hold.UnusedHoldRecipe(steps=2), noise_policy=policy,
        )
    if host == "mid_scale_identity":
        return mid_scale_identity.run_arm("locked", steps=2, noise_policy=policy)
    raise AssertionError(host)


@pytest.mark.parametrize("host", (
    "two_pole", "unipolar", "cover_leftover", "unused_token_hold",
    "mid_scale_identity",
))
def test_optional_noise_preserves_identity_and_reaches_both_training_paths(
    host, monkeypatch,
):
    control = _run(host, None, monkeypatch)
    zero = NoisePolicy(0.0, 0.0, 0.5, 2)
    assert _run(host, zero, monkeypatch) == control
    assert zero.receipt()["step_calls"] == 2

    noisy = NoisePolicy(0.029, 0.5, 0.5, 2)
    _run(host, noisy, monkeypatch)
    receipt = noisy.receipt()
    assert receipt["step_calls"] == 2
    assert receipt["input_nonzero_steps"] == 1
    assert receipt["output_train_elements"] > 0
    assert receipt["input_train_elements"] > 0

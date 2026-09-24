"""Focused fidelity tests for the isolated PR #84 smoothed-only adapter."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
from reports.toy100.pr84_smoothed_parity import _original_context, _state_sha


def test_five_point_critic_stencil_is_exact_and_restored():
    model = SimpleMLPDiscriminator(2, hidden_dim=8, n_hidden=1, fourier=0)
    points = torch.tensor([[.3, -.7], [1.1, .5]])
    original_forward = SimpleMLPDiscriminator.forward
    width = .15
    values = [original_forward(model, points)]
    for dim in range(2):
        shift = torch.zeros_like(points)
        shift[:, dim] = width
        values.extend((original_forward(model, points + shift),
                       original_forward(model, points - shift)))
    expected = torch.stack(values, 0).mean(0)
    rng_before = torch.get_rng_state().clone()
    with pr84_smoothed_candidate(task="mode_hold") as (recorder, _):
        recorder._smooth_on = True
        recorder._smooth_width = width
        assert torch.equal(model(points), expected)
    assert SimpleMLPDiscriminator.forward is original_forward
    assert torch.equal(torch.get_rng_state(), rng_before)


def _short_host(context, start_step):
    policy = NoisePolicy(.029, .5, .1, 1200)
    with context as (recorder, _):
        recorder.start_step = start_step
        result = mode_hold.train_mode_hold(
            mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy, diagnostics=True)
    return result, _state_sha(recorder), recorder


@pytest.mark.parametrize("start_step", [0, 1])
def test_three_step_host_matches_exact_40ec_weights_moments_rng_and_metrics(start_step):
    torch.set_num_threads(1)
    original = _short_host(_original_context("mode_hold"), start_step)
    cleaned = _short_host(pr84_smoothed_candidate(task="mode_hold"), start_step)
    assert original[:2] == cleaned[:2]
    assert original[2].rng_replay_verified == cleaned[2].rng_replay_verified == (
        6 if start_step == 0 else 4)
    for old, new in zip(original[2].records, cleaned[2].records):
        assert old["d"] == new["d"]
        assert old["g"]["rho"] == new["g"]["rho"]
        assert old["g"]["factor"] == new["g"]["factor"]
        assert old.get("critic_width") == new.get("critic_width")

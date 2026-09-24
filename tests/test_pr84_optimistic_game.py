"""Optimistic Adam on the PR84 committed step matches Algorithm 1."""

import torch

from reports.toy100.optimistic_adam_scratch import PREVIOUS_DIRECTION_KEY
from reports.toy100.pr84_optimistic_game import ALPHA, optimistic_adam_step
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
from reports.toy100.pr84_optimistic_game import pr84_optimistic_game
from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.pr84_smoothed_parity import _state_sha


def test_alpha_one_matches_two_current_minus_previous():
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=0.01, betas=(0.0, 0.99))
    expected = parameter.detach().clone()
    moment = torch.zeros_like(expected)
    second = torch.zeros_like(expected)
    previous = torch.zeros_like(expected)
    gradients = ([0.5, -0.2], [-0.1, 0.3])
    for t, gradient in enumerate(gradients, start=1):
        grad = torch.tensor(gradient, dtype=torch.float64)
        moment = grad
        second = 0.99 * second + 0.01 * grad.square()
        current = moment / ((second / (1 - 0.99 ** t)).sqrt() + 1e-8)
        expected = expected - 0.01 * ((1 + ALPHA) * current - ALPHA * previous)
        previous = current.clone()
        parameter.grad = grad
        optimistic_adam_step(optimizer, torch.optim.Adam.step)
        assert torch.allclose(parameter, expected, atol=1e-12, rtol=0)
    assert torch.allclose(optimizer.state[parameter][PREVIOUS_DIRECTION_KEY], previous)


def test_disabled_optimism_matches_pr84_on_three_steps():
    torch.set_num_threads(1)

    def run(factory):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with factory as (recorder, _):
            result = mode_hold.train_mode_hold(
                mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy, diagnostics=True)
        return result, _state_sha(recorder)

    assert run(pr84_smoothed_candidate(task="mode_hold")) == run(
        pr84_optimistic_game(task="mode_hold", optimism=False))


def test_enabled_optimism_moves_off_the_pr84_point():
    torch.set_num_threads(1)

    def weights(optimism):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with pr84_optimistic_game(task="mode_hold", optimism=optimism) as (recorder, _):
            mode_hold.train_mode_hold(
                mode_hold.ModeHoldRecipe(steps=2), noise_policy=policy)
            assert recorder.optimistic_updates > 0 if optimism else recorder.optimistic_updates == 0
            return _state_sha(recorder)

    assert weights(True) != weights(False)

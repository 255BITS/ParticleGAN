"""The idle field is exactly PR84 when disabled, and a flat critic does not fire."""

import torch

from reports.toy100.idle_barrier_field import LOOK_COUNT, LOOK_STEP, anti_gradient_peak


def test_flat_critic_push_is_exactly_zero():
    module = torch.nn.Linear(2, 1)
    torch.nn.init.zeros_(module.weight)
    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self(x).squeeze(-1)

    points = torch.tensor([[0., 0.], [1., -1.]])
    unit, slope, fires = anti_gradient_peak(forward, module, points)
    assert fires == 0
    assert torch.equal(unit, torch.zeros_like(points))
    assert torch.equal(slope, torch.zeros(2))
    assert LOOK_STEP == 0.5 and LOOK_COUNT == 4


def test_disabled_field_matches_pr84_on_two_steps():
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.idle_barrier_field import idle_barrier_field
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.pr84_smoothed_parity import _state_sha

    torch.set_num_threads(1)

    def run(context):
        policy = NoisePolicy(.029, .5, .1, 1200, output_noise_rng="isolated")
        with context as (recorder, _):
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2),
                                               noise_policy=policy, diagnostics=True)
        return result, _state_sha(recorder), policy.receipt()

    assert run(pr84_smoothed_candidate()) == run(idle_barrier_field(field=False))

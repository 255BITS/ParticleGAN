"""Mild anti-restoring G shrink: math, and disabled path matches PR84 factors."""

import torch

from reports.toy100.pr84_antirestoring_stay import mild_factor, signed_alignment
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
from reports.toy100.pr84_antirestoring_stay import pr84_antirestoring_stay


def test_mild_shrink_only_on_accepted_antirestoring_steps():
    factor, applied = mild_factor(.1, .25, -.4, advantage=-.01)
    assert applied and factor == .6
    factor, applied = mild_factor(.1, .25, -1., advantage=-.2)
    assert applied and factor == .5
    factor, applied = mild_factor(.1, .25, .2, advantage=-.2)
    assert not applied and factor == 1.
    factor, applied = mild_factor(.5, .25, -1., advantage=-.2)
    assert not applied and factor == .5
    factor, applied = mild_factor(.1, .25, -.4, mild=False, advantage=-.2)
    assert not applied and factor == 1.
    factor, applied = mild_factor(.1, .25, -.4, advantage=.02)
    assert not applied and factor == 1.
    factor, applied = mild_factor(.1, .25, -.4, advantage=0.)
    assert not applied and factor == 1.


def test_alignment_is_negative_when_gradient_change_opposes_a_restoring_hessian():
    base = [torch.zeros(2)]
    new = [torch.tensor([1., 0.])]
    g0 = [torch.zeros(2)]
    g1 = [torch.tensor([-1., 0.])]
    metric = [torch.ones(2)]
    kappa, rho, alignment = signed_alignment(base, new, g0, g1, metric)
    assert kappa < 0 and rho > 0 and alignment == -1.


def test_disabled_mild_matches_pr84_step_factors():
    torch.set_num_threads(1)
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    def run(factory):
        policy = NoisePolicy(.029, .5, .1, 1200)
        with factory as (recorder, _):
            mode_hold.train_mode_hold(
                mode_hold.ModeHoldRecipe(steps=3), noise_policy=policy, diagnostics=False)
        return [(row["g"]["rho"], row["g"]["factor"]) for row in recorder.records]

    assert run(pr84_antirestoring_stay(mild=False)) == run(pr84_smoothed_candidate())

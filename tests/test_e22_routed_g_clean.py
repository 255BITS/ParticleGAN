"""Software prerequisites only; the fixed512-update public command tests convergence."""
from copy import deepcopy
import math

import pytest
import torch

from examples import e22_routed_g_clean as toy


def scores():
    judges = [f"{arm}@{step}" for arm in toy.ARMS for step in toy.JUDGE_STEPS]
    return ({key:.7 for key in judges}, {key:.71 for key in judges}, {key:.701 for key in judges})


def flags():
    return dict(calibrated=True,bank_live=True,query_live=True,C_live=True)


def test_public_CPU_shape_ownership_and_private_evaluation_preflight_has_no_updates():
    before = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    result = toy.preflight(toy.make_data())
    assert result["pass"] and result["quality_updates"] == result["native_updates"] == 0
    assert result["initial_owners_exact"] and result["private_evaluation_state_immutable"]
    assert torch.equal(before,torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_terminal_game_metric_and_threshold_are_explicit():
    result = toy.scientific_gate(*scores(),**flags())
    assert result["pass"]
    assert result["threshold"] == -1e-4 and result["code_threshold"] == 1e-6
    assert math.isclose(result["metric_max_clean_minus_native_game"],-.01)


@pytest.mark.parametrize("judge",range(4))
def test_every_fixed_critic_can_reject_the_convergence_claim(judge):
    evidence = list(deepcopy(scores())); key = list(evidence[0])[judge]
    evidence[0][key] = evidence[1][key] - toy.WIN_MARGIN/2
    assert not toy.scientific_gate(*evidence,**flags())["pass"]
    evidence = list(deepcopy(scores())); evidence[2][key] = evidence[0][key]
    assert not toy.scientific_gate(*evidence,**flags())["pass"]


@pytest.mark.parametrize("requirement",tuple(flags()))
def test_calibration_and_live_particles_are_required(requirement):
    requirements = flags();requirements[requirement] = False
    assert not toy.scientific_gate(*scores(),**requirements)["pass"]


def test_missing_or_nonfinite_evidence_never_passes():
    evidence = list(deepcopy(scores()));evidence[0].pop(next(iter(evidence[0])))
    with pytest.raises(ValueError,match="four"):
        toy.scientific_gate(*evidence,**flags())
    evidence = list(deepcopy(scores()));evidence[0][next(iter(evidence[0]))] = math.nan
    with pytest.raises(ValueError,match="nonfinite"):
        toy.scientific_gate(*evidence,**flags())


def test_learned_nonfinite_table_or_gradient_is_rejected_without_updates():
    with torch.random.fork_rng(devices=[]): loop = toy.make_loop("native",toy.make_data())
    assert toy.learned_finite(loop.policy)
    loop.policy.table.grad = torch.full_like(loop.policy.table,math.nan)
    with pytest.raises(FloatingPointError,match="nonfinite learned"):
        toy.learned_finite(loop.policy)
    loop.policy.table.grad = None
    with torch.no_grad(): loop.policy.table[0,0] = math.inf
    with pytest.raises(FloatingPointError,match="nonfinite learned"):
        toy.learned_finite(loop.policy)
    assert loop.policy.completed_steps == 0

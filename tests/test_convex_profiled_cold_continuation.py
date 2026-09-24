from copy import deepcopy
import math

import torch

from reports.toy100.convex_readout_value import ReadoutProblem, fit_readout
from reports.toy100.pr84_convex_profiled_cold_continuation import attempt_value_move


def scalar_game(theta):
    real = torch.tensor([-1., 1.], dtype=torch.float64)
    jacobian = torch.ones(4, 1, 1, dtype=torch.float64)
    def problem():
        return ReadoutProblem((theta.detach()-real).unsqueeze(1), jacobian)
    def loss(weight):
        norm = (weight.square()+1e-12).sqrt()
        return torch.nn.functional.softplus((theta-real)*weight[0]).mean()+\
            torch.nn.functional.relu(norm-1).square().mean()
    return problem, loss


def test_consecutive_own_states_accumulate_valid_fixed_objective_bounds():
    theta = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    problem, loss = scalar_game(theta)
    readout, base = fit_readout(problem(), torch.zeros(1, dtype=torch.float64))
    initial_upper = base['certificate']['upper']
    lower_sum = 0.
    for _ in range(3):
        old = float(theta.detach())
        readout, base, receipt = attempt_value_move([theta], [torch.tensor(.2)], readout, base, loss, problem)
        assert receipt['accepted_alpha'] is not None
        assert 0 < float(theta.detach()) < old
        lower_sum += receipt['certified_decrease_lower_bound']
    # Intermediate gaps make the sum conservative relative to endpoint bounds.
    assert 0 < lower_sum <= base['certificate']['lower']-initial_upper+1e-12


def test_uninformative_valid_base_bound_rejects_all_trials_and_restores_exactly():
    theta = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    problem, loss = scalar_game(theta)
    readout, base = fit_readout(problem(), torch.zeros(1, dtype=torch.float64))
    loose = deepcopy(base)
    loose['certificate']['upper'] = math.log(2.)  # constant-critic feasible bound
    before = theta.detach().clone()
    final_readout, final_fit, receipt = attempt_value_move(
        [theta], [torch.tensor(.2)], readout, loose, loss, problem)
    assert receipt['status'] == 'NO_VERIFIED_DECREASE_REST'
    assert len(receipt['trials']) == 9
    assert torch.equal(theta, before) and torch.equal(final_readout, readout)
    assert final_fit == loose


def test_matched_zero_field_rests_without_any_trial_or_moment_update():
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    problem, loss = scalar_game(theta)
    readout, base = fit_readout(problem(), torch.zeros(1, dtype=torch.float64))
    final_readout, _, receipt = attempt_value_move([theta], [torch.tensor(.2)], readout, base, loss, problem)
    assert receipt['status'] == 'EXACT_ZERO_FIELD_REST'
    assert receipt['trials'] == [] and receipt['predicted'] == 0
    assert theta == 0 and torch.equal(final_readout, readout)

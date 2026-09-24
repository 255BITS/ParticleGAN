"""Later invalid-trial rejection must preserve all finite-path arithmetic."""

from unittest.mock import patch

import pytest
import torch

from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100.pr84_critic_refinement_finite import (
    InitialPointNonfinite, finite_trial_fit, pr84_critic_refinement_finite,
)
from reports.toy100.pr84_smoothed_parity import _without_runtime_timing


def quadratic(model, bank, *args):
    loss = (model.weight - 5.).square().sum()
    return loss, loss, loss * 0


def scripted_optimizer(values):
    class ScriptedLBFGS:
        def __init__(self, parameters, **options):
            self.p = list(parameters)[0]
            self.state = {self.p: {'n_iter': 1}}
        def zero_grad(self):
            self.p.grad = None
        def step(self, closure):
            for value in values:
                with torch.no_grad():
                    self.p.fill_(value)
                closure()
    return ScriptedLBFGS


def critic():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0)
    return model


def test_later_invalid_loss_stops_attempt_restores_best_and_counts_failed_trial():
    model = critic()
    with patch.object(fit, 'd_loss', quadratic), \
         patch.object(torch.optim, 'LBFGS', scripted_optimizer([0., 4., float('inf'), 5.])):
        row = finite_trial_fit(model, {}, None, None, 0, [torch.ones_like(model.weight)])
    assert model.weight.item() == 4.  # The unevaluated later5 point is never tried.
    assert row['closure_calls'] == 3 and row['finite_closure_calls'] == 2
    assert row['nonfinite_closure_calls'] == 1 and row['nonfinite_trial_rejected']
    assert row['extra_attempts'] == 0 and not row['closure_budget_exhausted']
    assert row['invalid_trials'][0]['closure'] == 3
    assert row['invalid_trials'][0]['total_loss'] is None
    assert row['invalid_trial_gradients_cleared'] and model.weight.grad is None


def test_finite_loss_with_nonfinite_gradient_is_also_a_declared_rejection():
    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.square().sum()
        @staticmethod
        def backward(ctx, grad):
            return torch.full((1, 1), float('nan'), dtype=torch.float64)
    def objective(model, bank, *args):
        loss = BadBackward.apply(model.weight) if model.weight.item() == 4 else model.weight.square().sum()
        return loss, loss, loss * 0
    model = critic()
    with patch.object(fit, 'd_loss', objective), \
         patch.object(torch.optim, 'LBFGS', scripted_optimizer([0., 4.])):
        row = finite_trial_fit(model, {}, None, None, 0, [torch.ones_like(model.weight)])
    assert model.weight.item() == 0
    assert row['closure_calls'] == 2 and row['invalid_trials'][0]['loss_finite']
    assert row['invalid_trials'][0]['nonfinite_gradient_elements'] == 1
    assert row['invalid_trial_gradients_cleared'] and model.weight.grad is None


def test_nonfinite_initial_field_remains_error_and_arbitrary_errors_are_not_caught():
    model = critic()
    with patch.object(fit, 'd_loss', quadratic), \
         patch.object(torch.optim, 'LBFGS', scripted_optimizer([float('nan')])):
        with pytest.raises(InitialPointNonfinite) as caught:
            finite_trial_fit(model, {}, None, None, 0, [torch.ones_like(model.weight)])
    assert caught.value.audit['closure'] == 1
    with patch.object(fit, 'd_loss', side_effect=FloatingPointError('unrelated field implementation error')):
        with pytest.raises(FloatingPointError, match='unrelated field implementation error'):
            finite_trial_fit(critic(), {}, None, None, 0, [torch.ones(1, 1)])


def project(actual, template):
    if isinstance(template, dict):
        return {key: project(actual[key], value) for key, value in template.items()}
    if isinstance(template, list):
        assert len(actual) == len(template)
        return [project(a, b) for a, b in zip(actual, template)]
    return actual


@pytest.mark.parametrize('budgeted', [False, True])
def test_finite_solver_path_is_bitwise_original_including_hard_budget_and_rest(budgeted):
    old, new = critic(), critic()
    with patch.object(fit, 'd_loss', quadratic):
        if budgeted:
            with patch.object(torch.optim, 'LBFGS', scripted_optimizer(range(100))):
                a = fit.relax(old, {}, None, None, 0, [torch.ones(1, 1)])
                b = finite_trial_fit(new, {}, None, None, 0, [torch.ones(1, 1)])
        else:
            a = fit.relax(old, {}, None, None, 0, [torch.ones(1, 1)])
            b = finite_trial_fit(new, {}, None, None, 0, [torch.ones(1, 1)])
    assert torch.equal(old.weight, new.weight)
    assert torch.equal(old.weight.grad, new.weight.grad)
    assert _without_runtime_timing(a) == project(_without_runtime_timing(b), _without_runtime_timing(a))
    assert b['closure_calls'] == b['finite_closure_calls'] <= 80
    assert not b['nonfinite_trial_rejected'] and b['nonfinite_closure_calls'] == 0
    if budgeted:
        assert b['closure_calls'] == 80 and b['closure_budget_exhausted']
    else:
        with patch.object(fit, 'd_loss', quadratic):
            row = finite_trial_fit(new, {}, None, None, 0, [torch.ones(1, 1)])
        assert row['closure_calls'] == 1 and row['iterations'] == 0 and new.weight.item() == 5


@pytest.mark.parametrize('task', ['mode_hold', 'trajectory'])
def test_real_finite_host_path_preserves_full_training_state_and_all_original_records(task):
    from tests.test_pr84_critic_refinement_cold import _cold_host
    from reports.toy100.pr84_critic_refinement_cold import pr84_critic_refinement_cold
    old = _cold_host(pr84_critic_refinement_cold(task=task), task, True)
    new = _cold_host(pr84_critic_refinement_finite(task=task), task, True)
    assert old[:3] == new[:3]
    assert all(torch.equal(a, b) if a is not None else b is None for a, b in zip(old[3], new[3]))
    original = _without_runtime_timing(old[-1].records)
    assert original == project(_without_runtime_timing(new[-1].records), original)
    assert new[-1].receipt()['later_nonfinite_rejections'] == 0

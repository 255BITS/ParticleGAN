"""Fixed-bank fit semantics, scope and exact alternating-host accounting."""

from unittest.mock import patch

import pytest
import torch

from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100.pr84_critic_refinement import (
    CriticRefinementRecorder, pr84_critic_refinement,
)


def test_fit_follows_objective_minimum_and_exact_zero_field_can_rest():
    critic = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        critic.weight.fill_(3.)
    def quadratic(model, bank, *args):
        loss = (model.weight - bank['target']).square().sum() / 2
        return loss, loss, loss * 0
    with patch.object(fit, 'd_loss', quadratic):
        row = fit.relax(critic, {'target': 5.}, None, None, 0, [torch.ones_like(critic.weight)])
        assert critic.weight.item() == pytest.approx(5., abs=1e-12)
        assert row['closure_calls'] <= 80 and row['iterations'] <= 40
        before = critic.weight.detach().clone()
        row = fit.relax(critic, {'target': 5.}, None, None, 0, [torch.ones_like(critic.weight)])
        assert torch.equal(critic.weight, before)
        assert row['closure_calls'] == 1 and row['iterations'] == 0


def test_fit_hard_closure_budget_restores_lowest_evaluated_loss_point():
    critic = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    def quadratic(model, bank, *args):
        loss = (model.weight - 5.).square().sum()
        return loss, loss, loss * 0
    class OverrunLBFGS:
        def __init__(self, parameters, **options):
            self.p = list(parameters)[0]
            self.state = {self.p: {'n_iter': 1}}
        def zero_grad(self):
            self.p.grad = None
        def step(self, closure):
            for index in range(100):
                with torch.no_grad():
                    self.p.fill_(float(index))
                closure()
    with patch.object(fit, 'd_loss', quadratic), patch.object(torch.optim, 'LBFGS', OverrunLBFGS):
        row = fit.relax(critic, {}, None, None, 0, [torch.ones_like(critic.weight)])
    assert row['closure_budget_exhausted'] and row['closure_calls'] == 80
    assert critic.weight.item() == 5.


@pytest.mark.parametrize('task', ['mode_hold', 'trajectory'])
def test_disabled_refinement_preserves_original_models_moments_ema_rng_metrics(task):
    from tests.test_pr84_opponent_prediction import _host
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _host(pr84_smoothed_candidate(task=task), task)
    disabled = _host(pr84_critic_refinement(task=task, refinement=False), task)
    assert original[:3] == disabled[:3]
    assert original[-1].records == disabled[-1].records
    assert disabled[-1].fit_gradient_evaluations == disabled[-1].parity_gradient_evaluations == 0


def _late_host(context, *, steps=1, isolated=False):
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, 0., .1, 1200, output_noise_rng='isolated' if isolated else None)
    with context as (rec, _):
        mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps), noise_policy=policy)
    rng = [torch.get_rng_state().clone(), rec._local['stream'].get_state().clone(),
           policy.input_stream.get_state().clone(),
           None if policy.output_stream is None else policy.output_stream.get_state().clone()]
    return rec, policy.receipt(), rng


@pytest.mark.parametrize('isolated', [False, True])
def test_active_exact_bank_refines_before_both_g_queries_without_advancing_rng_or_moments(isolated):
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _late_host(pr84_smoothed_candidate(), isolated=isolated)
    seen = []
    ordinary = CriticRefinementRecorder.step
    def observed(self, optimizer, ordinary_step, closure=None):
        if self.optimizers is not None and optimizer is self.optimizers[1] and self.phase in (1, 2):
            assert self._fit_row is not None
            seen.append((self.phase, self._smooth_width,
                         [p.detach().clone() for p in self._params(self.optimizers[0])]))
        return ordinary(self, optimizer, ordinary_step, closure)
    with patch.object(CriticRefinementRecorder, 'step', observed):
        active = _late_host(pr84_critic_refinement(), isolated=isolated)
    rec, noise, rng = active
    assert noise == original[1]
    assert all(torch.equal(a, b) if a is not None else b is None for a, b in zip(rng, original[2]))
    assert [item[0] for item in seen] == [1, 2]
    assert seen[0][1] == seen[1][1]
    assert all(torch.equal(a, b) for a, b in zip(seen[0][2], seen[1][2]))
    assert all(torch.equal(p, saved) for p, saved in zip(rec._params(rec.optimizers[0]), seen[1][2]))
    row = rec.refinement_records[0]
    assert row['first_bank_gradient_bitwise_equal']
    assert row['best_training_loss'] <= row['initial_training_loss']
    assert 1 <= row['closure_calls'] <= 80 and row['iterations'] <= 40
    assert rec.bank_rng_verified == rec.fit_rng_verified == rec.parity_gradient_evaluations == 1
    assert rec.rng_replay_verified == 2
    for opt, record in rec.rows.items():
        assert record['calls'] == 3
        assert all(int(opt.state[p]['step']) == 1 for group in opt.param_groups for p in group['params'])
    assert rec.receipt()['per_role_gradient_evaluations'] == dict(d=4+row['closure_calls'], g=3)


def test_unsupported_active_input_noise_and_conditional_host_are_rejected():
    from tests.test_pr84_opponent_prediction import _host
    with pytest.raises(ValueError, match='input_sigma=0'):
        _host(pr84_critic_refinement())
    with pytest.raises(ValueError, match='only been declared for mode_hold'):
        with pr84_critic_refinement(task='trajectory'):
            pass


def test_failed_refinement_restores_accepted_d_and_rng_before_raising():
    captured = {}
    def broken(critic, *args):
        captured['critic'] = critic
        captured['params'] = [p.detach().clone() for p in critic.parameters()]
        captured['rng'] = torch.get_rng_state().clone()
        with torch.no_grad():
            for p in critic.parameters():
                p.add_(1)
        torch.rand(7)
        raise RuntimeError('synthetic fitting failure')
    with patch.object(fit, 'relax', broken), pytest.raises(RuntimeError, match='synthetic fitting failure'):
        _late_host(pr84_critic_refinement())
    assert all(torch.equal(p, saved) for p, saved in zip(captured['critic'].parameters(), captured['params']))
    assert torch.equal(torch.get_rng_state(), captured['rng'])

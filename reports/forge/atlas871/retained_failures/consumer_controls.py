"""Synthetic CPU optimizer/consumer contracts. AUTHORED_NOT_RUN by Source author.

No model, target data, random draw or scientific task is constructed. Variable
beta fixtures test the applied Adam convention only, not a Recipe change.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from particlegan import k3p
from particlegan.continuous import OptimizerSurprise
from particlegan.policy import UpdatePolicy

BASE_LR = .00425
BASE_BETAS = (0., .999)
DIRECT_BETAS = (0., .9)


class Tester:
    def __init__(self):
        self.calls = []
    def observe(self, params, ratio, step=None):
        self.calls.append((params, ratio, step))


class Surprise:
    def __init__(self):
        self.calls = []
    def observe(self, key, optimizer, group):
        self.calls.append((key, optimizer, group, OptimizerSurprise.group_q(optimizer, group)))


def fixture(*, direct=True, extra=False, gain=True):
    p = torch.nn.Parameter(torch.zeros(12, 1, dtype=torch.float64))
    groups = [p]
    other = None
    if extra:
        other = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        groups = [{'params': [p]}, {'params': [other]}]
    opt = k3p.K3PGeneratorAdam(groups, direct_particles=[p] if direct else None,
        latent_table=None, direct_betas=DIRECT_BETAS, direct_gain=gain,
        lr=BASE_LR, betas=BASE_BETAS, amsgrad=True)
    return p, other, opt


def gradient(p, scale=1.):
    p.grad = torch.linspace(-1., 1., p.numel(), dtype=p.dtype).reshape_as(p) * scale


def consume(opt):
    policy = object.__new__(UpdatePolicy)
    testers = [Tester() for _ in opt.param_groups]
    policy.optimizers = [opt]
    policy.initial_lrs = [[BASE_LR for _ in opt.param_groups]]
    policy.lr_settle = SimpleNamespace(testers=[testers])
    policy.completed_steps = 7
    policy.surprise = Surprise()
    policy._settle_observe(0)
    return testers, policy.surprise


class DirectStepConsumers(unittest.TestCase):
    def test_actual_adam_and_rng_are_unchanged_before_controller_consumption(self):
        p, _, opt = fixture()
        q = torch.nn.Parameter(torch.zeros_like(p))
        reference = torch.optim.Adam([q], lr=BASE_LR, betas=BASE_BETAS, amsgrad=True)
        k3p._scope_reference_adam_graph_checks(reference)
        rng = torch.get_rng_state().clone()
        for scale in (1., 1., .01, -1.):
            gradient(p, scale)
            q.grad = p.grad.clone()
            opt.step()
            actual_group = opt._direct_step_group
            self.assertIs(actual_group[0], opt.param_groups[0])
            reference.param_groups[0]['lr'] = actual_group[1]
            reference.param_groups[0]['betas'] = actual_group[2]
            reference.step()
            reference.param_groups[0]['lr'] = BASE_LR
            reference.param_groups[0]['betas'] = BASE_BETAS
            self.assertTrue(torch.equal(p, q))
            for key in ('step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                self.assertTrue(torch.equal(opt.state[p][key], reference.state[q][key]))
            self.assertEqual(opt.param_groups[0]['lr'], BASE_LR)
            self.assertEqual(opt.param_groups[0]['betas'], BASE_BETAS)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_existing_consumers_receive_applied_gain_and_beta_without_group_writes(self):
        p, _, opt = fixture()
        for _ in range(2):
            gradient(p)
            opt.step()
        group = opt.param_groups[0]
        original = dict(group)
        versions = (p._version, p.grad._version, opt.state[p]['exp_avg_sq']._version)
        testers, surprise = consume(opt)
        applied_lr = BASE_LR * opt.direct_response.last_gain
        self.assertGreater(opt.direct_response.last_gain, 1.)
        self.assertEqual(testers[0].calls[0][1], applied_lr / BASE_LR)
        self.assertIs(testers[0].calls[0][0], group['params'])
        self.assertEqual(testers[0].calls[0][2], 8)
        _, owner, observed, actual_q = surprise.calls[0]
        self.assertIs(owner, opt)
        self.assertIsNot(observed, group)
        self.assertIs(observed['params'], group['params'])
        self.assertEqual(observed['betas'], DIRECT_BETAS)
        state = opt.state[p]
        expected = (p.grad.abs() / ((state['exp_avg_sq'] / (1.-.9 ** float(state['step']))).sqrt() + group['eps'])).mean()
        self.assertTrue(torch.equal(actual_q, expected))
        wrong_q = OptimizerSurprise.group_q(opt, group)
        self.assertLess(float(wrong_q), float(actual_q))
        self.assertEqual(set(group), set(original))
        for key in group:
            if key == 'params': self.assertIs(group[key], original[key])
            else: self.assertEqual(group[key], original[key])
        self.assertEqual(versions, (p._version, p.grad._version, state['exp_avg_sq']._version))

    def test_nondirect_optimizer_retains_original_consumer_group_and_values(self):
        p, _, opt = fixture(direct=False)
        gradient(p)
        opt.step()
        self.assertIsNone(opt._direct_step_group)
        testers, surprise = consume(opt)
        self.assertEqual(testers[0].calls[0][1], 1.)
        self.assertIs(surprise.calls[0][2], opt.param_groups[0])
        self.assertEqual(surprise.calls[0][2]['betas'], BASE_BETAS)

    def test_no_gradient_or_failed_step_cannot_reuse_an_old_receipt(self):
        p, _, opt = fixture()
        gradient(p)
        opt.step()
        self.assertIsNotNone(opt._direct_step_group)
        p.grad = None
        opt.step()
        self.assertIsNone(opt._direct_step_group)
        gradient(p)
        with patch.object(k3p, '_adam_step', side_effect=RuntimeError('inert Adam failure')):
            with self.assertRaisesRegex(RuntimeError, 'inert Adam failure'):
                opt.step()
        self.assertIsNone(opt._direct_step_group)
        self.assertEqual(opt.param_groups[0]['lr'], BASE_LR)
        self.assertEqual(opt.param_groups[0]['betas'], BASE_BETAS)
        opt.step()
        self.assertIsNotNone(opt._direct_step_group)
        def failed_closure():
            raise RuntimeError('inert closure failure')
        with self.assertRaisesRegex(RuntimeError, 'inert closure failure'):
            opt.step(failed_closure)
        self.assertIsNone(opt._direct_step_group)

    def test_completed_checkpoint_wire_stays_original_and_load_clears_transient(self):
        p, _, opt = fixture()
        gradient(p)
        opt.step()
        saved = deepcopy(opt.state_dict())
        self.assertEqual(set(saved), {'state', 'param_groups', 'regularizer'})
        self.assertEqual(set(saved['regularizer']), {'latent', 'direct'})
        self.assertEqual(set(saved['regularizer']['direct']['state']), {'started'})
        self.assertNotIn('_direct_step_group', saved)
        opt.load_state_dict(saved)
        self.assertIsNone(opt._direct_step_group)
        testers, surprise = consume(opt)
        self.assertEqual(testers[0].calls[0][1], 1.)
        self.assertIs(surprise.calls[0][2], opt.param_groups[0])
        gradient(p)
        opt.step()
        self.assertIsNotNone(opt._direct_step_group)

    def test_only_the_identical_direct_group_uses_the_receipt(self):
        p, other, opt = fixture(extra=True)
        gradient(p)
        other.grad = torch.ones_like(other)
        opt.step()
        testers, surprise = consume(opt)
        self.assertEqual(surprise.calls[0][2]['betas'], DIRECT_BETAS)
        self.assertIs(surprise.calls[1][2], opt.param_groups[1])
        self.assertEqual(surprise.calls[1][2]['betas'], BASE_BETAS)
        # Equal group contents are insufficient: a replaced group cannot use
        # the old object's successful-step metadata.
        opt.param_groups[0] = dict(opt.param_groups[0])
        _, second = consume(opt)
        self.assertIs(second.calls[0][2], opt.param_groups[0])
        self.assertEqual(second.calls[0][2]['betas'], BASE_BETAS)

    def test_amsgrad_surprise_remains_raw_second_moment_not_actual_move(self):
        p, _, opt = fixture(gain=False)
        gradient(p, 1.)
        opt.step()
        gradient(p, .01)
        opt.step()
        _, surprise = consume(opt)
        actual_q = surprise.calls[0][3]
        state, group = opt.state[p], opt.param_groups[0]
        bc = 1.-.9 ** float(state['step'])
        raw_q = (p.grad.abs() / ((state['exp_avg_sq'] / bc).sqrt() + group['eps'])).mean()
        max_q = (p.grad.abs() / ((state['max_exp_avg_sq'] / bc).sqrt() + group['eps'])).mean()
        self.assertTrue(torch.equal(actual_q, raw_q))
        self.assertGreater(float(actual_q), float(max_q))

    def test_variable_beta_fixture_preserves_current_power_adam_convention(self):
        p, _, opt = fixture(gain=False)
        gradient(p)
        opt.step()
        opt.direct_response.betas = (0., .8)
        gradient(p)
        opt.step()
        _, surprise = consume(opt)
        state, group = opt.state[p], opt.param_groups[0]
        actual_q = surprise.calls[0][3]
        current_power = (p.grad.abs() / ((state['exp_avg_sq'] / (1.-.8**2)).sqrt() + group['eps'])).mean()
        history_product = (p.grad.abs() / ((state['exp_avg_sq'] / (1.-.9*.8)).sqrt() + group['eps'])).mean()
        self.assertEqual(surprise.calls[0][2]['betas'], (0., .8))
        self.assertTrue(torch.equal(actual_q, current_power))
        self.assertFalse(torch.equal(actual_q, history_product))
        self.assertEqual(group['betas'], BASE_BETAS)


if __name__ == '__main__':
    unittest.main()

"""Cumulative emitted-law likelihood targets fitted from pre-G parameters.

Diagnostic scope: unconditional twelve-particle ring. Each existing native D
bank enters the complete history since learner activation once. Warm diagnostic
activation starts a fresh history, not a reconstruction of earlier banks.
Only the current bank supplies
global replacement candidates. Full-history positive GH5 quadrature defines the
objective; GH9 is an independent numerical audit. Fixed budgets, no target
labels, no model-response decay. This adds an explicit likelihood objective.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from reports.toy100 import reallocation_smoothed_candidate as host
from reports.toy100.forward_kl_free_filter import (
    quadrature, cross_entropy, global_donor, em_centroids)
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_critic_refinement_capture import _clone, _sha, _rng

METHOD = 'pr84_cumulative_forward_kl_current_bank_donors_prestart_fit'
# Derived once from the cold first native real bank before this experiment.
# It is not selected by quality or recomputed to improve a warm result.
WIDTH = 0.031286240422040236
SCHEMA = 'cumulative-forward-kl-real-banks-v1'


class ForwardKLRecorder(host.ReallocationRecorder):
    def __init__(self, *, start_step=0, task='mode_hold', correction=True):
        super().__init__(start_step=start_step, task=task, correction=correction)
        self.banks = []
        self.resume_boundary = start_step
        self.first_bank_id = start_step+1
        self.learner_bank_count = 0
        self.learner_last_observed_step = None

    def learner_state_dict(self):
        return dict(schema=SCHEMA, width=WIDTH, first_bank_id=self.first_bank_id,
                    banks=_clone(self.banks), bank_count=self.learner_bank_count,
                    last_bank_id=self.learner_last_observed_step)

    def load_learner_state_dict(self, state):
        keys = {'schema','width','first_bank_id','banks','bank_count','last_bank_id'}
        if (set(state) != keys or state['schema'] != SCHEMA or state['width'] != WIDTH
                or type(state['first_bank_id']) is not int or state['first_bank_id'] < 1
                or type(state['banks']) is not list
                or type(state['bank_count']) is not int
                or state['bank_count'] < 1
                or type(state['last_bank_id']) is not int
                or state['bank_count'] != len(state['banks'])):
            raise ValueError('invalid complete likelihood learner state')
        n = state['bank_count']
        expected_last = state['first_bank_id']+n-1 if n else None
        if state['last_bank_id'] != expected_last or expected_last != self.resume_boundary:
            raise ValueError('likelihood bank clock differs from ordered history')
        for bank in state['banks']:
            if (not isinstance(bank, torch.Tensor) or bank.dtype != torch.float32
                    or bank.device.type != 'cpu' or bank.shape != (128,2)
                    or not bool(torch.isfinite(bank).all())):
                raise ValueError('invalid native real bank in likelihood history')
        self.banks = _clone(state['banks'])
        self.first_bank_id = state['first_bank_id']
        self.learner_bank_count = n
        self.learner_last_observed_step = expected_last

    @torch.no_grad()
    def correct(self, optimizer):
        local = self._local
        if local['noise_policy'].output_noise_learnable:
            raise ValueError('likelihood diagnostic requires fixed output-noise scale')
        step = local['step']+1
        if step != self.first_bank_id+len(self.banks):
            raise RuntimeError('likelihood update is missing or repeating a native bank')
        if self.real.shape != (128,2) or self.real.dtype != torch.float32:
            raise ValueError('likelihood diagnostic requires native real128')
        clean = getattr(local['generator'], 'model', local['generator'])
        prior = local['prior']
        if len(prior.z) != 12:
            raise ValueError('twelve-particle diagnostic only')
        params = self._params(optimizer)
        native = [p.detach().clone() for p in params]
        rng = _sha(_rng(local))
        owner = _sha(dict(d=local['critic'].state_dict(), od=local['opt_d'].state_dict(),
                          og=optimizer.state_dict()))
        self.banks.append(self.real.detach().clone())
        self.learner_bank_count += 1
        self.learner_last_observed_step = step
        real = torch.cat(self.banks)
        update = quadrature(real, WIDTH, 5)
        audit = quadrature(real, WIDTH, 9)
        variance = WIDTH**2+float(local['noise_policy'].output_sigma)**2
        cost = lambda points: float(cross_entropy(*update, points, variance))
        pre_cost = cost(self.pre_points)
        native_cost = cost(clean(prior.z).detach())
        proposed, donors = global_donor(self.real, self.pre_points, *update,
            variance, limit=12, audit=audit)
        target, em = em_centroids(proposed, *update, variance, limit=20, audit=audit)
        target_cost = cost(target)
        for parameter, saved in zip(params, self.g_base):
            parameter.copy_(saved)
        fit = fit_output_targets(clean, prior.z, target)
        fitted_cost = cost(clean(prior.z).detach())
        eps = 64*torch.finfo(torch.float64).eps*max(1., abs(pre_cost))
        if target_cost > pre_cost+eps:
            raise RuntimeError('likelihood output target increased its objective')
        if fit['status'] != 'CONVERGED':
            selected, final = 'rest', pre_cost
            saved_params = self.g_base
        elif fitted_cost < min(pre_cost, native_cost)-eps:
            selected, final = 'joint_fit', fitted_cost
            saved_params = None
        elif native_cost < pre_cost-eps:
            selected, final = 'native_gan', native_cost
            saved_params = native
        else:
            selected, final = 'rest', pre_cost
            saved_params = self.g_base
        if saved_params is not None:
            for parameter, saved in zip(params, saved_params):
                parameter.copy_(saved)
        actual = cost(clean(prior.z).detach())
        if abs(actual-final) > 1e-10*max(1., abs(final)) or actual > pre_cost+eps:
            raise RuntimeError('likelihood whole-map selection mismatch')
        if rng != _sha(_rng(local)):
            raise RuntimeError('likelihood correction consumed training randomness')
        if owner != _sha(dict(d=local['critic'].state_dict(), od=local['opt_d'].state_dict(),
                              og=optimizer.state_dict())):
            raise RuntimeError('likelihood correction modified D or Adam state')
        self.rng_checks += 1
        self.owner_checks += 1
        row = dict(step=step, selected=selected, pre_cost=pre_cost,
            native_cost=native_cost, target_cost=target_cost, fitted_cost=fitted_cost,
            final_cost=final, fit=fit, donors=donors, em=em,
            audit9_pre=float(cross_entropy(*audit, self.pre_points, variance)),
            audit9_final=float(cross_entropy(*audit, clean(prior.z).detach(), variance)),
            bank_count=len(self.banks), history_samples=len(real), variance=variance,
            learner_state_sha256=_sha(self.learner_state_dict()))
        self.corrections.append(row)
        self.row['reallocation'] = {key:row[key] for key in
            ('selected','pre_cost','native_cost','target_cost','fitted_cost','final_cost')}

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD, scratch_optimizer_policy=METHOD,
            added_objective='complete empirical history since activation, emitted-law smoothed forward KL, GH5',
            output_allocation='current native D bank donors <=12, then equal-weight EM <=20',
            acceptance='actual GH5 loss of pre-G/native/converged pre-start fit; failed fit rests',
            width=WIDTH, learner_state_required=self.correction and self.task == 'mode_hold',
            history_first_absolute_bank=self.first_bank_id,
            learner_state_sha256=_sha(self.learner_state_dict()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def forward_kl_neural_candidate(*, task='mode_hold', start_step=0, correction=True):
    with patch.object(host, 'ReallocationRecorder', ForwardKLRecorder):
        with host.reallocation_smoothed_candidate(task=task, start_step=start_step,
                                                   correction=correction) as value:
            yield value

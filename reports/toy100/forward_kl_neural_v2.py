"""Unlaunched PR84 neural realization of finite-GH9 output targets.

The native alternating D and G Adam steps run once. After the native bounded G
step, the oracle-free cumulative-data operator proposes a clean 12x2 target
using GH5 and finite-GH9 selection (with remembered-data search only after
exact rest). Joint G/prior Gauss--Newton starts from pre-G parameters. Its
actual output is kept only after convergence and strict finite-GH9 decrease;
otherwise pre-G G/prior parameters are restored. No native G fallback is used.

This is an explicit new likelihood objective on the unconditional ring host,
not a general GAN guarantee or an executable training recommendation.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from reports.toy100 import reallocation_smoothed_candidate as host
from reports.toy100.forward_kl_free_filter import cross_entropy
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_critic_refinement_capture import _clone, _sha, _rng


METHOD = 'pr84_finite_gh9_remembered_target_prestart_joint_fit_rest_v2'
SCHEMA = 'finite-gh9-complete-native-real-history-v2'
WIDTH = 0.031286240422040236


class ForwardKLV2Recorder(host.ReallocationRecorder):
    def __init__(self, *, start_step=0, task='mode_hold', correction=True,
                 history_mode='activate', target_operator=None):
        if history_mode not in ('activate', 'resume'):
            raise ValueError('history_mode must explicitly activate or resume the learner')
        if history_mode == 'resume' and start_step < 1:
            raise ValueError('resuming likelihood history requires a completed host step')
        super().__init__(start_step=start_step, task=task, correction=correction)
        self.history_mode = history_mode
        self.resume_boundary = start_step
        self.loaded_history = False
        self.banks = []
        self.first_bank_id = start_step + 1
        self.learner_bank_count = 0
        self.learner_last_observed_step = None
        self.target_operator = target_operator
        self.gradient_checks = 0

    def learner_state_dict(self):
        return dict(schema=SCHEMA, width=WIDTH, first_bank_id=self.first_bank_id,
                    banks=_clone(self.banks), bank_count=self.learner_bank_count,
                    last_bank_id=self.learner_last_observed_step)

    def load_learner_state_dict(self, state):
        if self.history_mode != 'resume' or self.loaded_history or self.banks:
            raise ValueError('likelihood history loads once at a resume boundary')
        keys = {'schema', 'width', 'first_bank_id', 'banks', 'bank_count', 'last_bank_id'}
        if (not isinstance(state, dict) or set(state) != keys
                or state['schema'] != SCHEMA or state['width'] != WIDTH
                or type(state['first_bank_id']) is not int or state['first_bank_id'] < 1
                or type(state['bank_count']) is not int or state['bank_count'] < 1
                or type(state['last_bank_id']) is not int
                or type(state['banks']) is not list
                or state['bank_count'] != len(state['banks'])):
            raise ValueError('invalid finite-GH9 learner state')
        expected = state['first_bank_id'] + state['bank_count'] - 1
        if expected != self.resume_boundary or state['last_bank_id'] != expected:
            raise ValueError('likelihood history does not end at this host checkpoint')
        for bank in state['banks']:
            if (not isinstance(bank, torch.Tensor) or bank.dtype != torch.float32
                    or bank.device.type != 'cpu' or bank.shape != (128, 2)
                    or not bool(torch.isfinite(bank).all())):
                raise ValueError('invalid native real bank in likelihood history')
        self.banks = _clone(state['banks'])
        self.first_bank_id = state['first_bank_id']
        self.learner_bank_count = len(self.banks)
        self.learner_last_observed_step = expected
        self.loaded_history = True

    def phases(self, step, opt_d, opt_g, local):
        for phase in super().phases(step, opt_d, opt_g, local):
            if (self.correction and self.task == 'mode_hold' and self.enabled
                    and not self.passthrough and self.history_mode == 'resume'
                    and not self.loaded_history):
                raise RuntimeError('resume requires the source-bound likelihood history')
            yield phase

    def _pure_target(self, history, current_bank, pre_points, sigma):
        operator = self.target_operator
        if operator is None:
            from reports.toy100.forward_kl_gh9_remembered import propose_target
            operator = propose_target
        return operator(history, current_bank, pre_points, WIDTH, sigma)

    @torch.no_grad()
    def correct(self, optimizer):
        local = self._local
        if local.get('slow') is not None or local['recipe'].particle_l2 != 0:
            raise ValueError('unconditional unregularized-prior ring scope only')
        policy = local['noise_policy']
        if policy.output_noise_learnable or policy.output_scale is not None:
            raise ValueError('finite-GH9 mixture requires the fixed output-noise law')
        step = local['step'] + 1
        if self.history_mode == 'resume' and not self.loaded_history:
            raise RuntimeError('missing source-bound likelihood history at resume')
        if step != self.first_bank_id + len(self.banks):
            raise RuntimeError('native real history missed or repeated an outer update')
        if (self.real.shape != (128, 2) or self.real.dtype != torch.float32
                or self.real.device.type != 'cpu' or not bool(torch.isfinite(self.real).all())):
            raise ValueError('native finite real128 bank required')
        clean = getattr(local['generator'], 'model', local['generator'])
        prior = local['prior']
        if len(prior.z) != 12 or self.pre_points.shape != (12, 2):
            raise ValueError('finite-GH9 diagnostic requires twelve clean R2 outputs')
        params = self._params(optimizer)
        if len(params) != len(self.g_base):
            raise RuntimeError('pre-G parameter archive differs from G optimizer')
        rng_before = _sha(_rng(local))
        d_opt_before = _sha(dict(d=local['critic'].state_dict(),
            od=local['opt_d'].state_dict(), og=optimizer.state_dict()))
        grads_before = _sha(dict(g=[None if p.grad is None else p.grad.detach().clone()
                                     for p in params],
                                 d=[None if p.grad is None else p.grad.detach().clone()
                                     for p in local['critic'].parameters()]))

        self.banks.append(self.real.detach().clone())
        self.learner_bank_count += 1
        self.learner_last_observed_step = step
        history = torch.cat(self.banks)
        sigma = float(policy.output_sigma)
        variance = WIDTH**2 + sigma**2
        pure_row = None
        fit = None
        selected = 'EXACT_REST'
        try:
            target, pure_row, finite9 = self._pure_target(
                history, self.real, self.pre_points, sigma)
            locations, weights = finite9
            if (target.shape != (12, 2) or not bool(torch.isfinite(target).all())
                    or locations.shape != (81*len(history), 2)
                    or weights.shape != (len(locations),)
                    or not bool((weights > 0).all())
                    or abs(float(weights.sum())-1.) > 1e-12
                    or any(key in pure_row for key in ('initial_quality', 'final_quality'))):
                raise RuntimeError('pure target must supply oracle-free finite-GH9 data')
            pre_cost = float(pure_row['initial_audit9'])
            target_cost = float(pure_row['final_audit9'])
            eps = 64*torch.finfo(torch.float64).eps*max(1., abs(pre_cost))
            if (not torch.isfinite(torch.tensor([pre_cost, target_cost])).all()
                    or target_cost > pre_cost+eps):
                raise RuntimeError('pure output target raised its finite-GH9 objective')
            if pure_row['selected'] == 'EXACT_REST':
                if not torch.equal(target.double(), self.pre_points.double()):
                    raise RuntimeError('pure exact rest differs from pre-G output')
            elif target_cost >= pre_cost-eps:
                raise RuntimeError('non-rest pure output target did not strictly improve GH9')

            # Native G moments have advanced once, but no native parameter
            # displacement is allowed into this separate likelihood fit.
            for parameter, saved in zip(params, self.g_base):
                parameter.copy_(saved)
            if pure_row['selected'] != 'EXACT_REST':
                fit = fit_output_targets(clean, prior.z, target)
                fitted_points = clean(prior.z).detach()
                fitted_cost = float(cross_entropy(locations, weights,
                                                   fitted_points, variance))
                if fit['status'] == 'CONVERGED' and fitted_cost < pre_cost-eps:
                    selected = 'FINITE_GH9_FITTED_TARGET'
            else:
                fitted_cost = pre_cost
            if selected == 'EXACT_REST':
                for parameter, saved in zip(params, self.g_base):
                    parameter.copy_(saved)
                actual_points = clean(prior.z).detach()
                if not torch.equal(actual_points, self.pre_points):
                    raise RuntimeError('likelihood rest did not restore pre-G output')
                final_cost = pre_cost
            else:
                final_cost = fitted_cost
                actual_points = clean(prior.z).detach()
                actual = float(cross_entropy(locations, weights,
                                             actual_points, variance))
                if abs(actual-final_cost) > 1e-10*max(1., abs(final_cost)):
                    raise RuntimeError('fitted finite-GH9 cost changed after selection')
            if selected != 'EXACT_REST' and final_cost >= pre_cost-eps:
                raise RuntimeError('applied neural map failed finite-GH9 descent')
            if rng_before != _sha(_rng(local)):
                raise RuntimeError('finite-GH9 correction consumed training randomness')
            if d_opt_before != _sha(dict(d=local['critic'].state_dict(),
                    od=local['opt_d'].state_dict(), og=optimizer.state_dict())):
                raise RuntimeError('finite-GH9 correction modified D or Adam state')
            grads_after = _sha(dict(g=[None if p.grad is None else p.grad.detach().clone()
                                        for p in params],
                                    d=[None if p.grad is None else p.grad.detach().clone()
                                        for p in local['critic'].parameters()]))
            if grads_before != grads_after:
                raise RuntimeError('finite-GH9 correction changed native gradient buffers')
        except BaseException:
            for parameter, saved in zip(params, self.g_base):
                parameter.copy_(saved)
            self.banks.pop()
            self.learner_bank_count -= 1
            self.learner_last_observed_step = (step-1 if self.banks else None)
            raise

        self.rng_checks += 1
        self.owner_checks += 1
        self.gradient_checks += 1
        row = dict(step=step, selected=selected, pre_gh9=pre_cost,
            target_gh9=target_cost, fitted_gh9=fitted_cost, final_gh9=final_cost,
            pure=pure_row, fit=fit, bank_count=len(self.banks),
            first_bank_id=self.first_bank_id, history_samples=len(history),
            output_variance=variance,
            learner_state_sha256=_sha(self.learner_state_dict()))
        self.corrections.append(row)
        self.row['reallocation'] = {key: row[key] for key in
            ('selected', 'pre_gh9', 'target_gh9', 'fitted_gh9', 'final_gh9')}

    def receipt(self):
        result = super().receipt()
        active = self.correction and self.task == 'mode_hold'
        result.update(method=METHOD, scratch_optimizer_policy=METHOD,
            shared_gate_eligible=False,
            added_objective='finite GH9 cumulative smoothed empirical forward cross-entropy',
            output_allocation='GH5 current-bank proposal, GH9 whole-map acceptance; '
                'remembered-data donor only after exact rest',
            acceptance='converged pre-G joint fit with actual strict finite-GH9 descent; '
                'otherwise restore pre-G G/prior; no native G parameter fallback',
            conditional_scope='unchanged PR84; no likelihood correction',
            history_mode=self.history_mode, history_first_absolute_bank=self.first_bank_id,
            learner_state_required=active, learner_state_sha256=_sha(self.learner_state_dict()),
            correction_gradient_checks=self.gradient_checks,
            width=WIDTH, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def forward_kl_neural_v2(*, task='mode_hold', start_step=0, correction=True,
                         history_mode='activate', target_operator=None):
    with patch.object(host, 'ReallocationRecorder',
                      lambda **kwargs: ForwardKLV2Recorder(
                          **kwargs, history_mode=history_mode,
                          target_operator=target_operator)):
        with host.reallocation_smoothed_candidate(task=task, start_step=start_step,
                                                   correction=correction) as value:
            yield value

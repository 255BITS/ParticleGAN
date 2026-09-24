"""PR84 plus data-only global allocation and whole-map nonlinear acceptance.

This adds an explicit sampled C+Q objective. A discrete output proposal can
reallocate a redundant particle to any observed real point; joint G/prior
Gauss--Newton realizes the target. Final selection minimizes actual C+Q among
pre-update, native bounded GAN, and converged fitted proposals. The preceding
GAN curvature bound does not bound this additional move. No target centers,
HQ score, group count, clock gain, LR decay or critic zero pull enters it.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.chamfer_discrete_reallocation import greedy_real_reallocate, _cost
from reports.toy100.chamfer_pullback import chamfer_targets
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_critic_refinement_capture import _sha, _rng

METHOD = 'pr84_data_reallocation_joint_output_fit_with_whole_map_acceptance'


class ReallocationRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, task='mode_hold', correction=True):
        super().__init__(start_step=start_step)
        self.task, self.correction = task, correction
        self.corrections = []
        self.batch_checks = self.rng_checks = self.owner_checks = 0
        self.phase_samples = []

    def capture_sample(self, value):
        if self.correction and self.task == 'mode_hold' and self.enabled and not self.passthrough and self.phase is not None:
            self.phase_samples.append(value.detach().clone())

    def phases(self, step, opt_d, opt_g, local):
        for phase in super().phases(step, opt_d, opt_g, local):
            self.phase_samples = []
            yield phase
            if self.correction and self.task == 'mode_hold' and self.enabled and not self.passthrough:
                if len(self.phase_samples) != 2:
                    raise RuntimeError('expected the native D and G real minibatches')
                if phase == 0:
                    self.native_samples = self.phase_samples
                elif not all(torch.equal(a,b) for a,b in zip(self.native_samples,self.phase_samples)):
                    raise RuntimeError('native real minibatches did not replay exactly')

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        active = (self.correction and self.task == 'mode_hold' and self.enabled
                  and not self.passthrough)
        if active and optimizer is self.optimizers[0]:
            local = self._local
            if local.get('slow') is not None or local['recipe'].particle_l2 != 0:
                raise ValueError('unconditional unregularized-prior ring scope only')
            clean = getattr(local['generator'], 'model', local['generator'])
            if len(self.phase_samples) != 1:
                raise RuntimeError('D step must follow exactly one captured real minibatch')
            if self.phase == 0:
                self.real = self.phase_samples[0]
                self.pre_points = clean(local['prior'].z).detach().clone()
            elif not torch.equal(self.real, self.phase_samples[0]):
                raise RuntimeError('replayed real D minibatch changed')
            self.batch_checks += 1
        result = super().step(optimizer, ordinary_step, closure)
        if active and optimizer is self.optimizers[1] and self.phase == 2:
            self.correct(optimizer)
        return result

    @torch.no_grad()
    def correct(self, optimizer):
        local = self._local
        clean = getattr(local['generator'], 'model', local['generator'])
        prior = local['prior']
        parameters = self._params(optimizer)
        native = [value.detach().clone() for value in parameters]
        rng = _sha(_rng(local))
        owner = _sha(dict(d=local['critic'].state_dict(),
                          od=local['opt_d'].state_dict(), og=optimizer.state_dict()))
        pre_cost = _cost(self.real, self.pre_points)
        native_points = clean(prior.z).detach()
        native_cost = _cost(self.real, native_points)
        allocated, allocation = greedy_real_reallocate(self.real, self.pre_points)
        target, counts, _, _ = chamfer_targets(self.real.double(), allocated.double())
        target_cost = _cost(self.real, target)
        if target_cost > pre_cost + 1e-10 * max(1., pre_cost):
            raise RuntimeError('output proposal increased its actual C+Q objective')
        fit = fit_output_targets(clean, prior.z, target)
        fitted_cost = _cost(self.real, clean(prior.z).detach())
        tolerance = 64 * torch.finfo(torch.float64).eps * max(1., pre_cost)
        if fit['status'] == 'CONVERGED' and fitted_cost < min(pre_cost, native_cost) - tolerance:
            selected, final_cost = 'joint_fit', fitted_cost
        elif native_cost < pre_cost - tolerance:
            selected, final_cost = 'native_gan', native_cost
            for value, saved in zip(parameters, native):
                value.copy_(saved)
        else:
            selected, final_cost = 'rest', pre_cost
            for value, saved in zip(parameters, self.g_base):
                value.copy_(saved)
        if rng != _sha(_rng(local)):
            raise RuntimeError('data allocation or fitting consumed training randomness')
        self.rng_checks += 1
        if owner != _sha(dict(d=local['critic'].state_dict(),
                             od=local['opt_d'].state_dict(), og=optimizer.state_dict())):
            raise RuntimeError('joint output correction modified D or Adam state')
        self.owner_checks += 1
        actual = _cost(self.real, clean(prior.z).detach())
        if abs(actual-final_cost) > 1e-10*max(1.,abs(final_cost)) or actual > pre_cost + tolerance:
            raise RuntimeError('whole-map acceptance or restoration mismatch')
        row = dict(step=local['step']+1, selected=selected,
            pre_cost=pre_cost, native_cost=native_cost, target_cost=target_cost,
            fitted_cost=fitted_cost, final_cost=final_cost, allocation=allocation,
            target_assigned_counts=counts.tolist(), fit=fit)
        self.corrections.append(row)
        self.row['reallocation'] = {key:row[key] for key in
            ('selected','pre_cost','native_cost','target_cost','fitted_cost','final_cost')}

    def receipt(self):
        result = super().receipt()
        jacobians = sum(len(row['fit']['records']) for row in self.corrections)
        trials = sum(len(record['trials']) for row in self.corrections for record in row['fit']['records'])
        result['native_game_fields_per_outer_step'] = result['gradient_evaluations_per_outer_step']
        if jacobians:
            result['gradient_evaluations_per_outer_step'] = None
        result.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
            correction=self.correction, task=self.task, corrections=self.corrections,
            native_batch_checks=self.batch_checks, correction_rng_checks=self.rng_checks,
            correction_owner_checks=self.owner_checks,
            additional_joint_output_jacobians=jacobians, additional_nonlinear_output_trials=trials,
            added_objective='unit-mean real-to-support plus support-to-real squared distance',
            output_allocation='up to N globally best strictly improving donor-to-real-sample replacements, then one fixed-assignment C+Q minimization',
            nonlinear_fit='joint G and prior,20 GN iterations maximum,12 halvings maximum',
            acceptance='lowest actual C+Q among pre-update, native GAN, and converged fit',
            curvature_scope='original bound controls only native GAN proposal; not additional data fit',
            conditional_scope='unchanged PR84 update; no marginal correction',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def reallocation_smoothed_candidate(*, task='mode_hold', start_step=0, correction=True):
    native_sample = mode_hold.sample_ring
    with patch.object(frozen, 'SmoothedBothBoundRecorder',
        lambda *,start_step=0: ReallocationRecorder(start_step=start_step,task=task,correction=correction)):
        with frozen.pr84_smoothed_candidate(task=task,start_step=start_step) as value:
            recorder, _ = value
            def observed_sample(*args, **kwargs):
                sample = native_sample(*args, **kwargs)
                recorder.capture_sample(sample)
                return sample
            with patch.object(mode_hold, 'sample_ring', observed_sample):
                yield value

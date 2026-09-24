"""Warm-only PR84 with one nonlocal proposal scored by its original G loss.

No data-fitting objective is added. One native real point proposes replacing
one pre-G clean output; joint GN realizes that fixed cloud. Final choice is
the lowest original paired Rp G loss among pre-G, native bounded G, and a
converged joint fit. The original G bound only controls the native proposal.
"""
from contextlib import contextmanager, ExitStack
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from particlegan import ParticlePrior
from particlegan.gan_loss import GANLoss
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.reallocation_smoothed_candidate import ReallocationRecorder
from reports.toy100.joint_output_pullback import fit_output_targets
from reports.toy100.pr84_critic_refinement_capture import _sha, _rng

METHOD = 'pr84_original_G_loss_nonlocal_proposal_with_native_comparison'


@torch.no_grad()
def select_proposal(score, gan, fake, batch, candidates, particles):
    """Exact separable paired Rp loss; score already includes one stencil."""
    if gan.mode != 'rp' or gan.loss_type != 'logistic':
        raise ValueError('declared Rp logistic loss required')
    real_logits, fake_logits = score(batch['real']).reshape(-1), score(fake).reshape(-1)
    if real_logits.shape != fake_logits.shape:
        raise ValueError('paired native logits changed shape')
    terms = F.softplus(real_logits-fake_logits)
    before = float(gan.g_loss(fake_logits, real_logits))
    scores = torch.empty(particles, len(candidates), dtype=torch.float64)
    for donor in range(particles):
        mask = batch['indices'] == donor
        if not bool(mask.any()):
            scores[donor].fill_(before)
            continue
        points = candidates[:, None, :] + batch['sigma']*batch['noise'][mask][None, :, :]
        logits = score(points.reshape(-1, points.shape[-1])).reshape(len(candidates), -1)
        replacement = F.softplus(real_logits[mask][None, :]-logits)
        scores[donor] = (terms[~mask].double().sum()+replacement.double().sum(1))/len(fake)
    selected = int(scores.argmin())
    donor, sample = divmod(selected, len(candidates))
    point = candidates[sample].detach().clone()
    trial = fake.clone()
    mask = batch['indices'] == donor
    trial[mask] = point + batch['sigma']*batch['noise'][mask]
    actual = float(gan.g_loss(score(trial).reshape(-1), real_logits))
    return dict(donor=donor, real_sample=sample, target=point.tolist(),
                pre_loss=before, proposal_loss=actual, improves=actual<before,
                enumerated_minimum=float(scores.min()), proposals=particles*len(candidates))


class AdversarialReallocationRecorder(ReallocationRecorder):
    def __init__(self, **options):
        super().__init__(**options)
        self.in_correction = False
        self.phase_indices, self.phase_outputs = [], []
        self.phase_G_loss = None
        self.native_G_loss_checks = 0
        self.batch_noise_checks = 0

    def active(self):
        return (self.correction and self.task == 'mode_hold' and self.enabled
                and not self.passthrough and self.phase is not None and not self.in_correction)

    def phases(self, step, opt_d, opt_g, local):
        for phase in super().phases(step, opt_d, opt_g, local):
            self.phase_indices, self.phase_outputs = [], []
            self.phase_G_loss = None
            yield phase
            if self.active():
                if len(self.phase_indices) != 2 or len(self.phase_outputs) != 2:
                    raise RuntimeError('expected exactly two native latent/output draws per phase')
                fingerprint = [(index, output['noise']) for index,output in
                               zip(self.phase_indices, self.phase_outputs)]
                if phase == 0:
                    self.draw_reference = fingerprint
                else:
                    if any(not torch.equal(a,b) for pair,old in zip(fingerprint,self.draw_reference)
                           for a,b in zip(pair,old)):
                        raise RuntimeError('particle indices or output-noise draws changed during replay')
                self.batch_noise_checks += 1

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self.active() and optimizer is self.optimizers[1] and self.phase == 1:
            local, policy = self._local, self._local['noise_policy']
            if (policy.input_sigma != 0 or policy.output_scale is not None
                    or policy.output_stream is not None or local['recipe'].fm_weight != 0
                    or local['recipe'].vicreg_weight != 0):
                raise ValueError('warm zero-input/global-fixed-output pure-Rp scope only')
            if len(self.phase_indices)!=2 or len(self.phase_outputs)!=2 or len(self.phase_samples)!=2:
                raise RuntimeError('incomplete native G batch capture')
            self.g_batch = dict(indices=self.phase_indices[1].clone(),
                noise=self.phase_outputs[1]['noise'].clone(), sigma=policy.output_sigma,
                real=self.phase_samples[1].clone(), emitted=self.phase_outputs[1]['emitted'].clone())
            self.native_pre_loss = self.phase_G_loss
            self.frozen_width, self.frozen_smoothing = self._smooth_width, self._smooth_on
        return super().step(optimizer, ordinary_step, closure)

    @torch.no_grad()
    def correct(self, optimizer):
        local = self._local
        clean = getattr(local['generator'], 'model', local['generator'])
        critic = getattr(local['critic'], 'model', local['critic'])
        prior, gan = local['prior'], local['gan']
        params = self._params(optimizer)
        native = [p.detach().clone() for p in params]
        native_state = self._smooth_on, self._smooth_width
        rng = _sha(_rng(local))
        owner = _sha((critic.state_dict(), local['opt_d'].state_dict(), optimizer.state_dict(),
                      [p.grad for p in self._params(local['opt_d'])+params]))
        self.in_correction = True
        self._smooth_on, self._smooth_width = self.frozen_smoothing, self.frozen_width
        calls = dict(critic_score_batches=0, critic_score_points=0)
        try:
            def score(points):
                calls['critic_score_batches'] += 1
                calls['critic_score_points'] += len(points)
                return critic(points)

            def objective():
                fake = clean(prior.z[self.g_batch['indices']])+self.g_batch['sigma']*self.g_batch['noise']
                a,b = score(fake).reshape(-1), score(self.g_batch['real']).reshape(-1)
                if a.shape != b.shape:
                    raise RuntimeError('candidate paired logits changed shape')
                return float(gan.g_loss(a,b))

            native_loss = objective()
            for p,value in zip(params,self.g_base): p.copy_(value)
            fake = clean(prior.z[self.g_batch['indices']])+self.g_batch['sigma']*self.g_batch['noise']
            if not torch.equal(fake,self.g_batch['emitted']):
                raise RuntimeError('pre-G reconstruction changed native emitted sample bytes')
            pre_loss = objective()
            if pre_loss != self.native_pre_loss:
                raise RuntimeError('phase1 native Rp G loss does not replay exactly')
            self.native_G_loss_checks += 1
            selection = select_proposal(score,gan,fake,self.g_batch,self.real,len(prior.z))
            if selection['pre_loss'] != pre_loss:
                raise RuntimeError('enumeration baseline differs from paired native loss')
            target = self.pre_points.clone()
            fit = dict(status='SKIPPED_NONIMPROVING_PROPOSAL',records=[])
            fitted_loss = pre_loss
            if selection['improves']:
                target[selection['donor']] = self.real[selection['real_sample']]
                fit = fit_output_targets(clean,prior.z,target)
                fitted_loss = objective()
            if fit['status']=='CONVERGED' and fitted_loss < min(pre_loss,native_loss):
                selected, final_loss = 'joint_fit',fitted_loss
            elif native_loss < pre_loss:
                selected, final_loss = 'native_gan',native_loss
                for p,value in zip(params,native): p.copy_(value)
            else:
                selected, final_loss = 'rest',pre_loss
                for p,value in zip(params,self.g_base): p.copy_(value)
            actual = objective()
            if actual != final_loss or actual > pre_loss:
                raise RuntimeError('original-G-loss selection/restoration mismatch')
            movement = [p.detach()-value for p,value in zip(params,self.g_base)]
            row = dict(step=local['step']+1, selected=selected, pre_loss=pre_loss,
                native_loss=native_loss, fitted_loss=fitted_loss, final_loss=final_loss,
                proposal=selection, fit=fit, stencil_width=self.frozen_width,
                stencil_enabled=self.frozen_smoothing, native_pre_loss_exact=True,
                joint_parameter_displacement=float(sum(v.double().square().sum() for v in movement).sqrt()),
                prior_parameter_displacement=float(movement[-1].double().norm()),
                clean_output_displacement=float((clean(prior.z)-self.pre_points).double().norm()),
                **calls)
        finally:
            self._smooth_on,self._smooth_width = native_state
            self.in_correction = False
        if rng != _sha(_rng(local)):
            raise RuntimeError('adversarial search/landing consumed training RNG')
        self.rng_checks += 1
        if owner != _sha((critic.state_dict(),local['opt_d'].state_dict(),optimizer.state_dict(),
                          [p.grad for p in self._params(local['opt_d'])+params])):
            raise RuntimeError('adversarial search/landing changed D, Adam or gradient buffers')
        self.owner_checks += 1
        self.corrections.append(row)
        self.row['adversarial_reallocation'] = {key:row[key] for key in
            ('selected','pre_loss','native_loss','fitted_loss','final_loss','stencil_width')}

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD,scratch_optimizer_policy=METHOD,
            added_objective='none; original paired Rp logistic G loss and native G-side stencil',
            output_allocation='one globally best donor/native-D-real-sample proposal by the unchanged G loss',
            acceptance='converged joint fit only if loss below both pre/native; otherwise lower native/rest',
            curvature_scope='bounds apply only to native Adam proposals; additional global jump is unbounded by them',
            input_noise_scope='active correction requires zero discriminator input noise',
            native_G_loss_exact_checks=self.native_G_loss_checks,
            native_index_and_noise_checks=self.batch_noise_checks,
            extra_critic_score_batches=sum(row['critic_score_batches'] for row in self.corrections),
            extra_critic_score_points=sum(row['critic_score_points'] for row in self.corrections),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def adversarial_reallocation(*,task='mode_hold',start_step=0,correction=True):
    native_sample, native_prior = mode_hold.sample_ring, ParticlePrior.sample
    native_output, native_G_loss = NoisePolicy.output, GANLoss.g_loss
    with patch.object(frozen,'SmoothedBothBoundRecorder',
        lambda *,start_step=0:AdversarialReallocationRecorder(start_step=start_step,task=task,correction=correction)):
        with frozen.pr84_smoothed_candidate(task=task,start_step=start_step) as value:
            recorder,_ = value
            def sampled(*args,**kwargs):
                sample = native_sample(*args,**kwargs)
                recorder.capture_sample(sample)
                return sample
            def latent(prior,*args,**kwargs):
                output = native_prior(prior,*args,**kwargs)
                if recorder.active(): recorder.phase_indices.append(output[1].detach().clone())
                return output
            def output(policy,generated,**kwargs):
                active = recorder.active() and not policy._evaluating
                if active:
                    if policy.output_stream is not None or policy.output_scale is not None:
                        raise ValueError('warm global fixed-output-noise scope only')
                    rng = torch.get_rng_state().clone()
                value = native_output(policy,generated,**kwargs)
                if active:
                    with torch.random.fork_rng(devices=[]):
                        torch.set_rng_state(rng)
                        noise = torch.randn_like(generated) if policy.output_sigma else torch.zeros_like(generated)
                    if not torch.equal(value.detach(),generated.detach()+policy.output_sigma*noise):
                        raise RuntimeError('output-noise capture did not reconstruct the native draw')
                    recorder.phase_outputs.append(dict(noise=noise.detach().clone(),emitted=value.detach().clone()))
                return value
            def g_loss(gan,fake,real=None):
                result = native_G_loss(gan,fake,real)
                if recorder.active(): recorder.phase_G_loss = float(result.detach())
                return result
            with ExitStack() as stack:
                stack.enter_context(patch.object(mode_hold,'sample_ring',sampled))
                stack.enter_context(patch.object(ParticlePrior,'sample',latent))
                stack.enter_context(patch.object(NoisePolicy,'output',output))
                stack.enter_context(patch.object(GANLoss,'g_loss',g_loss))
                yield value

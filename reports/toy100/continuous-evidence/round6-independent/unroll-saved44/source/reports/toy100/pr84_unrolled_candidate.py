"""Warm-only one-step total G response on the original alternating PR84 rule.

One virtual sharp penalized D step uses the saved post-Adam diagonal metric.
It has no persistent D/moment update. G differentiates its stencil objective
through that step; the original G own-curvature bound sees the same frozen
starting critic, metric, stencil and draws at both query points. This changes
G's general-sum surrogate, not merely its preconditioner. No L-BFGS fit lives
here. Cold input-noise/conditional hosts are intentionally undeclared.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
from torch.func import functional_call

from benchmarks.locked_shared import mode_hold
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.functional_b_cap import functional_b_cap
from reports.toy100.pr84_critic_refinement_capture import _sha, _rng


METHOD = 'pr84_one_virtual_penalized_d_step_total_g_gradient'


@torch.no_grad()
def cached_batches(local):
    policy = local['noise_policy']
    if policy.input_sigma != 0 or policy.output_scale is not None:
        raise ValueError('one-step candidate is scoped to zero input noise and fixed output scale')
    if (local['batch'] != 128 or local['gan'].mode != 'rp'
            or local['gan'].loss_type != 'logistic'
            or local['recipe'].particle_l2 != 0 or local['recipe'].fm_weight != 0
            or local['recipe'].vicreg_weight != 0):
        raise ValueError('one-step candidate requires the unchanged warm ring objective')
    stream = torch.Generator().set_state(local['stream'].get_state())
    output = (None if policy.output_stream is None else
              torch.Generator().set_state(policy.output_stream.get_state()))
    clean = getattr(local['generator'], 'model', local['generator'])
    before = _sha(_rng(local))
    with torch.random.fork_rng(devices=[]):
        real_d = mode_hold.sample_ring(local['means'], 128, mode_hold.SIGMA, stream)
        rows = []
        for role in ('d', 'g'):
            latent, indices = local['prior'].sample(128, generator=stream)
            values = clean(latent)
            noise = (torch.zeros_like(values) if not policy.output_sigma else
                     torch.randn_like(values) if output is None else
                     torch.randn(values.shape, generator=output, dtype=values.dtype))
            real = real_d if role == 'd' else mode_hold.sample_ring(
                local['means'], 128, mode_hold.SIGMA, stream)
            rows.append(dict(real=real.detach(), indices=indices.detach(), noise=noise.detach(),
                             sigma=policy.output_sigma))
    if before != _sha(_rng(local)):
        raise RuntimeError('cached native batches changed training random streams')
    return rows


def stencil(opponent, points, width):
    values = [opponent(points)]
    if width:
        for dim in range(2):
            shift = torch.zeros_like(points)
            shift[:, dim] = width
            values.extend((opponent(points + shift), opponent(points - shift)))
    return torch.stack(values).mean(0)


def field(local, batches, metric, width, *, response):
    """Reconstruct native tensors, then take a partial or total G gradient."""
    generator = getattr(local['generator'], 'model', local['generator'])
    critic = getattr(local['critic'], 'model', local['critic'])
    prior, gan, cap = local['prior'], local['gan'], local['regularizer']
    params = {name: p.detach().clone().requires_grad_(True)
              for name, p in critic.named_parameters()}
    sharp = lambda x: functional_call(critic, params, (x,))
    d, g = batches
    fake_d = generator(prior.z[d['indices']]) + d['sigma'] * d['noise']
    inner = gan.d_loss(sharp(d['real']), sharp(fake_d))
    inner = inner + functional_b_cap(cap, sharp, d['real'], fake_d, local['step'] + 1)
    dg = torch.autograd.grad(inner, tuple(params.values()), create_graph=response != 'base')
    if response == 'base':
        virtual = params
    else:
        virtual = {name: p - (m.detach() * grad.double()).to(p.dtype)
                   for (name, p), m, grad in zip(params.items(), metric, dg)}
        if response == 'detached':
            virtual = {name: p.detach() for name, p in virtual.items()}
    opponent = lambda x: functional_call(critic, virtual, (x,))
    fake_g = generator(prior.z[g['indices']]) + g['sigma'] * g['noise']
    loss = gan.g_loss(stencil(opponent, fake_g, width), stencil(opponent, g['real'], width))
    values = torch.autograd.grad(loss, list(generator.parameters()) + list(prior.parameters()))
    if not all(torch.isfinite(value).all() for value in (*dg, *values)):
        raise FloatingPointError('nonfinite virtual D or total G field')
    return [value.detach().clone() for value in values], [value.detach().clone() for value in dg]


class UnrolledRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, response='full'):
        super().__init__(start_step=start_step)
        if response not in ('full', 'detached', 'off'):
            raise ValueError(response)
        self.response = response
        self.active = False
        self.extra_d_queries = self.extra_g_queries = self.mixed_backprops = 0
        self.batch_rng_checks = self.field_rng_checks = self.native_g_parity_checks = 0
        self.native_d_batch_checks = 0
        self._width = None

    def phases(self, step, opt_d, opt_g, local):
        self.active = self.enabled and step >= self.start_step and self.response != 'off'
        self._width = None
        if self.active:
            self.batches = cached_batches(local)
            self.batch_rng_checks += 1
        yield from super().phases(step, opt_d, opt_g, local)

    def _arm_smoothed_critic(self):
        if not self.active or self.passthrough:
            return super()._arm_smoothed_critic()
        if self.phase == 2:
            if self._width is None:
                raise RuntimeError('missing frozen stencil')
            self._smooth_on, self._smooth_width = self._width
        else:
            super()._arm_smoothed_critic()
            if self.phase == 1:
                self._width = self._smooth_on, self._smooth_width

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if (self.active and not self.passthrough and self.phase == 0
                and optimizer is self.optimizers[0]):
            local, batch = self._local, self.batches[0]
            generator = getattr(local['generator'], 'model', local['generator'])
            fake = generator(local['prior'].z[batch['indices']]) + batch['sigma'] * batch['noise']
            if not torch.equal(batch['real'], local['real']) or not torch.equal(fake, local['fake']):
                raise RuntimeError('cached D batch differs from the native D block')
            self.native_d_batch_checks += 1
        if (self.active and not self.passthrough and self.phase in (1, 2)
                and optimizer is self.optimizers[1]):
            local = self._local
            before = _sha(_rng(local))
            width = self._smooth_width if self._smooth_on else 0.
            was_smooth = self._smooth_on
            self._smooth_on = False  # functional inner D is sharp; outer stencil is explicit.
            try:
                with torch.enable_grad():
                    # Check the cached native G reconstruction at both points.
                    base, _ = field(local, self.batches, self.metric_d, width, response='base')
                    actual = [p.grad.detach().clone() for p in self._params(optimizer)]
                    if not all(torch.equal(a, b) for a, b in zip(base, actual)):
                        raise RuntimeError('cached partial G field differs from actual host gradient')
                    self.native_g_parity_checks += 1
                    used, _ = field(local, self.batches, self.metric_d, width, response=self.response)
                for p, value in zip(self._params(optimizer), used):
                    p.grad = value
            finally:
                self._smooth_on = was_smooth
            if before != _sha(_rng(local)):
                raise RuntimeError('virtual response field consumed training randomness')
            self.field_rng_checks += 1
            # Each reconstruction computes its native D field and G gradient.
            self.extra_d_queries += 2
            self.extra_g_queries += 2
            self.mixed_backprops += self.response == 'full'
        return super().step(optimizer, ordinary_step, closure)

    def receipt(self):
        result = super().receipt()
        result.update(method=METHOD, response=self.response, scratch_optimizer_policy=METHOD,
            shared_gate_eligible=False, additional_d_fields=self.extra_d_queries,
            additional_g_fields=self.extra_g_queries, total_response_backprops=self.mixed_backprops,
            native_g_parity_checks=self.native_g_parity_checks, batch_rng_checks=self.batch_rng_checks,
            native_d_batch_checks=self.native_d_batch_checks,
            field_rng_checks=self.field_rng_checks,
            per_role_gradient_queries=dict(d=3*self.outer_steps+self.extra_d_queries,
                                           g=3*self.outer_steps+self.extra_g_queries),
            virtual_metric='constant within the outer update; post-real-D Adam diagonal',
            persistent_d_update='original bounded D Adam only; no L-BFGS refinement',
            generator_objective='full derivative through one virtual sharp Rp plus cap D step',
            active_stencil_policy='phase1 width frozen through phase2',
            stationary_point_scope='general-sum surrogate can change stationary points',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result


@contextmanager
def pr84_unrolled_candidate(*, task='mode_hold', start_step=0, response='full'):
    if task != 'mode_hold':
        raise ValueError('only warm mode_hold is declared for this candidate')
    with patch.object(frozen, 'SmoothedBothBoundRecorder',
                      lambda *, start_step=0: UnrolledRecorder(start_step=start_step, response=response)):
        with frozen.pr84_smoothed_candidate(task=task, start_step=start_step) as value:
            yield value

"""Noise-faithful/conditional support for the frozen critic-refinement rule.

The solver, accepted-D placement, G own-bound, moment ownership and selection
rule are inherited unchanged from pr84_critic_refinement. This separate module
adds a cached empirical D objective for the original discriminator input noise
and the trajectory host's fixed slow conditioning. A bank is eight native
D batches: 1024 pairs for mode_hold; 96 pairs for the 12-row trajectory host.

Each native batch caches the four independent input perturbations consumed by
the actual sharp D block: real/fake logits, then real/fake gradient penalty.
The same cached draws are used for every closure. The penalty differentiates
with respect to the original fast/data coordinates; conditioning stays fixed.
No original RNG stream, noise clock or policy counter is advanced by the fit.
Zero-input-noise mode_hold reduces bitwise to the frozen warm adapter.
"""

from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold, trajectory
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from reports.toy100 import pr84_critic_refinement as warm
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100 import pr84_critic_relaxation as fit


METHOD = "pr84_alternating_bounded_empirical_critic_refinement_cold_support"
_WARM_BANK = warm.fixed_bank
_SHARP_LOSS = fit.d_loss


def _clone_stream(stream):
    value = torch.Generator()
    value.set_state(stream.get_state())
    return value


@torch.no_grad()
def fixed_bank(local, task):
    policy = local.get("noise_policy")
    if policy is None or policy.output_scale is not None:
        raise ValueError("cached-noise refinement requires a fixed output scale")
    gan, regularizer = local['gan'], local['regularizer']
    if (gan.mode != 'rp' or gan.loss_type != 'logistic' or regularizer.arm != 'b_cap'
            or regularizer.lazy_k != 1 or regularizer.method != 'autograd'
            or regularizer.target_anneal != 'none'):
        raise ValueError("cached-noise objective is declared only for constant autograd b_cap and Rp logistic")
    generator = getattr(local['generator'], 'model', local['generator'])
    critic = getattr(local['critic'], 'model', local['critic'])
    if next(generator.parameters()).device.type != 'cpu':
        raise ValueError("cached-noise refinement is limited to the frozen CPU hosts")
    expected = ((SimpleMLPGenerator, SimpleMLPDiscriminator) if task == 'mode_hold'
                else (trajectory._Generator, trajectory._Critic))
    if not isinstance(generator, expected[0]) or not isinstance(critic, expected[1]):
        raise ValueError("host model classes changed")
    if task == 'mode_hold' and policy.input_sigma == 0:
        first, bank = _WARM_BANK(local)
        for value in (first, bank):
            value.update(task=task, input_sigma=0., penalty_step=local['step'] + 1,
                         slow=None, input_noise=None)
        return first, bank
    stream = _clone_stream(local['stream']) if task == 'mode_hold' else None
    input_stream = _clone_stream(policy.input_stream)
    output_stream = _clone_stream(policy.output_stream) if policy.output_stream is not None else None
    rows = []
    with torch.random.fork_rng(devices=[]):
        for _ in range(warm.BANK_BATCHES):
            if task == 'mode_hold':
                if local['batch'] != warm.BANK_BATCH_SIZE:
                    raise ValueError("mode_hold native batch changed")
                real = mode_hold.sample_ring(local['means'], local['batch'], mode_hold.SIGMA, stream)
                latent, _ = local['prior'].sample(local['batch'], generator=stream)
                clean = generator(latent)
                slow = None
            else:
                slow = local['slow'].detach()
                real = local['paired'].detach()
                clean = generator(slow, local['prior'].z)
            if policy.output_sigma:
                noise = (torch.randn_like(clean) if output_stream is None else torch.randn(
                    clean.shape, generator=output_stream, dtype=clean.dtype, device=clean.device))
                fake = clean + policy.output_sigma * noise
            else:
                fake = clean
            # Preserve the host's four independent calls, including the two
            # inside the gradient penalty. Add only to fast/data coordinates.
            perturbations = ([torch.randn(real.shape, generator=input_stream,
                                 dtype=real.dtype, device=real.device)
                              for _ in range(4)] if policy.input_sigma else None)
            rows.append(dict(real=real.detach(), fake=fake.detach(), slow=slow,
                             input_noise=perturbations, task=task,
                             input_sigma=policy.input_sigma,
                             penalty_step=local['step'] + (task == 'mode_hold')))
    bank = {key: torch.cat([row[key] for row in rows]) for key in ('real', 'fake')}
    bank.update(task=task, input_sigma=policy.input_sigma,
                penalty_step=rows[0]['penalty_step'],
                slow=None if rows[0]['slow'] is None else torch.cat([row['slow'] for row in rows]),
                input_noise=None if rows[0]['input_noise'] is None else [
                    torch.cat([row['input_noise'][call] for row in rows]) for call in range(4)])
    return rows[0], bank


class _CachedInputCritic(torch.nn.Module):
    def __init__(self, critic, bank):
        super().__init__()
        self.critic, self.bank, self.calls = critic, bank, 0

    def forward(self, data):
        call = self.calls
        if call >= 4:
            raise RuntimeError("declared D objective has more than four critic input calls")
        self.calls += 1
        if self.bank['input_sigma']:
            noise = self.bank['input_noise'][call]
            if noise.shape != data.shape:
                raise RuntimeError("cached input noise does not match the evaluated D data")
            data = data + self.bank['input_sigma'] * noise
        slow = self.bank['slow']
        logits = self.critic(data) if slow is None else self.critic(slow, data)
        # Match trajectory._FastView exactly for its penalty calls.
        return logits.unsqueeze(-1) if slow is not None and call >= 2 else logits


def cached_d_loss(critic, bank, gan, regularizer, step):
    if bank['task'] == 'mode_hold' and bank['input_sigma'] == 0:
        return _SHARP_LOSS(critic, bank, gan, regularizer, bank['penalty_step'])
    view = _CachedInputCritic(critic, bank)
    result = _SHARP_LOSS(view, bank, gan, regularizer, bank['penalty_step'])
    if view.calls != 4:
        raise RuntimeError("declared D objective did not use all four cached input calls")
    return result


def _complete_bank_sha(bank):
    values = [bank['real'], bank['fake']]
    if bank['slow'] is not None:
        values.append(bank['slow'])
    if bank['input_noise'] is not None:
        values.extend(bank['input_noise'])
    return warm._tensor_sha(*values)


class ColdCriticRefinementRecorder(warm.CriticRefinementRecorder):
    def __init__(self, *, task, start_step=0, refinement=True):
        super().__init__(start_step=start_step, refinement=refinement)
        self.task = task

    def phases(self, step, opt_d, opt_g, local):
        for phase in super().phases(step, opt_d, opt_g, local):
            yield phase
        if self._active:
            row = self.refinement_records[-1]
            row.update(host_update=step + (self.task == 'mode_hold'),
                       bank_native_batch_size=len(self._first_batch['real']),
                       bank_pairs=len(self._bank['real']),
                       input_sigma=self._bank['input_sigma'],
                       conditioning='unchanged slow coordinates' if self.task == 'trajectory' else None,
                       complete_bank_sha256=_complete_bank_sha(self._bank),
                       complete_first_bank_sha256=_complete_bank_sha(self._first_batch))

    def receipt(self):
        value = super().receipt()
        rows = self.refinement_records
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
            refinement_scope='CPU mode_hold and trajectory, original data-only input noise, fixed output scale',
            task=self.task, bank_batch_size=(rows[0]['bank_native_batch_size'] if rows else None),
            bank_input_noise_policy='four frozen draws/native batch in actual D call order; no redraw per closure',
            fit_sample_pairs_evaluated=sum(row['closure_calls'] * row['bank_pairs'] for row in rows),
            bank_clean_g_forwards=len(rows) * warm.BANK_BATCHES,
            gradient_query_size_note='three native host batches/player plus one D parity batch and eight-native-batch fit closures',
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            frozen_warm_adapter_sha256=hashlib.sha256(Path(warm.__file__).read_bytes()).hexdigest())
        return value


@contextmanager
def pr84_critic_refinement_cold(*, task='mode_hold', start_step=0, refinement=True):
    if task not in ('mode_hold', 'trajectory'):
        raise ValueError('unsupported frozen refinement host')
    def factory(*, start_step=0):
        return ColdCriticRefinementRecorder(task=task, start_step=start_step, refinement=refinement)
    with ExitStack() as stack:
        stack.enter_context(patch.object(warm, 'fixed_bank', lambda local: fixed_bank(local, task)))
        stack.enter_context(patch.object(fit, 'd_loss', cached_d_loss))
        stack.enter_context(patch.object(frozen, 'SmoothedBothBoundRecorder', factory))
        yield stack.enter_context(frozen.pr84_smoothed_candidate(task=task, start_step=start_step))

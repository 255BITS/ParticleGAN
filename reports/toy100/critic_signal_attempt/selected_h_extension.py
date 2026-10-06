"""Interface-only extension of archived H to the remaining frozen hosts.

The selected discriminator objective/noise and all hyperparameters are imported
unchanged from critic_signal.py. The conditional view exposes the host's existing
noise stream to H's paired-score helper; it neither adds a forward nor draws RNG.
Auxiliary host G fits are explicitly disabled to honor H's adversarial-only rule.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from unittest.mock import patch

import torch
from benchmarks.transfer_suite.legacy_noise_adapters import _InputAdapter, NoisePolicy
from benchmarks.locked_shared.hosts import ae_gan_hold, unused_token_hold
from particlegan.grad_regularizers import GradRegularizer
import critic_signal

ARCHIVED_SIGNAL_POLICY = critic_signal.signal_policy


class ConditionalCriticView(_InputAdapter):
    """Callable view with the exact host critic and policy, no added noise."""
    def __init__(self, function):
        owners = [cell.cell_contents for cell in (getattr(function, '__closure__', None) or ())
                  if isinstance(cell.cell_contents, torch.nn.Module)]
        owners += [value for value in (getattr(function, '__defaults__', None) or ())
                   if isinstance(value, torch.nn.Module)]
        owners = list({id(owner): owner for owner in owners}.values())
        if len(owners) != 1:
            raise ValueError('conditional H plumbing requires one existing critic owner')
        owner = owners[0]
        policy = getattr(owner, 'noise_policy', None)
        if not isinstance(policy, NoisePolicy):
            raise ValueError('conditional critic must expose its existing NoisePolicy')
        super().__init__(owner, policy, data_index=0)
        self.function = function

    def forward(self, points):
        # The host score() already injects raw-space noise exactly once.
        return self.function(points)


@contextmanager
def extended_signal_policy(options):
    with ARCHIVED_SIGNAL_POLICY(options) as receipt, ExitStack() as stack:
        extension = {
            'policy': 'selected_h_faithful_host_plumbing_v1',
            'archived_candidate_unchanged': True,
            'conditional_callable_views': 0,
            'conditional_stream_reuse': 'existing host NoisePolicy.input_stream',
            'additional_optimizer_updates': 0,
            'additional_training_forwards': 0,
            'auxiliary_overrides': [],
        }
        receipt['host_extension'] = extension
        original_penalty = GradRegularizer.penalty
        original_ae = ae_gan_hold.train
        original_unused = unused_token_hold.train

        def conditional_penalty(reg, discriminator, *args, **kwargs):
            if not isinstance(discriminator, torch.nn.Module):
                discriminator = ConditionalCriticView(discriminator)
                extension['conditional_callable_views'] += 1
            return original_penalty(reg, discriminator, *args, **kwargs)

        def pure_ae(cfg, *args, **kwargs):
            fields = dict(reconstruction_weight=0., fm_weight=0., cover_weight=0., particle_l2=0.)
            extension['auxiliary_overrides'].append({
                'host': 'ae_gan_hold',
                'before': {key: getattr(cfg, key) for key in fields},
                'after': fields,
            })
            return original_ae(replace(cfg, **fields), *args, **kwargs)

        def pure_unused(cfg, *args, **kwargs):
            fields = dict(hold_weight=0., fm_weight=0., cover_weight=0., particle_l2=0.)
            extension['auxiliary_overrides'].append({
                'host': 'unused_token_hold',
                'before': {key: getattr(cfg, key) for key in fields},
                'after': fields,
            })
            return original_unused(replace(cfg, **fields), *args, **kwargs)

        stack.enter_context(patch.object(GradRegularizer, 'penalty', conditional_penalty))
        stack.enter_context(patch.object(ae_gan_hold, 'train', pure_ae))
        stack.enter_context(patch.object(unused_token_hold, 'train', pure_unused))
        yield receipt

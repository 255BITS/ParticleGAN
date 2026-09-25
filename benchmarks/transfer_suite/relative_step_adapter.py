"""Research-only, role-blind bound on the relative size of an Adam proposal.

This changes the optimizer update rule, not merely its declared learning rate.
The underlying Adam moments see unchanged gradients. Each tensor's proposal is
attenuated, never amplified, using its own pre-update RMS and a common floor.
"""
from contextlib import contextmanager
import math
import time
from unittest.mock import patch

import torch

from benchmarks.smart_descent.evaluate import FixedControl


def mechanism(fraction):
    if fraction is not None and (not math.isfinite(fraction) or fraction <= 0):
        raise ValueError('fraction must be positive and finite, or None for identity')
    return dict(kind='adam_relative_step_cap', fraction=fraction,
                parameter_rms_floor=.1, epsilon=1e-12, trace_interval=20,
                granularity='parameter tensor', moment_update='ordinary Adam from raw gradients',
                formula='delta * min(1, fraction * max(rms(parameter_before), floor) / (rms(delta) + epsilon))',
                role_blind=True, clock_input=False, metric_input=False)


@contextmanager
def adapted_steps(card, evidence):
    """Scope the research update rule to existing Adam optimizers in one episode."""
    if card != mechanism(card['fraction']):
        raise ValueError('mechanism must match the complete declared equation')
    original_step, original_control = torch.optim.Adam.step, FixedControl.step
    states = {}
    evidence.update(mechanism=card, trace=[], summaries={}, seconds=0.)

    def control(self, optimizer, completed_updates, role):
        # This annotation is solely for reporting; it never affects attenuation.
        optimizer._relative_step_report_role = role
        return original_control(self, optimizer, completed_updates, role)

    @torch.no_grad()
    def step(optimizer, *args, **kwargs):
        started = time.perf_counter()
        serial, count = states.setdefault(optimizer, [len(states), 0])
        before = [(gi, pi, p, p.detach().clone())
                  for gi, group in enumerate(optimizer.param_groups)
                  for pi, p in enumerate(group['params']) if p.grad is not None]
        proposal_started = time.perf_counter()
        result = original_step(optimizer, *args, **kwargs)
        proposal_seconds = time.perf_counter() - proposal_started
        for gi, pi, parameter, previous in before:
            delta = parameter - previous
            parameter_rms = float(previous.square().mean().sqrt())
            update_rms = float(delta.square().mean().sqrt())
            gradient_rms = float(parameter.grad.square().mean().sqrt())
            fraction = card['fraction']
            factor = (1. if fraction is None else
                      min(1., fraction * max(parameter_rms, card['parameter_rms_floor']) /
                          (update_rms + card['epsilon'])))
            if factor < 1.:
                parameter.copy_(previous + factor * delta)
            group = optimizer.param_groups[gi]
            role = ('prior' if group.get('_comparison_prior') else
                    getattr(optimizer, '_relative_step_report_role', 'unknown'))
            summary = evidence['summaries'].setdefault(role, dict(
                tensors=0, attenuated=0, factor_sum=0., minimum_factor=1.,
                proposal_rms_sum=0., applied_rms_sum=0., gradient_rms_sum=0.))
            summary['tensors'] += 1
            summary['attenuated'] += int(factor < 1.)
            summary['factor_sum'] += factor
            summary['minimum_factor'] = min(summary['minimum_factor'], factor)
            summary['proposal_rms_sum'] += update_rms
            summary['applied_rms_sum'] += update_rms * factor
            summary['gradient_rms_sum'] += gradient_rms
            if count % card['trace_interval'] == 0:
                evidence['trace'].append(dict(optimizer=serial, step=count, group=gi,
                    tensor=pi, role=role, shape=list(parameter.shape),
                    lr=group['lr'], betas=list(group['betas']), parameter_rms=parameter_rms,
                    gradient_rms=gradient_rms, proposal_rms=update_rms,
                    applied_rms=float((parameter - previous).square().mean().sqrt()), factor=factor))
        states[optimizer][1] += 1
        evidence['seconds'] += time.perf_counter() - started - proposal_seconds
        return result

    with patch.object(FixedControl, 'step', control), patch.object(torch.optim.Adam, 'step', step):
        yield

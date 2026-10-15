"""Recovered training-gradient signal estimator, with strict state validation.

The recurrence is from the frozen astra-dualnorm-history source manifest
9bc9e6ae14ccd13455bf8805bd68b9e3affd6e09e112a06be7a97b368248d61f;
the original gradient_signal.py SHA256 is
b7fd16b80604e7b0982aa426f0723b95ffa1d0637375a8e933598f2fbe16fdeb.
Its historical caller scaled learning rates, after gradient normalization.
This module estimates rho only: placement belongs to the caller. Optional
explicit gradients select a loss channel without replacing parameter.grad.
There is no evaluator, RNG, tunable decay, or EMA-gradient substitution.
"""
from copy import deepcopy
import math

import torch


class GradientSignal:
    def __init__(self, roles):
        self.roles = tuple(roles)
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError("gradient-signal roles must be nonempty and unique")
        self.packets = {}

    @torch.no_grad()
    def scale(self, groups, *, gradients=None):
        parameters = {role: [p for group in groups if group['role'] == role
                             for p in group['params']] for role in self.roles}
        if gradients is not None and set(gradients) != set(self.roles):
            raise ValueError("gradient-signal input roles mismatch")
        packets, scales = dict(self.packets), {}
        for role, params in parameters.items():
            if not params:
                continue
            inputs = [p.grad for p in params] if gradients is None else gradients[role]
            if len(inputs) != len(params):
                raise ValueError("gradient-signal input count mismatch")
            for parameter, gradient in zip(params, inputs):
                if gradient is not None and (
                        not isinstance(gradient, torch.Tensor) or gradient.is_sparse
                        or gradient.is_complex() or gradient.shape != parameter.shape
                        or gradient.device != parameter.device):
                    raise ValueError("invalid gradient-signal input tensor")
            values = [torch.zeros_like(p, dtype=torch.float64) if g is None else g.detach().double()
                      for p, g in zip(params, inputs)]
            if any(not bool(torch.isfinite(g).all()) for g in values):
                raise ValueError("Nonfinite gradient-signal input")
            previous = self.packets.get(role)
            if previous is None:
                previous = dict(means=[torch.zeros_like(g) for g in values],
                                second=values[0].new_zeros(()), weight=0., weight2=0.,
                                count=0, zero_steps=0, last={})
            beta = 7 / 8
            alpha = 1 - beta
            state = dict(
                means=[m * beta + g * alpha for m, g in zip(previous['means'], values)],
                second=previous['second'] * beta + alpha * sum(g.square().sum() for g in values),
                weight=previous['weight'] * beta + alpha,
                weight2=previous['weight2'] * beta * beta + alpha * alpha,
                count=previous['count'] + 1, zero_steps=previous['zero_steps'])
            effective = state['weight'] ** 2 / state['weight2']
            mean2 = sum((m / state['weight']).square().sum() for m in state['means'])
            empirical = (state['second'] / state['weight'] - mean2).clamp_min(0)
            variance = empirical / (1 - 1 / effective) if effective > 1 + 1e-12 else empirical.new_zeros(())
            signal = (mean2 - variance / effective).clamp_min(0)
            rho = float((signal / (signal + variance + 1e-16)).clamp(0, 1))
            if not math.isfinite(rho):
                raise ValueError("Nonfinite gradient-signal attenuation")
            state['zero_steps'] += int(rho == 0)
            state['last'] = dict(rho=rho, effective_gradient_count=effective,
                                 squared_signal=float(signal), gradient_variance=float(variance))
            packets[role], scales[role] = state, rho
        # Commit only after every selected role has a finite result.
        self.packets = packets
        return scales

    def state_dict(self):
        return deepcopy(dict(roles=list(self.roles), packets=self.packets))

    def validate(self, value, groups):
        if (not isinstance(value, dict) or set(value) != {'roles', 'packets'}
                or value['roles'] != list(self.roles)):
            raise ValueError("Gradient-signal checkpoint roles mismatch")
        expected = {role: [p for group in groups if group['role'] == role
                          for p in group['params']] for role in self.roles}
        if not isinstance(value['packets'], dict) or not set(value['packets']) <= set(expected):
            raise ValueError("Unknown gradient-signal role")
        for role, state in value['packets'].items():
            if (not expected[role] or not isinstance(state, dict)
                    or set(state) != {'means', 'second', 'weight', 'weight2', 'count', 'zero_steps', 'last'}):
                raise ValueError("Malformed gradient-signal packet")
            count, zero_steps = state['count'], state['zero_steps']
            if type(count) is not int or count < 1 or type(zero_steps) is not int or not 0 <= zero_steps <= count:
                raise ValueError("Bad gradient-signal counters")
            for key, expected_weight in (
                    ('weight', 1 - (7 / 8) ** count),
                    ('weight2', (1 / 8) ** 2 * (1 - (7 / 8) ** (2 * count)) / (1 - (7 / 8) ** 2))):
                weight = state[key]
                if (type(weight) not in (int, float) or not math.isfinite(weight) or not 0 < weight <= 1
                        or not math.isclose(weight, expected_weight, rel_tol=1e-12, abs_tol=1e-15)):
                    raise ValueError("Bad gradient-signal weights")
            if not isinstance(state['means'], list) or len(state['means']) != len(expected[role]):
                raise ValueError("Bad gradient-signal shape")
            tensors = state['means'] + [state['second']]
            if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float64
                   or not bool(torch.isfinite(t).all()) for t in tensors):
                raise ValueError("Bad gradient-signal tensors")
            if (state['second'].ndim != 0 or bool(state['second'] < 0)
                    or any(m.shape != p.shape for m, p in zip(state['means'], expected[role]))):
                raise ValueError("Bad gradient-signal second moment")
            receipt = state['last']
            if (not isinstance(receipt, dict)
                    or set(receipt) != {'rho', 'effective_gradient_count', 'squared_signal', 'gradient_variance'}
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in receipt.values())
                    or receipt['rho'] > 1
                    or not math.isclose(receipt['effective_gradient_count'], state['weight'] ** 2 / state['weight2'],
                                        rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError("Bad gradient-signal receipt")

    def load(self, value, groups):
        self.validate(value, groups)
        self.packets = deepcopy(value['packets'])
        for role, state in self.packets.items():
            params = [p for group in groups if group['role'] == role for p in group['params']]
            state['means'] = [m.to(device=p.device, dtype=torch.float64) for m, p in zip(state['means'], params)]
            state['second'] = state['second'].to(device=params[0].device, dtype=torch.float64)

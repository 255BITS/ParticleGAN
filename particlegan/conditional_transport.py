"""Explicit output-marginal consumer of the unchanged public local-v2 losses.

Conditioning is checked and documented but is not a transport coordinate. Thus
permuting output rows leaves this marginal loss unchanged: paired correctness
must still come from the host's original objectives and numerical identity gates.
"""
import torch
import math


class OutputMarginalTransport:
    def __init__(self, recipe):
        self.recipe = recipe
        self.calls = 0
        self.active_calls = 0
        self.loss_sum = 0.0
        self.maximum_loss = 0.0

    def add(self, total, fake, real, *, conditioning=None):
        if not (self.recipe.kinetic_transport_weight or self.recipe.kinetic_transport_local_weight):
            return total  # Disabled consumers impose no shape or hook requirement.
        if (fake.shape != real.shape or fake.ndim != 2 or len(fake) < 2
                or real.device != fake.device or fake.dtype != real.dtype
                or (conditioning is not None and (conditioning.ndim != 2
                    or len(conditioning) != len(fake) or conditioning.device != fake.device
                    or not bool(torch.isfinite(conditioning).all())))):
            raise ValueError('output marginal transport requires matched panels and conditioning')
        self.calls += 1
        term = fake.new_zeros(())
        if self.recipe.kinetic_transport_weight:
            term = term + self.recipe.kinetic_transport_loss(fake, real)
        if self.recipe.kinetic_transport_local_weight:
            term = term + self.recipe.kinetic_transport_local_loss(fake, real)
        value = float(term.detach())
        self.active_calls += 1
        self.loss_sum += value
        self.maximum_loss = max(self.maximum_loss, value)
        return total + term

    def state_dict(self):
        return dict(schema_version=1, consumer='output_marginal_v1',
                    conditioning_used_in_distance=False, calls=self.calls,
                    active_calls=self.active_calls, loss_sum=self.loss_sum,
                    maximum_loss=self.maximum_loss)

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or any(state[key] != expected[key] for key in
                       ('schema_version', 'consumer', 'conditioning_used_in_distance'))
                or any(type(state[key]) is not int or state[key] < 0 for key in ('calls', 'active_calls'))
                or state['active_calls'] > state['calls']
                or any(type(state[key]) not in (int, float) or not math.isfinite(state[key])
                       or state[key] < 0 for key in ('loss_sum', 'maximum_loss'))):
            raise ValueError('invalid output marginal transport checkpoint')
        for key in ('calls', 'active_calls', 'loss_sum', 'maximum_loss'):
            setattr(self, key, state[key])

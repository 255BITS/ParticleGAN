"""Detached current-batch confidence for an empirical transport force.

Block MMD supplies a signal and a plug-in standard error, not a calibrated
hypothesis test for dependent GAN outputs. No critic, history, new sample,
evaluation feedback, target geometry or random permutation enters.
"""
import math

import torch


@torch.no_grad()
def block_mmd_mobility(fake, real):
    """One-SE positive signal fraction with deterministic blocks of at most 32.

    All rows participate. Uneven panels use nearly equal contiguous blocks.
    The RBF width is detached real-batch total coordinate variance. Blocks
    average h(i,j)=k(x_i,x_j)+k(y_i,y_j)-k(x_i,y_j)-k(y_i,x_j), i != j.
    Fewer than four rows cannot estimate between-block uncertainty: mobility 0.
    """
    if (fake.ndim < 2 or fake.shape != real.shape or len(fake) < 2
            or not fake.is_floating_point() or fake.dtype != real.dtype
            or fake.device != real.device
            or not bool(torch.isfinite(fake).all() and torch.isfinite(real).all())):
        raise ValueError('confidence mobility requires matching finite floating panels')
    x, y = fake.detach().flatten(1), real.detach().flatten(1)
    eps = torch.finfo(x.dtype).eps
    width = (y - y.mean(0)).square().sum(1).mean().clamp_min(eps)
    blocks = min(max(2, math.ceil(len(x) / 32)), len(x) // 2)

    def statistic(a, b):
        # Batched direct distances avoid a matrix-multiplication precision path.
        xx = torch.cdist(a, a, compute_mode='donot_use_mm_for_euclid_dist').square()
        yy = torch.cdist(b, b, compute_mode='donot_use_mm_for_euclid_dist').square()
        xy = torch.cdist(a, b, compute_mode='donot_use_mm_for_euclid_dist').square()
        h = torch.exp(-xx / (2 * width)) + torch.exp(-yy / (2 * width))
        cross = torch.exp(-xy / (2 * width))
        h = h - cross - cross.transpose(-1, -2)
        size = a.shape[-2]
        return (h.sum((-2, -1)) - h.diagonal(dim1=-2, dim2=-1).sum(-1)) / (size * (size - 1))

    if blocks < 2:
        zero = x.new_zeros(())
        return zero, dict(blocks=blocks, rows=len(x), signal=0., standard_error=0.)
    if len(x) % blocks == 0:
        values = statistic(x.reshape(blocks, -1, x.shape[1]), y.reshape(blocks, -1, y.shape[1]))
    else:
        values = torch.stack([statistic(a, b) for a, b in
                              zip(x.tensor_split(blocks), y.tensor_split(blocks))])
    signal = values.mean()
    error = values.std(unbiased=True) / math.sqrt(blocks)
    mobility = (signal - error).clamp_min(0) / (signal.clamp_min(0) + error + eps)
    return mobility, dict(blocks=blocks, rows=len(x), signal=float(signal), standard_error=float(error))


class TransportMobility:
    """Audit counters only: the next multiplier never depends on stored state."""
    def __init__(self):
        self.stats = dict(calls=0, zero_calls=0, high_calls=0, rows=0, blocks=0,
                          mobility_sum=0., signal_sum=0., standard_error_sum=0.)

    def observe(self, fake, real):
        mobility, evidence = block_mmd_mobility(fake, real)
        value = float(mobility)
        self.stats['calls'] += 1
        self.stats['zero_calls'] += int(value == 0)
        self.stats['high_calls'] += int(value >= .9)
        for name in ('rows', 'blocks'):
            self.stats[name] += evidence[name]
        self.stats['mobility_sum'] += value
        self.stats['signal_sum'] += evidence['signal']
        self.stats['standard_error_sum'] += evidence['standard_error']
        return mobility

    def state_dict(self):
        return dict(schema_version=1, mode='block_mmd_v1', stats=dict(self.stats))

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or state['schema_version'] != 1 or state['mode'] != expected['mode']
                or not isinstance(state['stats'], dict) or state['stats'].keys() != self.stats.keys()):
            raise ValueError('invalid transport mobility checkpoint')
        stats = state['stats']
        if (any(type(stats[k]) is not int or stats[k] < 0 for k in ('calls', 'zero_calls', 'high_calls', 'rows', 'blocks'))
                or stats['zero_calls'] > stats['calls']
                or stats['high_calls'] > stats['calls'] - stats['zero_calls']
                or any(type(stats[k]) not in (int, float) or not math.isfinite(stats[k])
                       for k in ('mobility_sum', 'signal_sum', 'standard_error_sum'))
                or not 0 <= stats['mobility_sum'] <= stats['calls']
                or stats['standard_error_sum'] < 0):
            raise ValueError('invalid transport mobility counters')
        self.stats = dict(stats)

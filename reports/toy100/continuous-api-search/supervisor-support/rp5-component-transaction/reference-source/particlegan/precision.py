"""Experimental reversible precision, driven only by game/update evidence."""
from copy import deepcopy
import math
import torch


class ReversiblePrecision:
    """A contraction must finish before reducing mobility; every close can reopen.

    The reference EMA and all smoothing windows have fixed update time constants.
    Neither training age nor an evaluator horizon selects a learning rate.
    """
    def __init__(self, critic, variant="rp1"):
        self.reference = deepcopy(critic).eval().requires_grad_(False)
        self.state = dict(variant=variant, open=True, gap=None, gap_s=None,
                          velocity=0., contraction_peak=0., contracting=0,
                          activity=None, activity_peak=0., quiet_gap=None,
                          quiet_activity=None, quiet=0, shock=0, updates=0,
                          openings=1, closings=0, event="initial_open")
        if variant == "rp5":
            self.state["velocity_s"] = 0.

    def scales(self):
        gain = (1. if self.state['variant'] == 'rp5' else .2) if self.state['open'] else 0.
        return .01 + .99 * gain, .05 + .95 * gain

    @staticmethod
    def parameters(optimizer):
        return [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]

    @torch.no_grad()
    def update_activity(self, optimizer, before):
        terms = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and id(p) in before:
                    terms.append(((p - before[id(p)]) / group['lr']).flatten())
        return float(torch.cat(terms).square().mean().sqrt())

    def observe(self, critic, real, activity):
        with torch.enable_grad():
            x = real.detach().clone().requires_grad_(True)
            y = critic(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            g = torch.autograd.grad(y.sum(), x)[0].detach()
            x = real.detach().clone().requires_grad_(True)
            y = self.reference(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            ref = torch.autograd.grad(y.sum(), x)[0].detach()
            gap = float((g - ref).square().mean())
        with torch.no_grad():
            for old, new in zip(self.reference.parameters(), critic.parameters()):
                old.lerp_(new, .001)
            for old, new in zip(self.reference.buffers(), critic.buffers()):
                old.copy_(new)
        self.advance(gap, activity)

    def advance(self, gap, activity):
        s = self.state
        if not all(math.isfinite(v) and v >= 0 for v in (gap, activity)):
            raise ValueError('nonfinite precision evidence')
        s['updates'] += 1
        s['event'] = None
        prev = gap if s['gap_s'] is None else s['gap_s']
        s['gap'] = gap
        s['gap_s'] = .99 * prev + .01 * gap
        s['velocity'] = s['gap_s'] - prev
        velocity = s['velocity']
        if s['variant'] == 'rp5':
            s['velocity_s'] = .95 * s['velocity_s'] + .05 * velocity
            velocity = s['velocity_s']
        s['activity'] = activity if s['activity'] is None else .9 * s['activity'] + .1 * activity
        if s['open']:
            s['activity_peak'] = max(s['activity_peak'] * .9997, s['activity'])
            s['contracting'] = s['contracting'] + 1 if velocity < 0 else 0
            s['contraction_peak'] = max(s['contraction_peak'], -velocity)
            calm = (s['contracting'] >= 50 and s['contraction_peak'] > 0
                    and -velocity < .25 * s['contraction_peak']
                    and s['activity'] < .5 * s['activity_peak'])
            s['quiet'] = s['quiet'] + 1 if calm else 0
            if s['quiet'] >= 25:
                s.update(open=False, quiet_gap=max(s['gap_s'], 1e-12),
                         quiet_activity=max(s['activity'], 1e-12), quiet=0,
                         shock=0, event='close')
                s['closings'] += 1
        else:
            ratio = s['gap_s'] / s['quiet_gap']
            innovation = s['activity'] / s['quiet_activity']
            # Reference disagreement alone is ambiguous. Require active updates
            # and a growing excursion as well, with a short noise-rejection dwell.
            shock = ratio > 4 and innovation > 2 and velocity > 0
            s['shock'] = s['shock'] + 1 if shock else 0
            if ratio < 2 and innovation < 2:
                s['quiet_gap'] = .99 * s['quiet_gap'] + .01 * max(s['gap_s'], 1e-12)
                s['quiet_activity'] = .99 * s['quiet_activity'] + .01 * max(s['activity'], 1e-12)
            if s['shock'] >= 5:
                s.update(open=True, contraction_peak=0., contracting=0,
                         activity_peak=s['activity'], quiet=0, shock=0, event='reopen')
                s['openings'] += 1

    def state_dict(self):
        return deepcopy(dict(state=self.state, reference=self.reference.state_dict()))

    def load_state_dict(self, value):
        if set(value) != {'state', 'reference'} or value['state'].keys() != self.state.keys():
            raise ValueError('invalid precision checkpoint')
        if value['state']['variant'] != self.state['variant']:
            raise ValueError('precision variant differs')
        self.reference.load_state_dict(value['reference'])
        self.state = deepcopy(value['state'])

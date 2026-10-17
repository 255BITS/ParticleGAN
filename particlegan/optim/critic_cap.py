"""Opt-in finite critic displacement guard on caller-owned BCAP panels."""
from copy import deepcopy
import math

import torch

from .dualnorm import NormalizedOptimizer


class CriticCapOptimizer(NormalizedOptimizer):
    """One base update, at most nine same-panel finite cap trials.

    The paired public CriticPenalty binds every actual real/fake panel. No
    sampler or evaluation panel is called. If the current maximum slope is
    above kappa, its current value is the non-increase bound. Optimizer clocks
    advance once even when every trial is rejected. Checkpoints are supported
    at completed-step boundaries; pending caller closures are not serialized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cap_panels = []
        self.critic_cap_stats = dict(steps=0, probes=0, attempted_scales=0, accepted_steps=0,
            damped_steps=0, rejected_steps=0, backtracks=0, accepted_scale_sum=0.,
            minimum_scale=1., maximum_before=0., maximum_after=0.)

    def bind_cap_panel(self, evaluator, kappa):
        if not callable(evaluator) or not math.isfinite(kappa) or kappa <= 0:
            raise ValueError('finite cap requires a deterministic panel and positive kappa')
        self._cap_panels.append((evaluator, float(kappa)))

    def _measure(self):
        cpu = torch.get_rng_state()
        devices = sorted({p.device.index for g in self.param_groups for p in g['params']
                          if p.device.type == 'cuda'})
        cuda = {d: torch.cuda.get_rng_state(d) for d in devices}
        buffers = [(b, b.detach().clone()) for b in self.critic.buffers()]
        try:
            with torch.enable_grad():
                values = [float(evaluator().detach()) for evaluator, _ in self._cap_panels]
            if not values or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('finite cap panel must return finite nonnegative scalar slopes')
        finally:
            rng_changed = not torch.equal(cpu, torch.get_rng_state())
            rng_changed |= any(not torch.equal(s, torch.cuda.get_rng_state(d)) for d, s in cuda.items())
            buffer_changed = any(not torch.equal(b, old) for b, old in buffers)
            if rng_changed or buffer_changed:
                torch.set_rng_state(cpu)
                for d, state in cuda.items():
                    torch.cuda.set_rng_state(state, d)
                with torch.no_grad():
                    for b, old in buffers:
                        b.copy_(old)
                raise ValueError('finite cap probe consumed ambient RNG or mutated critic buffers')
        self.critic_cap_stats['probes'] += 1
        return values

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or not self._cap_panels:
            raise ValueError('finite cap step requires the paired critic penalty first')
        parameters = [p for g in self.param_groups for p in g['params']]
        originals = [p.detach().clone() for p in parameters]
        try:
            before = self._measure()
        except Exception:
            self._cap_panels.clear()
            raise
        bounds = [max(kappa, v) for v, (_, kappa) in zip(before, self._cap_panels)]
        prior_lr_max = self.record.lr_max
        result = super().step()
        proposed = [p.detach().clone() for p in parameters]
        displacements = [new-old for new, old in zip(proposed, originals)]
        accepted = False
        scale = 0.
        after = before
        try:
            for trial in range(9):
                alpha = 2. ** -trial
                # Preserve an accepted full step bitwise, including zero updates.
                if trial:
                    for p, old, delta in zip(parameters, originals, displacements):
                        p.copy_(old + alpha * delta)
                measured = self._measure()
                self.critic_cap_stats['attempted_scales'] += 1
                if all(v <= bound for v, bound in zip(measured, bounds)):
                    accepted, scale, after = True, alpha, measured
                    break
                self.critic_cap_stats['backtracks'] += int(trial < 8)
            if not accepted:
                for p, old in zip(parameters, originals):
                    p.copy_(old)
            stats = self.critic_cap_stats
            stats['steps'] += 1
            stats['accepted_steps'] += int(accepted)
            stats['damped_steps'] += int(scale < 1)
            stats['rejected_steps'] += int(not accepted)
            stats['accepted_scale_sum'] += scale
            stats['minimum_scale'] = min(stats['minimum_scale'], scale)
            stats['maximum_before'] = max(stats['maximum_before'], max(before))
            stats['maximum_after'] = max(stats['maximum_after'], max(after))
            # The base call owns one clock; its LR observation reflects the
            # accepted finite fraction rather than the rejected full proposal.
            self.record.lr_last = scale * max(float(g['lr']) for g in self.param_groups)
            self.record.lr_max = max(prior_lr_max, self.record.lr_last)
        except Exception:
            for p, old in zip(parameters, originals):
                p.copy_(old)
            raise
        finally:
            self._cap_panels.clear()
        return result

    def state_dict(self):
        if self._cap_panels:
            raise ValueError('checkpoint finite cap at completed-step boundaries')
        result = super().state_dict()
        result['critic_cap'] = dict(schema=1, stats=deepcopy(self.critic_cap_stats))
        return result

    def validate_state_dict(self, saved):
        base = dict(saved)
        meta = base.pop('critic_cap', None)
        if (not isinstance(meta, dict) or set(meta) != {'schema', 'stats'} or meta['schema'] != 1
                or not isinstance(meta['stats'], dict) or set(meta['stats']) != set(self.critic_cap_stats)):
            raise ValueError('invalid finite cap checkpoint')
        for key, value in meta['stats'].items():
            if type(self.critic_cap_stats[key]) is int:
                if type(value) is not int or value < 0:
                    raise ValueError('invalid finite cap counter')
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid finite cap statistic')
        if meta['stats']['minimum_scale'] > 1:
            raise ValueError('invalid finite cap scale')
        super().validate_state_dict(base)

    def load_state_dict(self, saved):
        self.validate_state_dict(saved)
        super().load_state_dict(saved)
        self.critic_cap_stats = deepcopy(saved['critic_cap']['stats'])
        self._cap_panels.clear()

"""Strict common-descent blend at full scale, without finite loss probes."""
from copy import deepcopy

import torch

from .constraint_geometry import ConstraintGeometryOptimizer
from .dualnorm import NormalizedOptimizer
from .strict_progress import common_descent


class DirectionBlendOptimizer(ConstraintGeometryOptimizer):
    """Ablation of StrictProgressOptimizer: identical direction, alpha always one.

    Opposed normals restore the original tensors. Otherwise apply the rounded
    full blend even when finite objective values increase. Inactive updates keep
    the base tensors bitwise; no evaluator is bound or called.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction_blend_stats = dict(conflict_steps=0, blended_steps=0,
            pareto_stalls=0, retained_norm_ratio_sum=0., max_retained_norm_ratio=0.)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or self._protected is None:
            raise ValueError('direction blend step requires protected-loss backward')
        parameters = self._parameters()
        originals = [p.detach().clone() for p in parameters]
        normals = self._protected.double()
        result = NormalizedOptimizer.step(self)
        displacement = torch.cat([(p - old).flatten() for p, old in zip(parameters, originals)])
        before = normals @ displacement.double()
        stats, blend = self.constraint_geometry_stats, self.direction_blend_stats
        try:
            if bool((before > 0).any()):
                blend['conflict_steps'] += 1
                direction, possible = common_descent(displacement, normals)
                blend['blended_steps' if possible else 'pareto_stalls'] += 1
                offset = 0
                for p, old in zip(parameters, originals):
                    size = p.numel()
                    p.copy_(old + direction[offset:offset + size].view_as(p) if possible else old)
                    offset += size
                applied = torch.cat([(p - old).flatten() for p, old in zip(parameters, originals)])
                ratio = float(applied.double().norm() / displacement.double().norm())
                blend['retained_norm_ratio_sum'] += ratio
                blend['max_retained_norm_ratio'] = max(blend['max_retained_norm_ratio'], ratio)
            applied = torch.cat([(p - old).flatten() for p, old in zip(parameters, originals)])
            stats['steps'] += 1
            stats['projected_steps'] += int(not torch.equal(applied, displacement))
            stats['max_derivative_before'] = max(stats['max_derivative_before'], float(before.max()))
            stats['max_derivative_after'] = max(stats['max_derivative_after'], float((normals @ applied.double()).max()))
        except Exception:
            for p, old in zip(parameters, originals):
                p.copy_(old)
            raise
        finally:
            self._protected = None
        return result

    def state_dict(self):
        saved = super().state_dict()
        saved['direction_blend'] = dict(schema=1, stats=deepcopy(self.direction_blend_stats))
        return saved

    def validate_state_dict(self, saved):
        base = dict(saved)
        meta = base.pop('direction_blend', None)
        if (not isinstance(meta, dict) or set(meta) != {'schema', 'stats'}
                or meta['schema'] != 1 or not isinstance(meta['stats'], dict)
                or set(meta['stats']) != set(self.direction_blend_stats)):
            raise ValueError('invalid direction blend checkpoint')
        for key, value in meta['stats'].items():
            if isinstance(self.direction_blend_stats[key], int):
                if type(value) is not int or value < 0:
                    raise ValueError('invalid direction blend counter')
            elif type(value) not in (int, float) or value < 0 or not torch.isfinite(torch.tensor(value)):
                raise ValueError('invalid direction blend statistic')
        super().validate_state_dict(base)

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        self.direction_blend_stats = deepcopy(saved['direction_blend']['stats'])

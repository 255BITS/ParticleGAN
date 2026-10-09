"""Bounded common descent and same-batch finite realization on conflict steps."""
from copy import deepcopy

import torch

from .constraint_geometry import ConstraintGeometryOptimizer, project_nonascent
from .dualnorm import NormalizedOptimizer


def common_descent(displacement, normals):
    """Half feasible motion, half unit-gradient common descent; zero at opposition."""
    a = normals.double()
    lengths = a.norm(dim=1)
    a = a[lengths > 0] / lengths[lengths > 0, None]
    if not 1 <= len(a) <= 2:
        raise ValueError('strict progress needs one or two nonzero normals')
    common = a.mean(0)  # Minimum-norm convex combination of one/two unit normals.
    if float(common.norm()) <= 1e-12:
        return torch.zeros_like(displacement), False
    protected = project_nonascent(displacement, normals).double()
    descent = -displacement.double().norm() * common / common.norm()
    direction = (protected + descent) / 2
    return direction.to(displacement), True


class StrictProgressOptimizer(ConstraintGeometryOptimizer):
    """One base optimizer clock; at most nine deterministic finite trial forwards.

    A supplied evaluator must reuse the actual training batch/latent perturbations,
    protected losses and fixed critic, without sampling or mutable forward buffers.
    Inactive steps never call it and retain the base step bitwise. Pending-state
    restore requires rebinding this caller-owned evaluator before step().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._protected_values = self._protected_evaluator = None
        self.strict_progress_stats = dict(conflict_steps=0, accepted_steps=0,
            rejected_steps=0, pareto_stalls=0, probes=0, backtracks=0,
            min_accepted_scale=1., accepted_scale_sum=0., retained_norm_ratio_sum=0.,
            max_retained_norm_ratio=0., protected_decrease_0_sum=0.,
            protected_decrease_1_sum=0., max_accepted_armijo_violation=0.)

    def bind_protected_evaluator(self, evaluator):
        if not callable(evaluator):
            raise ValueError('strict progress requires a deterministic protected evaluator')
        self._protected_evaluator = evaluator

    def bind_protected_losses(self, losses, *, protected_evaluator=None):
        losses = tuple(losses)
        self.bind_protected_evaluator(protected_evaluator)
        super().bind_protected_losses(losses)
        self._protected_values = torch.stack([loss.detach() for loss in losses]).double()

    def _evaluate(self):
        cpu = torch.get_rng_state()
        device = self._parameters()[0].device
        cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        try:
            values = tuple(self._protected_evaluator())
            if len(values) != len(self._protected_values):
                raise ValueError('protected evaluator changed objective count')
            result = torch.stack([value.detach() for value in values]).double()
            if result.shape != self._protected_values.shape:
                raise ValueError('protected evaluator must return scalar losses')
        finally:
            changed = not torch.equal(cpu, torch.get_rng_state())
            if cuda is not None:
                changed |= not torch.equal(cuda, torch.cuda.get_rng_state(device))
            if changed:
                torch.set_rng_state(cpu)
                if cuda is not None:
                    torch.cuda.set_rng_state(cuda, device)
                raise ValueError('protected evaluator consumed ambient RNG')
        self.strict_progress_stats['probes'] += 1
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if (closure is not None or self._protected is None or self._protected_values is None
                or self._protected_evaluator is None):
            raise ValueError('strict progress step requires protected backward and evaluator')
        parameters = self._parameters()
        originals = [p.detach().clone() for p in parameters]
        normals = self._protected.double()

        def put(direction):
            offset = 0
            for p, old in zip(parameters, originals):
                size = p.numel()
                p.copy_(old + direction[offset:offset + size].view_as(p))
                offset += size

        def actual():
            return torch.cat([(p - old).flatten() for p, old in zip(parameters, originals)])

        result = NormalizedOptimizer.step(self)
        displacement = actual()
        before = normals @ displacement.double()
        conflict = bool((before > 0).any())
        stats, progress = self.constraint_geometry_stats, self.strict_progress_stats
        try:
            if conflict:
                progress['conflict_steps'] += 1
                direction, possible = common_descent(displacement, normals)
                accepted = False
                if not possible:
                    progress['pareto_stalls'] += 1
                else:
                    put(torch.zeros_like(displacement))
                    baseline = self._evaluate()
                    tolerance = 8 * torch.finfo(displacement.dtype).eps
                    if not torch.allclose(baseline, self._protected_values, rtol=tolerance, atol=tolerance):
                        raise ValueError('protected evaluator does not replay the original loss')
                    for trial in range(9):
                        scale = 2. ** -trial
                        put(direction * scale)
                        applied = actual()
                        derivatives = normals @ applied.double()
                        values = self._evaluate()
                        bound = baseline + 1e-4 * derivatives
                        nonzero = normals.norm(dim=1) > 0
                        if (bool(torch.isfinite(values).all())
                                and bool((derivatives[nonzero] < 0).all())
                                and bool((values <= bound).all())):
                            accepted = True
                            progress['accepted_steps'] += 1
                            progress['min_accepted_scale'] = min(progress['min_accepted_scale'], scale)
                            progress['accepted_scale_sum'] += scale
                            ratio = float(applied.double().norm() / displacement.double().norm())
                            progress['retained_norm_ratio_sum'] += ratio
                            progress['max_retained_norm_ratio'] = max(progress['max_retained_norm_ratio'], ratio)
                            for index, decrease in enumerate(baseline - values):
                                progress[f'protected_decrease_{index}_sum'] += float(decrease)
                            progress['max_accepted_armijo_violation'] = max(
                                progress['max_accepted_armijo_violation'], float((values - bound).max()))
                            break
                        progress['backtracks'] += int(trial < 8)
                if not accepted:
                    put(torch.zeros_like(displacement))
                    progress['rejected_steps'] += 1
            # Inactive: no recomposition, no evaluator, no additional optimizer call.
            applied = actual()
            stats['steps'] += 1
            stats['projected_steps'] += int(not torch.equal(applied, displacement))
            stats['max_derivative_before'] = max(stats['max_derivative_before'], float(before.max()))
            stats['max_derivative_after'] = max(stats['max_derivative_after'], float((normals @ applied.double()).max()))
        except Exception:
            for p, old in zip(parameters, originals):
                p.copy_(old)
            raise
        finally:
            self._protected = self._protected_values = self._protected_evaluator = None
        return result

    def state_dict(self):
        result = super().state_dict()
        result['strict_progress'] = dict(schema=1, stats=deepcopy(self.strict_progress_stats),
            pending_values=None if self._protected_values is None else self._protected_values.clone())
        return result

    def validate_state_dict(self, saved):
        base = dict(saved)
        meta = base.pop('strict_progress', None)
        if (not isinstance(meta, dict) or set(meta) != {'schema', 'stats', 'pending_values'}
                or meta['schema'] != 1 or not isinstance(meta['stats'], dict)
                or set(meta['stats']) != set(self.strict_progress_stats)):
            raise ValueError('invalid strict progress checkpoint')
        for key, value in meta['stats'].items():
            if isinstance(self.strict_progress_stats[key], int):
                if type(value) is not int or value < 0:
                    raise ValueError('invalid strict progress counter')
            elif type(value) not in (int, float) or value < 0 or not torch.isfinite(torch.tensor(value)):
                raise ValueError('invalid strict progress statistic')
        values = meta['pending_values']
        pending = base.get('constraint_geometry', {}).get('pending')
        if ((values is None) != (pending is None) or (values is not None and
                (not isinstance(values, torch.Tensor) or values.ndim != 1 or len(values) != len(pending)
                 or not bool(torch.isfinite(values).all())))):
            raise ValueError('invalid strict progress pending losses')
        super().validate_state_dict(base)

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        meta = saved['strict_progress']
        self.strict_progress_stats = deepcopy(meta['stats'])
        self._protected_values = None if meta['pending_values'] is None else meta['pending_values'].to(self._parameters()[0].device)
        self._protected_evaluator = None

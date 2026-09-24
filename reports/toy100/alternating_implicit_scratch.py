"""Safeguarded implicit response of a frozen-metric alternating game map.

The first evaluation is the actual host's ordinary Adam D step followed by
its G/prior step. Each moment is updated once. Let D1 be that rounded D
proposal, D0 its base, a=FD(D0,G0), and P the resulting fixed Adam metric.
Every later same-batch field query at (D,G) returns

    Ftilde_D = FD(D,G)
    Dvirtual = round(D1 + (D-D0) - P_D * (FD(D,G)-a))
    Ftilde_G = FG(Dvirtual,G).

Thus the base query reproduces the real alternating host exactly. This is
an LR-dependent alternating-map field, not the simultaneous GAN vector
field. The anchor preserves roots only up to ordinary Adam/dtype rounding.
The existing bounded GMRES and nonlinear residual guard act on Ftilde,
with their original settings; no new gain, objective or time schedule.
"""

from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from reports.toy100 import implicit_extra_scratch as parent


METHOD = 'same_sample_alternating_map_implicit_response'


class AlternatingImplicitRecorder(parent.ImplicitExtraRecorder):
    def __init__(self, explicit_control=False, **options):
        super().__init__(**options)
        self.explicit_control = explicit_control
        self.preconditioner = {}
        self.base_replays = []
        self.anchors = []
        self.base_query_outer = -1

    def phases(self, step, opt_d, opt_g, local):
        if not self.explicit_control or not self.enabled or step < self.start_step:
            yield from super().phases(step, opt_d, opt_g, local)
            return
        # Test-only control exercises the same phase-0 callbacks, then keeps
        # the exact rounded ordinary D-then-G point without a solve/query.
        iterator = super().phases(step, opt_d, opt_g, local)
        try:
            phase = next(iterator)
            if phase != 0:
                raise RuntimeError('ordinary alternating control missed base phase')
            yield phase
            if self.pending:
                raise RuntimeError('ordinary alternating control incomplete')
            with torch.no_grad():
                for p, value in self.ordinary_proposal.items():
                    p.copy_(value)
            self.phase = None
            self.base, self.point = {}, {}
            self.outer_steps += 1
            if self.accounting is not None:
                self.accounting(self.rows[opt_d]['calls'], self.outer_steps)
        finally:
            iterator.close()

    def _solve(self, scale, q0):
        if self.base_query_outer != self.outer_steps:
            expected = {p: value.clone() for p, value in self.field.items()}
            replay, actual = yield from self._evaluate(torch.zeros_like(q0), 'base_alternating_replay')
            if torch.count_nonzero(actual) or not torch.equal(replay, q0):
                raise RuntimeError('alternating base field replay differs')
            if any(not torch.equal(self.field[p], value) for p, value in expected.items()):
                raise RuntimeError('alternating base raw gradient replay differs')
            self.base_replays.append(dict(outer_step=self.outer_steps + 1,
                                          raw_gradients_bitwise_equal=True,
                                          metric_field_bitwise_equal=True))
            self.base_query_outer = self.outer_steps
        return (yield from super()._solve(scale, q0))

    def _record_metric(self, optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                denominator = (state['exp_avg_sq'] /
                               (1 - group['betas'][1] ** float(state['step']))).sqrt() + group['eps']
                if not torch.isfinite(denominator).all() or not (denominator > 0).all():
                    raise FloatingPointError('invalid alternating Adam metric')
                self.metric[p] = denominator
                self.preconditioner[p] = group['lr'] / denominator.double()

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self.passthrough:
            return ordinary_step(optimizer, closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError('alternating field outside declared point')
        if optimizer in self.pending or optimizer is not self.optimizers[len(self.pending)]:
            raise RuntimeError('alternating field order changed')
        gradients = {}
        for group in optimizer.param_groups:
            if (group['betas'][0] != 0 or group.get('weight_decay', 0) or
                any(group.get(key, False) for key in
                    ('amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay', 'fused'))):
                raise ValueError('alternating implicit response requires ordinary CPU Adam, beta1=0')
            if not math.isfinite(group['lr']) or group['lr'] <= 0:
                raise ValueError('invalid constant rate')
            for p in group['params']:
                if p.device.type != 'cpu' or p.grad is None or not torch.isfinite(p.grad).all():
                    raise FloatingPointError('missing/nonfinite/non-CPU alternating gradient')
                gradients[p] = p.grad.detach().clone()
        self.pending[optimizer] = gradients
        if optimizer is self.optimizers[0]:
            if any(not torch.equal(p, value) for p, value in self.point.items()):
                raise RuntimeError('parameter moved before alternating D capture')
            if self.phase == 0:
                self.base_d_field = {p: value.clone() for p, value in gradients.items()}
                ordinary_step(optimizer)
                self._record_metric(optimizer)
                self.base_d_proposal = {p: p.detach().clone() for p in gradients}
                anchor = sum(float(((self.base_d_proposal[p].double() - self.base[p].double()) +
                                    self.preconditioner[p] * gradients[p].double()).square().sum())
                             for p in gradients)
                self.anchors.append(dict(outer_step=self.outer_steps + 1,
                                         d_rounding_anchor_parameter_norm=math.sqrt(anchor)))
                self.virtual_d = self.base_d_proposal
            else:
                self.virtual_d = {}
                for p, gradient in gradients.items():
                    virtual = (self.base_d_proposal[p].double() +
                               (self.point[p].double() - self.base[p].double()) -
                               self.preconditioner[p] *
                               (gradient.double() - self.base_d_field[p].double())).to(p.dtype)
                    if not torch.isfinite(virtual).all():
                        raise FloatingPointError('nonfinite virtual D response')
                    p.copy_(virtual)
                    self.virtual_d[p] = virtual
            return
        for p, value in self.point.items():
            expected = self.virtual_d.get(p, value)
            if not torch.equal(p, expected):
                raise RuntimeError('parameter moved before alternating G capture')
        self.field = {p: value for values in self.pending.values() for p, value in values.items()}
        if self.phase == 0:
            ordinary_step(optimizer)
            self._record_metric(optimizer)
            self.root_metric = self._flat(self.preconditioner).sqrt()
            self.ordinary_proposal = self._copy_parameters()
            for p, value in self.base.items():
                p.copy_(value)
        else:
            for p in self.virtual_d:
                p.copy_(self.point[p])
        if any(not torch.equal(p, value) for p, value in self.point.items()):
            raise RuntimeError('query parameters not restored after alternating G capture')
        self.joint_points_verified += 1
        for target in self.optimizers:
            row = self.rows[target]
            rates = [float(group['lr']) for group in target.param_groups]
            if row['rates'] and rates != row['rates'][0]:
                raise RuntimeError('constant rate changed')
            row['calls'] += 1
            row['rates'].append(rates)
            row['diagnostics'].append([
                dict(parameters=sum(p.numel() for p in group['params']),
                     max_scaled_gradient=max(float((self.pending[target][p].double() /
                                                     self.metric[p].double()).abs().max())
                                             for p in group['params']))
                for group in target.param_groups])
        self.pending = {}

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
                     explicit_control=self.explicit_control,
                     field='FD(D,G), FG(round(D1+(D-D0)-PD*(FD(D,G)-FD0)),G)',
                     base_evaluation='ordinary rounded Adam D step, then G/prior gradient and Adam step',
                     metric='post-base bias-corrected Adam metrics; fixed through every query and retry',
                     query_d_response='virtual anchored fixed-metric D step; query parameters restored after G capture',
                     base_replays=self.base_replays, rounding_anchors=self.anchors,
                     scope='Implicit response of an LR-dependent anchored alternating-map field; not simultaneous-field implicit Euler',
                     root_scope='Field zeros preserved only up to the explicitly recorded ordinary-Adam rounding anchor',
                     output_weights='ordinary alternating proposal control' if self.explicit_control else
                                    'accepted alternating-map nonlinear-residual-checked joint proposal')
        return value


@contextmanager
def alternating_implicit(task='mode_hold', **options):
    with patch.object(parent, 'ImplicitExtraRecorder', AlternatingImplicitRecorder):
        with parent.implicit_extra(task=task, **options) as value:
            yield value

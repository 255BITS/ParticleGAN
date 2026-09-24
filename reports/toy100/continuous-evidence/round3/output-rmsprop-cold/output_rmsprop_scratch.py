"""Experimental output-space RMSProp with a damped joint Jacobian pullback.

Use the actual training loss derivative at clean G outputs, normalize it per
particle/output coordinate, and lift that desired displacement through the
G+prior Jacobian in Adam's parameter metric. This changes the optimizer only.
No target or evaluator is consulted. Extra output moments are research state;
this small exact-Jacobian implementation is not a production implementation.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
from reports.toy100 import joint_functional_metric_scratch as joint


def lift_output_step(jacobian, diagonal, desired, relative_ridge=1e-3):
    jp = jacobian * diagonal[None, :]
    kernel = jp @ jacobian.T
    ridge = relative_ridge * kernel.diagonal().mean().clamp_min(1e-15)
    kernel.diagonal().add_(ridge)
    return jp.T @ torch.linalg.solve(kernel, desired)


class OutputRMSProp(joint.JointFunctionalMetric):
    def __init__(self, output_step=.00425, observe_only=False):
        super().__init__(output_step, observe_only)
        self.v = self.count = None
        self.receipt.update(policy='output_rmsprop_joint_pullback_v1', beta2=.999,
            relative_ridge=1e-3, output_epsilon=1e-8, nonlinear_relative_tolerance=.25,
            maximum_fit_backtracks=12,
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def register(self, model):
        super().register(model)
        record = self.records[-1]
        def capture(module, inputs, output):
            if not self.measuring and torch.is_grad_enabled() and output.requires_grad:
                def gradient(value):
                    record['output_gradient'] = value.detach().clone()
                output.register_hook(gradient)
        self.hooks.append(model.register_forward_hook(capture))

    def step(self, optimizer, original_step, closure=None):
        owned = {id(p) for g in optimizer.param_groups for p in g['params']}
        matches = [r for r in self.records if r['ids'] & owned]
        if not matches or self.observe_only:
            return super().step(optimizer, original_step, closure)
        record = matches[0]
        inputs = record['inputs']
        prior = [p for g in optimizer.param_groups if g.get('_comparison_prior') for p in g['params']][0]
        params = [p for g in optimizer.param_groups for p in g['params']]
        before = [p.detach().clone() for p in params]
        widths = [x.shape[1] for x in inputs]
        unique, inverse, counts = torch.unique(torch.cat(inputs, 1), dim=0, return_inverse=True, return_counts=True)
        args = tuple(unique.split(widths, 1))
        match = (args[-1][:, None, :] == prior.detach()[None, :, :]).all(-1)
        if not bool((match.sum(1) == 1).all()):
            raise ValueError('ambiguous particle identities')
        indices = match.long().argmax(1)
        if len(indices.unique()) != len(indices):
            raise ValueError('one conditioning row per particle required')
        gradient = record['output_gradient'].double()
        summed = torch.zeros(len(unique), gradient.shape[1], dtype=torch.float64)
        summed.index_add_(0, inverse, gradient)
        # Remove only batch averaging; duplicate observations are averaged.
        field = summed * len(gradient) / counts[:, None]
        if self.v is None:
            self.v = torch.zeros(len(prior), gradient.shape[1], dtype=torch.float64)
            self.count = torch.zeros(len(prior), dtype=torch.int64)
        self.v[indices] = .999*self.v[indices] + .001*field.square()
        self.count[indices] += 1
        variance = self.v[indices] / (1-.999**self.count[indices])[:, None]
        desired = -self.output_step*field/(variance.sqrt()+1e-8)
        weighted = (desired * counts.sqrt()[:, None]).flatten()
        def correction(proposal, diagonal, jacobian, ridge):
            delta = lift_output_step(jacobian, diagonal, weighted)
            self.predicted = (jacobian@delta).reshape_as(desired)/counts.sqrt()[:, None]
            return delta
        with torch.no_grad():
            base_outputs = record['model'](*args).detach()
        with patch.object(joint, 'metric_correction', correction):
            result = super().step(optimizer, original_step, closure)
        # Verify the local output model, not task quality or the game loss.
        # Restore all parameters together when the lift is nonlinear.
        after = [p.detach().clone() for p in params]
        factor = 1.
        self.measuring = True
        try:
            with torch.no_grad():
                for attempt in range(13):
                    for p, b, n in zip(params, before, after):
                        p.copy_(torch.lerp(b, n, factor) if factor < 1 else n)
                    actual = record['model'](*args[:-1], prior[indices])-base_outputs
                    error = ((actual-factor*self.predicted).square().sum(1)*counts).sum().sqrt()
                    scale = (self.predicted.square().sum(1)*counts).sum().sqrt()*factor
                    if float(error) <= .25*float(scale)+1e-8:
                        break
                    factor *= .5
                else:
                    factor = 0.
                    for p, b in zip(params, before):
                        p.copy_(b)
                    actual.zero_()
        finally:
            self.measuring = False
        self.receipt['updates'][-1].update(output_moment_steps=self.count.tolist(),
            fit_factor=factor, fit_trials=attempt+1,
            desired_output_rms=float(desired.square().sum(1).mean().sqrt()),
            accepted_output_rms=float((actual.square().sum(1)*counts).sum().div(len(gradient)).sqrt()))
        return result


@contextmanager
def output_rmsprop(*, output_step=.00425, observe_only=False, state=None):
    from reports.toy100 import functional_metric_scratch as base
    with patch.object(base, 'FunctionalMetric', OutputRMSProp):
        with base.functional_metric(output_step=output_step, observe_only=observe_only, state=state) as receipt:
            yield receipt

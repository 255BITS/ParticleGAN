"""Explicit new gradient-space adaptation of the recovered signal estimator.

For effective generator-role parameters only:
  adversarial: observe a; replace g by g + (rho(a) - 1) * a
  whole:       observe g; replace g by rho(g) * g
where a is the current original adversarial gradient and g is the current total
gradient. All other loss channels stay fresh. This is NOT the archived
post-normalization LR attenuation. Both arms then take one original DualNorm
step and direction blend against the current, unmodified protected normals.
"""
from copy import deepcopy

import torch

from .direction_blend import DirectionBlendOptimizer
from .gradient_signal import GradientSignal


class ComponentHistoryDirectionBlendOptimizer(DirectionBlendOptimizer):
    def __init__(self, *args, generator_history_mode, **kwargs):
        if generator_history_mode not in ('adversarial', 'whole'):
            raise ValueError("active component history requires adversarial or whole mode")
        super().__init__(*args, **kwargs)
        self.generator_history_mode = generator_history_mode
        self.gradient_signal = GradientSignal(('generator',))
        self._history_pending = False
        self._history_failed = False

    def _require_healthy_history(self):
        if self._history_failed:
            raise RuntimeError("component-history step failed; restore a valid checkpoint before reuse")

    def bind_protected_losses(self, losses, *, protected_evaluator=None):
        self._require_healthy_history()
        return super().bind_protected_losses(losses, protected_evaluator=protected_evaluator)

    def add_param_group(self, group):
        if hasattr(self, 'gradient_signal'):
            raise ValueError("construct component history after all parameter groups are defined")
        super().add_param_group(group)

    def backward_with_component_history(self, loss, *, adversarial_loss=None):
        self._require_healthy_history()
        if self._history_pending or self._protected is None:
            raise ValueError("component history requires fresh protected-loss binding")
        if (not isinstance(adversarial_loss, torch.Tensor) or adversarial_loss.ndim != 0
                or not adversarial_loss.requires_grad):
            raise ValueError("component history requires the explicit current adversarial scalar")
        parameters = [p for group in self.param_groups if group['role'] == 'generator'
                      for p in group['params']]
        if any(p.grad is not None and bool((p.grad != 0).any()) for p in self._parameters()):
            raise ValueError("component history does not support accumulated gradients")
        adversarial = None
        if self.generator_history_mode == 'adversarial' and parameters:
            adversarial = torch.autograd.grad(adversarial_loss, parameters, retain_graph=True, allow_unused=True)
        loss.backward()
        total = [p.grad for p in parameters]
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in total):
            raise ValueError("nonfinite component-history total gradient")
        observed = adversarial if adversarial is not None else total
        previous = self.gradient_signal.packets
        try:
            scales = self.gradient_signal.scale(self.param_groups, gradients={'generator': observed})
            rho = scales.get('generator', 1.)
            transformed = []
            for gradient, component in zip(total, observed):
                if component is None or rho == 1.:
                    transformed.append(gradient)
                elif gradient is None:
                    raise ValueError("adversarial gradient is absent from the total loss")
                elif self.generator_history_mode == 'whole':
                    transformed.append(gradient * rho)
                else:
                    transformed.append(gradient + component.detach() * (rho - 1.))
            if any(g is not None and not bool(torch.isfinite(g).all()) for g in transformed):
                raise ValueError("nonfinite component-history transformed gradient")
        except Exception:
            self.gradient_signal.packets = previous
            raise
        for parameter, gradient in zip(parameters, transformed):
            parameter.grad = gradient
        self._history_pending = True

    @torch.no_grad()
    def step(self, closure=None):
        self._require_healthy_history()
        try:
            if closure is not None or not self._history_pending:
                raise ValueError("component history step requires explicit component-history backward")
            result = super().step()
        except BaseException:
            # The inherited numerical step can fail after some parameters or
            # optimizer clocks have changed. Do not export or reuse that state.
            # A valid explicit restore is required; there is no hidden retry.
            self._history_failed = True
            raise
        self._history_pending = False
        return result

    def state_dict(self):
        self._require_healthy_history()
        saved = super().state_dict()
        saved['component_history'] = dict(
            schema=1, mode=self.generator_history_mode, placement='before_normalization',
            pending=self._history_pending, signal=self.gradient_signal.state_dict())
        return saved

    def validate_state_dict(self, saved):
        if not isinstance(saved, dict):
            raise ValueError("invalid component-history checkpoint")
        base = dict(saved)
        meta = base.pop('component_history', None)
        if (not isinstance(meta, dict) or set(meta) != {'schema', 'mode', 'placement', 'pending', 'signal'}
                or meta['schema'] != 1 or meta['mode'] != self.generator_history_mode
                or meta['placement'] != 'before_normalization' or type(meta['pending']) is not bool):
            raise ValueError("invalid component-history checkpoint settings")
        self.gradient_signal.validate(meta['signal'], self.param_groups)
        if meta['pending'] and base.get('constraint_geometry', {}).get('pending') is None:
            raise ValueError("component-history checkpoint is missing protected normals")
        super().validate_state_dict(base)
        if any(group['role'] == 'generator' and group['params'] for group in self.param_groups):
            expected_count = base['constraint_geometry']['stats']['steps'] + int(meta['pending'])
            packet = meta['signal']['packets'].get('generator')
            if (packet is None and expected_count != 0) or (packet is not None and packet['count'] != expected_count):
                raise ValueError("component-history clock differs from optimizer updates")

    def load_state_dict(self, saved):
        # The base loader dynamically calls the complete validator first.
        super().load_state_dict(saved)
        meta = saved['component_history']
        self.gradient_signal.load(meta['signal'], self.param_groups)
        self._history_pending = deepcopy(meta['pending'])
        self._history_failed = False

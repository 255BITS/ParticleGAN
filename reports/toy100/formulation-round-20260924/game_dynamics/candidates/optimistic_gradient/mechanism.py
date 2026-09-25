"""Predict gradients before Adam; adapt the second moment to the prediction."""
import torch


class Correction:
    def __init__(self):
        self.calls = 0
        self.corrected_tensors = 0

    def step(self, opt, original_step, *args, **kwargs):
        entries = []
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                raw = p.grad
                # Keep custom memory outside state until Adam initializes it.
                previous = opt.state.get(p, {}).get('game_previous_gradient')
                if previous is not None:
                    p.grad = 2 * raw - previous
                    self.corrected_tensors += 1
                entries.append((p, raw))
        try:
            result = original_step(opt, *args, **kwargs)
            for p, raw in entries:
                assert p.device.type == 'cuda' and raw.device == p.device
                opt.state[p]['game_previous_gradient'] = raw.detach().clone()
        finally:
            for p, raw in entries:
                p.grad = raw
        self.calls += 1
        return result

    def receipt(self):
        return dict(mechanism='optimistic_gradient', calls=self.calls,
                    corrected_tensors=self.corrected_tensors,
                    extra_forward_evaluations=0, extra_backward_evaluations=0,
                    extra_optimizer_updates=0, memory_device='cuda')

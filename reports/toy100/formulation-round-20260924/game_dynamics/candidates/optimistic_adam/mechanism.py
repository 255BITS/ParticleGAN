"""Optimism on Adam directions; no extra gradients, losses or random draws."""
import torch


class Correction:
    def __init__(self):
        self.calls = 0
        self.corrected_tensors = 0

    def step(self, opt, original_step, *args, **kwargs):
        entries = [(p, p.detach().clone(), float(g['lr']))
                   for g in opt.param_groups for p in g['params'] if p.grad is not None]
        result = original_step(opt, *args, **kwargs)
        with torch.no_grad():
            for p, before, lr in entries:
                assert lr > 0 and p.device.type == 'cuda'
                direction = (p - before) / lr
                previous = opt.state[p].get('game_previous_direction')
                if previous is not None:
                    p.add_(direction - previous, alpha=lr)
                    self.corrected_tensors += 1
                opt.state[p]['game_previous_direction'] = direction
        self.calls += 1
        return result

    def receipt(self):
        return dict(mechanism='optimistic_adam', calls=self.calls,
                    corrected_tensors=self.corrected_tensors,
                    extra_forward_evaluations=0, extra_backward_evaluations=0,
                    extra_optimizer_updates=0, memory_device='cuda')

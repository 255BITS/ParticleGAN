"""Particle-only Adam with a second moment shared over table rows."""
import json
import torch


class ParticleUpdate:
    def __init__(self):
        self.tables = {}
        self.counts = {}
        self.trace = []

    def register(self, prior):
        self.tables[id(prior.z)] = prior.z

    @torch.no_grad()
    def step(self, optimizer, original_step, *args, **kwargs):
        pending = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) in self.tables and p.grad is not None:
                    assert p.ndim == 2 and p.device.type == 'cuda'
                    assert group['weight_decay'] == 0 and not group['amsgrad']
                    pending.append((p, p.grad, group))
                    p.grad = None
        result = original_step(optimizer, *args, **kwargs)
        for p, gradient, group in pending:
            p.grad = gradient
            state = optimizer.state[p]
            if not state:
                state['step'] = torch.zeros((), device=p.device)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p[:1])
            beta1, beta2 = group['betas']
            state['step'].add_(1)
            n = self.counts.get(id(p), 0) + 1
            self.counts[id(p)] = n
            state['exp_avg'].lerp_(gradient, 1 - beta1)
            pooled = gradient.square().mean(dim=0, keepdim=True)
            state['exp_avg_sq'].lerp_(pooled, 1 - beta2)
            denom = (state['exp_avg_sq'] / (1 - beta2 ** n)).sqrt().add_(group['eps'])
            displacement = -group['lr'] / (1 - beta1 ** n) * state['exp_avg'] / denom
            p.add_(displacement)
            if n == 1 or n % 200 == 0:
                row = dict(step=n, lr=group['lr'], shape=list(p.shape),
                           active_fraction=float((gradient.norm(dim=1) > 0).float().mean()),
                           gradient_rms=float(gradient.square().mean().sqrt()),
                           displacement_rms=float(displacement.square().mean().sqrt()),
                           displacement_max=float(displacement.norm(dim=1).max()),
                           device=str(p.device), moment_device=str(state['exp_avg_sq'].device))
                self.trace.append(row)
                print(json.dumps(dict(event='particle_update', **row)), flush=True)
        return result

    def receipt(self):
        return dict(updates=list(self.counts.values()), trace=self.trace,
                    extra_forward=0, extra_backward=0, extra_optimizer_updates=0)

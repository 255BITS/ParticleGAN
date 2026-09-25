"""Direction-preserving particle Adam on each particle's sampled-visit clock."""
import json
import torch


class ParticleUpdate:
    def __init__(self):
        self.tables = {}
        self.counts = {}
        self.trace = []
        self.hook_calls = 0

    def register(self, prior):
        self.tables[id(prior.z)] = prior.z

    def sampled(self, prior, latent, indices):
        if not latent.requires_grad:
            return
        def normalize(gradient):
            counts = torch.bincount(indices, minlength=prior.num_particles)
            weights = (len(indices) / prior.num_particles) / counts[indices].to(gradient.dtype)
            self.hook_calls += 1
            return gradient * weights[:, None]
        latent.register_hook(normalize)

    @torch.no_grad()
    def step(self, optimizer, original_step, *args, **kwargs):
        pending = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) in self.tables and p.grad is not None:
                    assert p.device.type == 'cuda' and group['weight_decay'] == 0
                    pending.append((p, p.grad, group))
                    p.grad = None
        result = original_step(optimizer, *args, **kwargs)
        for p, gradient, group in pending:
            p.grad = gradient
            active = gradient.square().sum(dim=1, keepdim=True) > 0
            exposure = active.to(p.dtype).mean()
            state = optimizer.state[p]
            if not state:
                state['step'] = torch.zeros((), device=p.device)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p[:, :1])
                state['visits'] = torch.zeros_like(p[:, :1])
                state['exposure_ema'] = exposure.clone()
            beta1, beta2 = group['betas']
            state['step'].add_(1)
            state['visits'].add_(active)
            state['exposure_ema'].lerp_(exposure, 1 - beta2)
            state['exp_avg'].copy_(torch.where(active,
                state['exp_avg'] * beta1 + gradient * (1 - beta1), state['exp_avg']))
            state['exp_avg_sq'].copy_(torch.where(active,
                state['exp_avg_sq'] * beta2 + gradient.square().mean(dim=1, keepdim=True) * (1 - beta2),
                state['exp_avg_sq']))
            first_bias = (1 - beta1 ** state['visits']).clamp_min(group['eps'])
            second_bias = (1 - beta2 ** state['visits']).clamp_min(group['eps'])
            denom = (state['exp_avg_sq'] / second_bias).sqrt().add_(group['eps'])
            direction = state['exp_avg'] / first_bias / denom
            # Sparse dense-clock Adam has roughly 1/sqrt(q) conditional scale.
            # Preserve that energy scale while using conditional-visit moments.
            displacement = -group['lr'] * direction / state['exposure_ema'].clamp_min(group['eps']).sqrt()
            displacement = torch.where(active, displacement, 0)
            p.add_(displacement)
            n = self.counts.get(id(p), 0) + 1
            self.counts[id(p)] = n
            if n == 1 or n % 200 == 0:
                row = dict(step=n, lr=group['lr'], shape=list(p.shape),
                           exposure=float(exposure), exposure_ema=float(state['exposure_ema']),
                           min_visits=float(state['visits'].min()), max_visits=float(state['visits'].max()),
                           displacement_rms=float(displacement.square().mean().sqrt()),
                           displacement_max=float(displacement.norm(dim=1).max()),
                           device=str(p.device), moment_device=str(state['exp_avg_sq'].device))
                self.trace.append(row)
                print(json.dumps(dict(event='particle_update', **row)), flush=True)
        return result

    def receipt(self):
        return dict(updates=list(self.counts.values()), trace=self.trace,
                    hook_calls=self.hook_calls, extra_forward=0,
                    extra_backward=0, extra_optimizer_updates=0)

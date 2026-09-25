"""Amortize particle travel limits by observed sampling exposure."""
import json
import math
import torch


class ParticleUpdate:
    def __init__(self):
        self.tables = {}
        self.exposures = {}
        self.counts = {}
        self.trace = []
        self.hook_calls = 0

    def register(self, prior):
        self.tables[id(prior.z)] = prior.z

    def sampled(self, prior, latent, indices):
        if not latent.requires_grad:
            return
        def observe(gradient):
            counts = torch.bincount(indices, minlength=prior.num_particles)
            self.exposures[id(prior.z)] = (counts > 0).to(gradient.dtype).mean()
            self.hook_calls += 1
            return gradient
        latent.register_hook(observe)

    @torch.no_grad()
    def step(self, optimizer, original_step, *args, **kwargs):
        pending = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) in self.tables and p.grad is not None:
                    assert id(p) in self.exposures
                    pending.append((p, p.detach().clone(), group))
        result = original_step(optimizer, *args, **kwargs)
        for p, before, group in pending:
            n = self.counts.get(id(p), 0) + 1
            self.counts[id(p)] = n
            state = optimizer.state[p]
            q = self.exposures.pop(id(p))
            if 'exposure_ema' not in state:
                state['exposure_ema'] = q.clone()
            else:
                state['exposure_ema'].lerp_(q, 1 - group['betas'][1])
            # Expected motion budget per global iteration stays lr*sqrt(d),
            # even when only a fraction q of the table participates.
            cap = group['lr'] * math.sqrt(p[0].numel()) / state['exposure_ema']
            displacement = p - before
            norms = displacement.norm(dim=1)
            scale = (cap / norms.clamp_min(torch.finfo(p.dtype).tiny)).clamp(max=1)
            clipped = scale < 1
            p.copy_(torch.where(clipped[:, None], before + displacement * scale[:, None], p))
            if n == 1 or n % 200 == 0:
                row = dict(step=n, lr=group['lr'], shape=list(p.shape),
                           exposure=float(q), exposure_ema=float(state['exposure_ema']),
                           displacement_cap=float(cap), proposed_max=float(norms.max()),
                           clipped_fraction=float(clipped.float().mean()),
                           device=str(p.device), moment_device=str(state['exp_avg_sq'].device))
                self.trace.append(row)
                print(json.dumps(dict(event='particle_update', **row)), flush=True)
        return result

    def receipt(self):
        return dict(updates=list(self.counts.values()), trace=self.trace,
                    hook_calls=self.hook_calls, extra_forward=0,
                    extra_backward=0, extra_optimizer_updates=0)

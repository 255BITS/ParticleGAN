"""Particle-only inverse-neighborhood-density mobility with ordinary Adam."""
import json
import math
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
                if id(p) not in self.tables or p.grad is None:
                    continue
                indices = (p.grad.square().sum(dim=1) > 0).nonzero().flatten()
                before = p.detach().clone()
                positions = before[indices]
                bandwidth = p.new_zeros(())
                if len(indices) >= 2:
                    distance2 = torch.cdist(positions, positions).square()
                    # The median off-diagonal distance and log(n+1) define
                    # a scale from the current active latent cloud only.
                    upper = torch.triu_indices(len(indices), len(indices), offset=1, device=p.device)
                    bandwidth = (distance2[upper[0], upper[1]].median() / math.log(len(indices) + 1)).clamp_min(group['eps'])
                    density = torch.exp(-distance2 / bandwidth).sum(dim=1)
                    mobility = density.rsqrt()
                    mobility = mobility / mobility.square().mean().sqrt()
                else:
                    mobility = torch.ones(len(indices), dtype=p.dtype, device=p.device)
                pending.append((p, before, indices, mobility, bandwidth, group))
        result = original_step(optimizer, *args, **kwargs)
        for p, before, indices, mobility, bandwidth, group in pending:
            displacement = p - before
            p.index_copy_(0, indices, before[indices] + displacement[indices] * mobility[:, None])
            n = self.counts.get(id(p), 0) + 1
            self.counts[id(p)] = n
            if n == 1 or n % 200 == 0:
                row = dict(step=n, lr=group['lr'], shape=list(p.shape), active_rows=len(indices),
                           bandwidth=float(bandwidth), mobility_min=float(mobility.min()) if len(indices) else 0,
                           mobility_max=float(mobility.max()) if len(indices) else 0,
                           displacement_rms=float((p-before).square().mean().sqrt()),
                           displacement_max=float((p-before).norm(dim=1).max()),
                           device=str(p.device), moment_device=str(optimizer.state[p]['exp_avg_sq'].device))
                self.trace.append(row)
                print(json.dumps(dict(event='particle_update', **row)), flush=True)
        return result

    def receipt(self):
        return dict(updates=list(self.counts.values()), trace=self.trace,
                    hook_calls=self.hook_calls, extra_forward=0,
                    extra_backward=0, extra_optimizer_updates=0)

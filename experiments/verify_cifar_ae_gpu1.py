#!/usr/bin/env python
"""Targeted gradient/EMA checks for the isolated GPU-1 performance harness."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.profile_cifar_ae_gpu1 import torch, DirectDiscriminator, DirectGenerator, build_encoder, GANLoss, GradientPenalty

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(24002)
torch.cuda.manual_seed_all(24002)
# Tight diagnostic tolerances; benchmarks retain production TF32 settings.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
d = DirectDiscriminator(32).cuda()
real = torch.randn(64, 3, 32, 32, device='cuda').tanh()
fake = torch.randn_like(real).tanh()
adv = GANLoss()
result = {}
for cap in (1., 0.):
    # cap=0 explicitly exercises nonzero penalty gradients on the initial D.
    pen = GradientPenalty(lazy_k=8, kappa=cap)
    reference = None
    for variant in ('baseline', 'baseline_repeat', 'reuse_bcap', 'batched_d', 'pruned_d', 'reuse_pruned_d'):
        d.zero_grad(set_to_none=True)
        if variant.startswith('baseline') or variant == 'pruned_d':
            dp = pen(d, real, fake, 8)
            r, f = d(real), d(fake)
        else:
            xr, xf = real.detach().clone().requires_grad_(True), fake.detach().clone().requires_grad_(True)
            if variant == 'batched_d':
                joined = torch.cat([xr, xf])
                r, f = d(joined).chunk(2)
                grads = torch.autograd.grad(r.sum() + f.sum(), joined, create_graph=True)[0].chunk(2)
            else:
                r, f = d(xr), d(xf)
                grads = [torch.autograd.grad(y.sum(), x, create_graph=True)[0] for y, x in ((r, xr), (f, xf))]
            dp = 4 * sum(((v.square().flatten(1).sum(1) + 1e-12).sqrt() - cap).relu().square().mean() for v in grads)
        loss = adv.d_loss(r, f) + dp
        if variant in ('pruned_d', 'reuse_pruned_d'):
            loss.backward(inputs=[p for p in d.parameters() if p.requires_grad])
        else:
            loss.backward()
        grad = torch.cat([p.grad.flatten() for p in d.parameters() if p.grad is not None])
        if reference is None:
            reference = grad.clone()
        row = {'loss': float(loss.detach()), 'penalty': float(dp.detach()), 'gradient_relative_l2': float((grad-reference).norm()/reference.norm()),
               'gradient_max_abs': float((grad-reference).abs().max()), 'finite': bool(torch.isfinite(grad).all())}
        result[f'cap={cap}/{variant}'] = row
        print('GRAD_CHECK', variant, cap, json.dumps(row), flush=True)
        assert row['finite'] and row['gradient_relative_l2'] < 1e-4

sources = [DirectGenerator(64, 32).cuda(), build_encoder({'encoder_backbone':'scratch','z_dim':64,'width':32}).cuda()]
a = [copy.deepcopy(m).requires_grad_(False) for m in sources]
b = [copy.deepcopy(m).requires_grad_(False) for m in sources]
with torch.no_grad():
    for m in sources:
        for p in m.parameters():
            p.add_(.01 * torch.randn_like(p))
    for target, source in zip(a, sources):
        for p, q in zip(target.parameters(), source.parameters()):
            p.lerp_(q, .005)
    torch._foreach_lerp_([p for m in b for p in m.parameters()], [p for m in sources for p in m.parameters()], .005)
    maximum = max(float((p-q).abs().max()) for x,y in zip(a,b) for p,q in zip(x.parameters(),y.parameters()))
result['foreach_ema'] = {'max_abs': maximum, 'bitwise_equal': maximum == 0}
assert maximum == 0
out = Path('runs/cifar_particle_ae/performance_gpu1/gradient_checks.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print('GRADIENT_CHECKS_COMPLETE', out, flush=True)

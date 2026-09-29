"""Equality of the fast KD-tree replica, the torch replica and the SHIPPED `_isolated` (flags and p-values) on a native-like table with strays."""
import numpy as np, torch, time
from simlib import *
from kdlib import kd_scores, bh
g = torch.Generator().manual_seed(1)
co = torch.arange(10.) - 4.5; ce = torch.stack(torch.meshgrid(co, co, indexing='ij'), -1).reshape(-1, 2)
def draw(n, sd=.03):
    c = torch.randint(0, 100, (n,), generator=g); return ce[c] + sd * torch.randn(n, 2, generator=g)
R = draw(20000)
for n_stray in (0, 30, 60, 300):
    q = draw(20000 - n_stray); ang = torch.rand(n_stray, generator=g) * 6.28; r = .15 + .2 * torch.rand(n_stray, generator=g)
    cen = ce[torch.randint(0, 100, (n_stray,), generator=g)]
    q = torch.cat([q, cen + torch.stack([r * ang.cos(), r * ang.sin()], 1)])
    t = time.time(); f_ship = shipped_flags(q, R); t1 = time.time() - t
    p_t, s_t = isolation_scores(q.double(), R.double(), 10)
    t = time.time(); p_k, s_k, *_ = kd_scores(q.numpy(), R.numpy(), 10); t2 = time.time() - t
    f_kd = torch.from_numpy(bh(p_k))
    print(f'strays {n_stray:4d}: shipped flagged {int(f_ship.sum()):4d} ({t1:.2f}s) | torch replica {int(bh_flags(p_t).sum()):4d} | kd replica {int(f_kd.sum()):4d} ({t2:.2f}s) | flags equal shipped/kd: {bool(torch.equal(f_ship, f_kd))} '
          f'| max |p_torch - p_kd| {float((p_t - torch.from_numpy(p_k)).abs().max()):.2e} | rows with p differing {int(((p_t - torch.from_numpy(p_k)).abs() > 1e-12).sum())}')

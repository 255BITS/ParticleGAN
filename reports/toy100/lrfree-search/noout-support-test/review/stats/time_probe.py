import time, torch, numpy as np
from simlib import *
g = torch.Generator().manual_seed(0)
co = torch.arange(10.) - 4.5; ce = torch.stack(torch.meshgrid(co, co, indexing='ij'), -1).reshape(-1, 2)
def draw(n, g, sd=.03):
    c = torch.randint(0, 100, (n,), generator=g); return ce[c] + sd * torch.randn(n, 2, generator=g), c
R, _ = draw(20000, g); q, _ = draw(20000, g)
for dim in (2, 128):
    if dim == 128:
        f = relu_map(2, 128, depth=2); Rf, qf = f(R * 3), f(q * 3)
    else:
        Rf, qf = R, q
    t = time.time(); fl = shipped_flags(qf, Rf); t1 = time.time() - t
    t = time.time(); p, s = isolation_scores(qf.double(), Rf.double(), 10); t2 = time.time() - t
    print(dim, 'shipped %.2fs flags %d | replica %.2fs equal %s' % (t1, int(fl.sum()), t2, bool(torch.equal(bh_flags(p), fl))))

"""Precision limit of the float32 shortlist in _knn (shared by the ordinary birth-death and the new support test): fraction of queries whose k-th neighbour distance is wrong,
for 4 tight clusters (sd sigma, in d dimensions, 1000 points each) whose centres are S apart (after centring the features are of size ~S)."""
import sys, torch
torch.set_num_threads(1)
sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17')
from particlegan.birth_death import _knn
def run(d, ratio, k=10, n=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = 1.0; sigma = S / ratio
    centres = torch.randn(4, d, generator=g, dtype=torch.float64); centres = centres / centres.norm(dim=1, keepdim=True) * S
    x = (centres.repeat_interleave(n, 0) + sigma * torch.randn(4 * n, d, generator=g, dtype=torch.float64))
    x = x - x.mean(0, keepdim=True)
    exact = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist'); exact.fill_diagonal_(float('inf'))
    truth = exact.topk(k, dim=1, largest=False).values[:, -1]
    got = _knn(x, x, k, exclude=torch.arange(len(x)), shortlist_dtype=torch.float32)[0][:, -1]
    rel = (got - truth) / truth
    return float((rel.abs() > 1e-9).double().mean()), float(rel.max()), float(truth.median() / sigma)
print('d  S/sigma   frac wrong k-th distance   max rel error   median r_k / sigma')
for d in (2, 8, 32):
    for ratio in (1e1, 1e2, 1e3, 3e3, 1e4):
        f, mx, rk = run(d, ratio)
        print(f'{d:2d} {ratio:8.0f}   {f:8.4f}                   {mx:9.3f}       {rk:6.2f}')

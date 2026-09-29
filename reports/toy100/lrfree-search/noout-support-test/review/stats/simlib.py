"""Shared helpers for the E17 statistical review (CPU only, <= 8 threads).
Loads the REAL `_isolated` of pkg-E17 (importlib, no package init) so every simulation calls the shipped code, and provides a p-value
replica (`isolation_scores`) that is asserted equal to the shipped flags in `check_replica`."""
import importlib.util, math, sys, time, os
import numpy as np, torch
torch.set_num_threads(8)
PKG = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17/particlegan/birth_death.py'
_spec = importlib.util.spec_from_file_location('bd17_isolated', PKG)
bd17 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bd17)
_knn = bd17._knn
Q = 0.05
class _Self:                       # stands in for the ParticleBirthDeath instance: `_isolated` only reads self.Q
    Q = 0.05
def K_of(n, q=Q):                  # the trainer's k
    return min(math.ceil((math.log2(n / q) + 1) / 2), n - 2)
def shipped_flags(q, R, k=None, fast=torch.float32):
    k = k or K_of(len(q))
    return bd17.ParticleBirthDeath._isolated(_Self(), q.double(), R.double(), k, fast)

@torch.no_grad()
def isolation_scores(q, R, k, fast=torch.float32, return_all=False):
    """Replica of `_isolated` returning (p, score, null_scores, rho_at_neighbours) instead of the flag mask; identical arithmetic."""
    R1, R2 = R[0::2], R[1::2]
    rho = _knn(R1, R1, k, exclude=torch.arange(len(R1), device=R.device), shortlist_dtype=fast)[0][:, -1]
    positive = rho[rho > 0]
    floor = float(positive.median()) * 1e-3 if len(positive) else 1.
    def score(u):
        d, ix = _knn(u, R1, k, shortlist_dtype=fast)
        b = rho[ix].sort(dim=1).values[:, (k - 1) // 2].clamp_min(floor)
        return d[:, -1] / b, d[:, -1], b
    null, _, _ = score(R2)
    null = null.sort().values
    s, a, b = score(q)
    p = (1. + (len(null) - torch.searchsorted(null, s)).double()) / (1. + len(null))
    if return_all:
        return p, s, null, a, b
    return p, s

def bh_flags(p, level=Q):
    m = len(p); ps = p.sort().values
    passed = (ps <= torch.arange(1, m + 1, dtype=p.dtype) * level / m).nonzero()
    if not len(passed):
        return torch.zeros(m, dtype=torch.bool)
    return p <= ps[int(passed[-1])]

def check_replica(q, R, k=None):
    k = k or K_of(len(q))
    p, _ = isolation_scores(q.double(), R.double(), k)
    a, b = bh_flags(p), shipped_flags(q, R, k)
    return bool(torch.equal(a, b)), int(a.sum()), int(b.sum())

def relu_map(d_in, h, seed=0, bias_sd=0.5, depth=1):
    """Random ReLU feature map (stand-in for critic features): x -> relu(W x + b) (depth layers), fixed by seed. Inputs are expected O(1)-scaled per coordinate."""
    g = torch.Generator().manual_seed(1000 + seed)
    Ws, bs, dim = [], [], d_in
    for _ in range(depth):
        Ws.append(torch.randn(dim, h, generator=g) / math.sqrt(dim)); bs.append(bias_sd * torch.randn(h, generator=g)); dim = h
    def f(x):
        y = x
        for W, b in zip(Ws, bs):
            y = torch.relu(y @ W + b)
        return y
    return f
def bh_np(p, level=Q):
    return bh_flags(torch.as_tensor(p), level).numpy()

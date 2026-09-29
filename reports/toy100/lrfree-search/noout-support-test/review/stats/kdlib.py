"""Fast exact replica of `_isolated` for low-dimensional feature spaces (scipy cKDTree instead of the brute-force matmul kNN); same arithmetic:
rho = leave-one-out k-th NN radius in R1, score = a(u) / lower-median_k rho(nn_j(u)), p = split-conformal against R2. Verified equal to the shipped flags in check_kd.py."""
import numpy as np
from scipy.spatial import cKDTree
WORKERS = 2
def kd_scores(q, R, k, workers=None, floor_mult=1e-3):
    workers = workers or WORKERS
    R = np.asarray(R, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    R1, R2 = R[0::2], R[1::2]
    tree = cKDTree(R1)
    d, _ = tree.query(R1, k=k + 1, workers=workers)
    rho = d[:, k]                                   # (k+1)-th smallest incl. the point itself at distance 0 == k-th neighbour excluding self
    positive = rho[rho > 0]
    floor = float(np.median(positive)) * floor_mult if len(positive) else 1.
    def score(u):
        dd, ix = tree.query(u, k=k, workers=workers)
        b = np.sort(rho[ix], axis=1)[:, (k - 1) // 2]
        return dd[:, -1] / np.maximum(b, floor), dd[:, -1], b
    null = np.sort(score(R2)[0])
    s, a, b = score(q)
    p = (1. + (len(null) - np.searchsorted(null, s, side='left'))) / (1. + len(null))
    return p, s, null, a, b
def bh(p, level=0.05):
    m = len(p); o = np.argsort(p, kind='stable'); ps = p[o]
    ok = np.nonzero(ps <= level * np.arange(1, m + 1) / m)[0]
    if not len(ok):
        return np.zeros(m, bool)
    return p <= ps[ok.max()]

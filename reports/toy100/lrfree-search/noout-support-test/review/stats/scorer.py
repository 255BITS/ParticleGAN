"""Reference-set scorer: build once per reservoir R (split R1/R2, rho, null scores), then score any number of query sets.
backend 'kd'    : scipy cKDTree (exact; low dimension; used for 2-D families) - identical to the shipped code in check_kd.py
backend 'torch' : the shipped `_knn` (float32 matmul shortlist + exact float64 recompute) - the shipped arithmetic, used in feature spaces of dim > 8."""
import numpy as np, torch
from kdlib import bh
from simlib import bd17, _knn

class Ref:
    def __init__(self, R, k, backend='kd', workers=2, floor_mult=1e-3, fast=torch.float32):
        self.k, self.backend, self.workers, self.fast = k, backend, workers, fast
        if backend == 'kd':
            from scipy.spatial import cKDTree
            R = np.asarray(R, dtype=np.float64); R1, R2 = R[0::2], R[1::2]
            self.tree = cKDTree(R1)
            d, _ = self.tree.query(R1, k=k + 1, workers=workers); self.rho = d[:, k]
        else:
            R = torch.as_tensor(R).double(); R1, R2 = R[0::2], R[1::2]
            self.R1 = R1
            self.rho = _knn(R1, R1, k, exclude=torch.arange(len(R1)), shortlist_dtype=fast)[0][:, -1].numpy()
        positive = self.rho[self.rho > 0]
        self.floor = float(np.median(positive)) * floor_mult if len(positive) else 1.
        self.n_zero_rho = int((self.rho == 0).sum())
        self.R2 = R2
        self.null = np.sort(self.score(R2)[0])
    def score(self, u):
        k = self.k
        if self.backend == 'kd':
            dd, ix = self.tree.query(np.asarray(u, dtype=np.float64), k=k, workers=self.workers); dk = dd[:, -1]
        else:
            d, ix = _knn(torch.as_tensor(u).double(), self.R1, k, shortlist_dtype=self.fast); dk = d[:, -1].numpy(); ix = ix.numpy()
        b = np.sort(self.rho[ix], axis=1)[:, (k - 1) // 2]
        return dk / np.maximum(b, self.floor), dk, b
    def p(self, u, return_all=False):
        s, a, b = self.score(u)
        p = (1. + (len(self.null) - np.searchsorted(self.null, s, side='left'))) / (1. + len(self.null))
        return (p, s, a, b) if return_all else p

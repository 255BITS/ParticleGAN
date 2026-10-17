"""Deterministic draws for ``particlegan.init``: hashed QR matrices, R2 points, bias patterns.

Everything is computed in float64 on the CPU from integer keys; no RNG state is read or consumed.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


def _hash_u01(key, n):
    with np.errstate(over="ignore"):
        x = (np.arange(n, dtype=np.uint64) + np.uint64(key)) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)


def key(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") >> 1


def semi_orthogonal(key, rows, cols):
    """[rows, cols] float64 with orthonormal columns (rows>=cols) or orthonormal rows (rows<cols)."""
    flip = rows < cols
    r, c = (cols, rows) if flip else (rows, cols)
    source = torch.special.ndtri(torch.from_numpy(_hash_u01(key, r * c)).reshape(r, c))
    q, rr = torch.linalg.qr(source)
    d = torch.sign(torch.diagonal(rr))
    q = q * torch.where(d == 0, torch.ones_like(d), d)
    return q.T.contiguous() if flip else q


def r2_points(n, d):
    """Roberts R2 low-discrepancy sequence in [0,1)^d (generalised golden ratio)."""
    phi = 2.0
    for _ in range(64):
        phi = (1 + phi) ** (1.0 / (d + 1))
    alpha = np.array([(1 / phi) ** (j + 1) for j in range(d)])
    return torch.from_numpy(np.mod(0.5 + np.arange(1, n + 1)[:, None] * alpha[None, :], 1.0))


def pattern(key, shape, mean, std):
    """Zero-mean, unit-variance hashed pattern, then scaled to ``mean``/``std``."""
    n = math.prod(shape)
    u = torch.from_numpy(_hash_u01(key, n)) * 2 - 1
    if n > 1:
        u = (u - u.mean()) / u.std(unbiased=False)
    return (mean + std * u).reshape(tuple(shape))

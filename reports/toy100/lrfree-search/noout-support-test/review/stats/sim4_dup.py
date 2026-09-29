"""Q4b: exact duplicates among the reals. The reservoir (20,000 rows) is drawn WITH REPLACEMENT from a finite dataset of M unique points of the 100-mode law
(a finite dataset much smaller than the reservoir, or an epoch-repeating stream). Fakes: (a) fresh draws of the continuous law (a good, non-memorising generator),
(b) exact copies of dataset points (a memorising generator). Report: fraction of reference points with rho = 0 (duplicates >= k), flagged fraction, guard outcome (acts
only if 0 < flagged <= .05 N), legit rows moved.
usage: python sim4_dup.py EVALS"""
import sys, numpy as np
import fam
from scorer import Ref
from kdlib import bh
E = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N, K = 20000, 10
F = fam.grid100(); g = np.random.default_rng(77)
print(f'{"M unique":>9s} | R1 copies/point | rho=0 share | fakes | flagged (mean; min-max) | guard acts | share of table moved | P(p<=1e-3)/1e-3 | P(p<=1e-2)/1e-2')
for M in (300, 1000, 2000, 3000, 4000, 6000, 10000, 15000, 20000, 200000):
    rows = {'fresh draws': [], 'memorised copies': []}
    for ev in range(E):
        data = F.real(M, g)[0]                                     # the finite dataset
        R = data[g.integers(0, M, N)]                              # reservoir: iid draws with replacement from it
        ref = Ref(R, K, 'kd', workers=2)
        for kind in rows:
            q = F.real(N, g)[0] if kind == 'fresh draws' else data[g.integers(0, M, N)]
            p = ref.p(q); fl = bh(p); n = int(fl.sum()); acts = 0 < n <= .05 * N
            rows[kind].append((n, acts, n if acts else 0, ref.n_zero_rho / (N // 2), float((p <= 1e-3).mean()) / 1e-3, float((p <= 1e-2).mean()) / 1e-2))
    for kind, v in rows.items():
        v = np.array(v, dtype=float)
        print(f'{M:9d} | {N / 2 / M:15.2f} | {v[:, 3].mean():11.3f} | {kind:16s} | {v[:, 0].mean():8.1f} ({v[:, 0].min():.0f}-{v[:, 0].max():.0f}) | {v[:, 1].mean():.2f} | {v[:, 2].mean() / N:.4f} | {v[:, 4].mean():7.2f} | {v[:, 5].mean():7.2f}', flush=True)

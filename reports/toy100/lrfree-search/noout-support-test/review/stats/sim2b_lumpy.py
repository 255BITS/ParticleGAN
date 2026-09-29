"""Q2b: dependence stress. Exact null with perfect positive dependence: the table consists of N/m groups, each group = ONE fresh draw of the real law repeated m times (m identical rows).
p-values are comonotone within a group. BH (PRDS) still bounds the FDR = P(any flag) here (all flags are false) by q, but a flagged group moves m rows at once.
usage: python sim2b_lumpy.py EVALS"""
import sys, numpy as np
import fam
from scorer import Ref
from kdlib import bh
E = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N, K = 20000, 10
F = fam.grid100(); g = np.random.default_rng(5)
print(f'{"rows per group m":>17s} | groups | P(any flag) (= FDR) | mean flagged rows/eval | flagged rows when flagged (mean, max) | 95% upper bound on P(any flag)')
from scipy.stats import beta
for m in (1, 5, 20, 50, 100, 200, 500, 1000):
    G = N // m; nfl = []
    for ev in range(E):
        R = F.real(N, g)[0]; ref = Ref(R, K, 'kd', workers=2)
        base = F.real(G, g)[0]; q = np.repeat(base, m, axis=0)
        nfl.append(int(bh(ref.p(q)).sum()))
    nfl = np.array(nfl); k = int((nfl > 0).sum()); up = beta.ppf(.975, k + 1, E - k) if k < E else 1.
    pos = nfl[nfl > 0]
    print(f'{m:17d} | {G:6d} | {k / E:19.4f} | {nfl.mean():22.2f} | {(pos.mean() if len(pos) else 0):8.1f}, {nfl.max():5d} | {up:.4f}', flush=True)

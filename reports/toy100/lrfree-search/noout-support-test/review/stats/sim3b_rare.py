"""Q3b / Q6: are very rare components erased when the BH threshold is lifted by a stray population? Mixture: 13 components (unequal masses .32-.002, sd .03-.15) + 3 extra
rare components (masses .0002 / .0005 / .001 = 4 / 10 / 20 rows of 20,000, sd .03) at random positions. Table rows are fresh draws of the same law ('iid') or clean atoms of the
generator with sigma_out .029 ('clean'); m1 far strays (6-12 sigma from a random comp centre) replace bulk rows. Per component: mean number of its rows flagged per evaluation,
share of its rows flagged.  usage: python sim3b_rare.py EVALS"""
import sys, numpy as np
import fam
from scorer import Ref
from kdlib import bh
E = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N, K = 20000, 10
U = fam.unequal_mixed(); g = np.random.default_rng(3)
extra_c = []
while len(extra_c) < 3:
    c = g.uniform(-4, 4, 2)
    if all(np.linalg.norm(c - o) > 1.2 for o in list(U.c) + extra_c): extra_c.append(c)
C = np.vstack([U.c, extra_c]); Sd = np.concatenate([U.s, [.03, .03, .03]]); M = np.concatenate([U.m * (1 - .0017), [.0002, .0005, .001]]); M = M / M.sum()
F = fam.Mixture(C, Sd, M, name='13 comps + 3 rare (4/10/20 rows)')
names = [f'c{j} m={F.m[j]:.4f}' for j in range(F.M)]
print(F.name, '| expected rows per component in the table:', {j: round(F.m[j] * N, 1) for j in range(13, 16)})
for bulk in ('iid', 'clean'):
    for m1 in (0, 60, 300, 1000):
        acc = np.zeros(F.M); tot = np.zeros(F.M); nfl = []
        for ev in range(E):
            R = F.real(N, g)[0]; ref = Ref(R, K, 'kd', workers=2)
            lab = g.choice(F.M, N - m1, p=F.m)
            rows = (F.c[lab] + F.s[lab][:, None] * g.standard_normal((N - m1, 2))) if bulk == 'iid' else (F.c[lab] + np.sqrt(np.maximum(F.s[lab] ** 2 - .029 ** 2, 0.))[:, None] * g.standard_normal((N - m1, 2)))
            st = C[g.integers(0, 13, m1)] + g.uniform(6, 12, m1)[:, None] * .03 * np.stack([np.cos(t := g.uniform(0, 6.283, m1)), np.sin(t)], 1) if m1 else np.zeros((0, 2))
            fl = bh(ref.p(np.concatenate([rows, st]))); nfl.append(int(fl.sum()))
            for j in range(F.M): acc[j] += fl[:N - m1][lab == j].sum(); tot[j] += (lab == j).sum()
        print(f'bulk={bulk:5s} strays={m1:4d}: flagged/eval {np.mean(nfl):6.1f} | share of a component\'s rows flagged, smallest 6 comps: ' + ' '.join(f'{names[j].split()[1]}:{acc[j] / max(tot[j], 1):.4f}' for j in list(np.argsort(F.m)[:6][::-1])) + f' | largest comp: {acc[np.argmax(F.m)] / tot[np.argmax(F.m)]:.5f}', flush=True)

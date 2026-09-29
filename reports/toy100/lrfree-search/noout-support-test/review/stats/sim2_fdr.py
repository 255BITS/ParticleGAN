"""Q2: dependence, FDR and the 0-40 stray regime of the shipped test (fast exact replica, verified in check_kd.py). Family: 100 modes, sd .03, N=20000, k=10.
bulk 'iid'   : bulk rows are fresh draws of the real law each evaluation (exact conformal null, independent rows)
bulk 'clean' : bulk rows = fixed clean centres of a generator with sigma_out = .029 (table fixed, fresh reservoir every evaluation)
strays: m1 rows replaced by planted strays displaced from a random mode centre by a radius (in sigma = .03 units)
   far  : radius U[6, 12]   mid : radius U[3, 8]   r4 : radius exactly 4   r5 : radius exactly 5
report per m1: mean flagged R, mean false flags V (bulk rows flagged), P(R>0), FDR = mean V/max(R,1), recall of the strays.
usage: python sim2_fdr.py EVALS"""
import sys, json, time, numpy as np
import fam
from scorer import Ref
from kdlib import bh
E = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N, K, SDV = 20000, 10, .03
F = fam.grid100(); g = np.random.default_rng(2024)
centres = F.c
def strays(m1, kind):
    if m1 == 0: return np.zeros((0, 2))
    c = centres[g.integers(0, 100, m1)]; th = g.uniform(0, 2 * np.pi, m1)
    r = {'far': lambda: g.uniform(6, 12, m1), 'mid': lambda: g.uniform(3, 8, m1), 'r4': lambda: np.full(m1, 4.), 'r5': lambda: np.full(m1, 5.)}[kind]() * SDV
    return c + r[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
m1s = [0, 5, 20, 39, 40, 41, 60, 100, 200, 300, 500, 1000, 2000, 4000]
clean_fixed = F.clean(N, g, .029)[0]
out = {}
for bulk, kinds in (('iid', ('far', 'mid', 'r4')), ('clean', ('far', 'r4'))):
    for kind in kinds:
        print(f'== bulk={bulk} strays={kind} | {E} evaluations each (fresh reservoir, fresh strays{", fresh bulk" if bulk == "iid" else ""})')
        print('   m1 | mean R | mean V | P(R>0) | P(V>0) | FDR=E[V/max(R,1)] | recall | max V')
        for m1 in m1s:
            Rs, Vs, Ss = [], [], []; ptail = np.zeros(4); ntot = 0
            for ev in range(E):
                R = F.real(N, g)[0]; ref = Ref(R, K, 'kd', workers=2)
                bulk_rows = F.real(N - m1, g)[0] if bulk == 'iid' else clean_fixed[g.permutation(N)[:N - m1]]
                q = np.concatenate([bulk_rows, strays(m1, kind)])
                pq = ref.p(q); fl = bh(pq); ptail += [(pq[:N - m1] <= t).sum() for t in (1e-4, 1e-3, 1e-2, .1)]; ntot += N - m1; Rs.append(int(fl.sum())); Vs.append(int(fl[:N - m1].sum())); Ss.append(int(fl[N - m1:].sum()))
            Rs, Vs, Ss = map(np.array, (Rs, Vs, Ss))
            fdr = float(np.mean(Vs / np.maximum(Rs, 1))); rec = float(Ss.mean() / m1) if m1 else float('nan')
            tailtxt = ' | bulk P(p<=t)/t at 1e-4,1e-3,1e-2,.1: ' + ' '.join(f'{v / ntot / t:.3f}' for v, t in zip(ptail, (1e-4, 1e-3, 1e-2, .1))) if m1 in (0, 300) else ''
            print(f'{m1:5d} | {Rs.mean():7.2f} | {Vs.mean():6.3f} | {np.mean(Rs > 0):.3f} | {np.mean(Vs > 0):.3f} | {fdr:.4f} | {rec:.3f} | {Vs.max()}' + tailtxt, flush=True)
            out[f'{bulk}/{kind}/{m1}'] = dict(R=Rs.tolist(), V=Vs.tolist(), S=Ss.tolist())
json.dump(out, open('out/sim2_fdr.json', 'w'))

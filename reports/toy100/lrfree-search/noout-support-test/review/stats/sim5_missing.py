"""Q5: non-iid reservoir. The table holds every component (fixed clean rows, sigma_out = .029); the FIFO reservoir holds only some of them (class-sorted or drifting
stream: the last 20,000 real rows cover only part of the data law). Flagged fraction as a function of the missing mass mu (share of the table on components absent from the
reservoir), whether the guard (act only if 0 < flagged <= .05 N) lets the mechanism act, and the number of LEGITIMATE rows re-drawn when it acts.
Families: grid100 (each mode 1%) and unequal (13 components, masses .32 ... .002). usage: python sim5_missing.py EVALS"""
import sys, numpy as np
import fam
from scorer import Ref
from kdlib import bh
E = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N, K = 20000, 10
g = np.random.default_rng(555)
def run(F, absent_sets, tag, bulk_kind):
    print(f'\n== {F.name} | table rows {bulk_kind} | reservoir = renormalised law on the PRESENT components | {E} evaluations each')
    print('   missing mass mu | absent comps | mean flagged | flagged/N | recall of absent rows | legit(present) rows flagged | guard acts (share of evals) | legit rows re-drawn per eval when acting')
    for ab in absent_sets:
        ab = np.array(sorted(ab), dtype=int); present = np.setdiff1d(np.arange(F.M), ab)
        mp = F.m[present] / F.m[present].sum()
        for ev in range(1):
            pass
        rows = []
        lab_t = g.choice(F.M, size=N, p=F.m)
        tab = (F.c[lab_t] + np.sqrt(np.maximum(F.s[lab_t] ** 2 - .029 ** 2, 0.))[:, None] * g.standard_normal((N, F.d))) if bulk_kind == 'clean' else (F.c[lab_t] + F.s[lab_t][:, None] * g.standard_normal((N, F.d)))
        mu = float(np.isin(lab_t, ab).mean()); is_abs = np.isin(lab_t, ab)
        for ev in range(E):
            labr = g.choice(present, size=N, p=mp); R = F.c[labr] + F.s[labr][:, None] * g.standard_normal((N, F.d))
            ref = Ref(R, K, 'kd', workers=2); fl = bh(ref.p(tab)); n = int(fl.sum()); acts = 0 < n <= .05 * N
            rows.append((n, fl[is_abs].sum() / max(is_abs.sum(), 1), fl[~is_abs].sum(), acts, int(fl[~is_abs].sum()) if acts else 0))
        v = np.array(rows, dtype=float)
        print(f'   {mu:14.4f} | {len(ab):12d} | {v[:, 0].mean():12.1f} | {v[:, 0].mean() / N:9.4f} | {v[:, 1].mean():21.3f} | {v[:, 2].mean():27.2f} | {v[:, 3].mean():27.2f} | {v[:, 4].mean():.2f}', flush=True)
G = fam.grid100()
run(G, [[], [0], [0, 1], list(range(3)), list(range(5)), list(range(8)), list(range(10)), list(range(20)), list(range(50)), list(range(70)), list(range(90))], 'g100', 'clean')
U = fam.unequal_mixed()
order = np.argsort(U.m)          # smallest mass first
run(U, [[]] + [list(order[:j]) for j in range(1, 13)], 'unequal', 'clean')

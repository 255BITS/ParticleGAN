"""Q5b: lag / drift and class-sorted FIFO. (a) drift: the reservoir (last 10 batches) is the real law of about 5 batches ago; the table follows the current law. A fraction f of the modes
moved by delta sigma (sigma = .03) in the meantime (the other modes did not). Flagged fraction, guard, and how many rows of the MOVED modes (legitimate rows) are flagged.
(b) class-sorted stream: 10 classes (each 10 of the 100 modes = 10% of the data), each class delivered for `run` consecutive rows in a fixed cycle, FIFO of 20,000: for every
evaluation instant along an epoch, the classes present in the reservoir, missing mass of a table that holds all classes in proportion, flagged fraction, guard.
usage: python sim5b_drift.py EVALS"""
import sys, numpy as np
import fam
from scorer import Ref
from kdlib import bh
E = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N, K, S = 20000, 10, .03
F = fam.grid100(); g = np.random.default_rng(808)
print('(a) drift: fraction f of the 100 modes moved by delta sigma (random direction) between the reservoir and the table; table rows = clean centres of the CURRENT law')
print('   f     delta | flagged/N (mean) | flagged rows in moved modes / moved rows | flagged rows in unmoved modes | guard acts (share) | legit rows re-drawn per acting evaluation')
for f in (.01, .03, .05, .10, 1.0):
    for delta in (2, 4, 6, 8, 16):
        rows = []
        nm = max(1, int(round(f * 100)))
        for ev in range(E):
            moved = g.choice(100, nm, replace=False); th = g.uniform(0, 6.283, nm); shift = np.zeros((100, 2)); shift[moved] = delta * S * np.stack([np.cos(th), np.sin(th)], 1)
            lab = g.integers(0, 100, N); R = F.c[lab] + S * g.standard_normal((N, 2))                          # reservoir: OLD law (unshifted)
            labt = g.integers(0, 100, N); tab = F.c[labt] + shift[labt] + np.sqrt(S ** 2 - .029 ** 2) * g.standard_normal((N, 2))   # table: current law (moved modes shifted), clean rows
            fl = bh(Ref(R, K, 'kd', workers=2).p(tab)); mv = np.isin(labt, moved); n = int(fl.sum()); acts = 0 < n <= .05 * N
            rows.append((n / N, fl[mv].sum() / max(mv.sum(), 1), fl[~mv].sum(), acts, int(fl[mv].sum()) if acts else 0))
        v = np.array(rows, dtype=float)
        print(f'  {f:5.2f} {delta:5d} | {v[:, 0].mean():16.4f} | {v[:, 1].mean():40.3f} | {v[:, 2].mean():29.2f} | {v[:, 3].mean():18.2f} | {v[:, 4].mean():.1f}', flush=True)
print('\n(b) class-sorted stream, 10 classes of 10 modes (10% each), 6000 rows per class per pass (MNIST-like), FIFO of the last 20,000 rows, evaluation every 20,480 rows (10 batches of 2048)')
print('   evaluation | classes in the reservoir (rows) | missing mass of a full table | flagged/N | guard acts | legit rows re-drawn')
stream_len = 60000; pos = 0
def rows_of_stream(start, n):     # class of stream row i = (i // 6000) % 10
    idx = (start + np.arange(n)) % stream_len; return (idx // 6000) % 10
for ev in range(10):
    pos += 20480
    cls = rows_of_stream(pos - 20000, 20000); present = np.unique(cls); cnt = np.bincount(cls, minlength=10)
    modes_of = lambda c: np.arange(c * 10, c * 10 + 10)
    lab = np.array([g.choice(modes_of(c)) for c in cls]); R = F.c[lab] + S * g.standard_normal((N, 2))
    labt = g.integers(0, 100, N); tab = F.c[labt] + np.sqrt(S ** 2 - .029 ** 2) * g.standard_normal((N, 2))
    fl = bh(Ref(R, K, 'kd', workers=2).p(tab)); missing = float((~np.isin(labt // 10, present)).mean()); n = int(fl.sum()); acts = 0 < n <= .05 * N
    print(f'   {ev:10d} | {dict((int(c), int(cnt[c])) for c in present if cnt[c])} | {missing:.3f} | {n / N:.3f} | {acts} | {int(fl[np.isin(labt // 10, present)].sum())}', flush=True)

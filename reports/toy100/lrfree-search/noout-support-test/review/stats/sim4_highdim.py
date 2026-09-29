"""Q4: recall / false flags in high-dimensional feature spaces (random ReLU feature maps of d-dimensional Gaussian mixtures; the intrinsic dimension is d).
Per evaluation: fresh reservoir R (20,000 noisy reals), bulk rows = 19,700 fresh real draws (exact null), 300 planted clean strays = centre + l * u (u a random unit
vector in DATA space, l in multiples of the noise radius r0 = s sqrt(d)), scored at their clean positions. Recall at BH .05 and legit rows flagged.
usage: python sim4_highdim.py D EVALS THREADS"""
import sys, json, time, numpy as np, torch
import fam
from scorer import Ref
from kdlib import bh
d, E, thr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); torch.set_num_threads(thr)
N, K = 20000, 10
F = fam.grid100() if d == 2 else fam.highdim(d)
backend = 'kd' if d == 2 else 'torch'
g = np.random.default_rng(4242 + d)
r0 = float(F.s[0]) * np.sqrt(d)
mult = [0.25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8]
t0 = time.time()
rec = {m1: np.zeros(len(mult)) for m1 in (60, 300)}; leg = {m1: [] for m1 in (60, 300)}; null_cv = []; med_a = np.zeros(len(mult)); med_b = 0.; p_str_med = np.zeros(len(mult))
for ev in range(E):
    R = F.feat(F.real(N, g)[0]); ref = Ref(R, K, backend, workers=thr)
    null_cv.append(float(np.std(ref.null) / np.mean(ref.null)))
    bulk = F.real(N, g)[0]; pb, sb, ab, bb = ref.p(F.feat(bulk), return_all=True)
    med_b += np.median(bb) / E
    for j, mu in enumerate(mult):
        lab = g.integers(0, F.M, 300); u = g.standard_normal((300, F.d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        st = F.c[lab] + mu * r0 * u
        ps, ss, a_s, b_s = ref.p(F.feat(st), return_all=True)
        med_a[j] += np.median(a_s / b_s) / E; p_str_med[j] += np.median(ps) / E
        for m1 in (60, 300):
            p = np.concatenate([pb[:N - m1], ps[:m1]]); fl = bh(p); rec[m1][j] += fl[N - m1:].mean() / E
            if j == 0: leg[m1].append(int(fl[:N - m1].sum()))
    if ev % 10 == 0: print(f'  eval {ev}/{E} {time.time() - t0:.0f}s', flush=True)
print(f'== d={d} ({F.name}) | noise radius r0 = s sqrt(d) = {r0:.4f} | {E} evaluations | null score CV (std/mean of the calibration scores) {np.mean(null_cv):.3f} | median local scale b {med_b:.4g}')
print('stray displacement l / r0      : ' + ' '.join(f'{m:7g}' for m in mult))
print('median score of strays         : ' + ' '.join(f'{v:7.2f}' for v in med_a))
print('median p of strays             : ' + ' '.join(f'{v:7.1e}' for v in p_str_med))
for m1 in (60, 300):
    print(f'recall (m1={m1:3d} strays)         : ' + ' '.join(f'{v:7.3f}' for v in rec[m1]) + f' | legit rows flagged per eval (l=.25 r0): mean {np.mean(leg[m1]):.3f} max {max(leg[m1])}')
json.dump(dict(d=d, mult=mult, rec={str(k): v.tolist() for k, v in rec.items()}, null_cv=float(np.mean(null_cv))), open(f'out/sim4_recall_d{d}.json', 'w'))

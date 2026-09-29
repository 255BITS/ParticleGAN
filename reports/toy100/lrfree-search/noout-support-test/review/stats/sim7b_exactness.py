"""(b) of the gauge question: does normalising the features by a statistic keep the split-conformal p-values exact? Exact null (rows = fresh iid draws of the real law), N = 4000 rows,
reservoir 4000 (R1 = 2000 reference, R2 = 2000 calibration), features = random ReLU(256) image of a 32-D Gaussian mixture (intrinsic dimension 32): n/h = 8, so an in-sample fit is
easy to see. The transformation T is estimated from
   R1   (function of the reference half only: exact by the conformal argument)
   R    (R1 + R2: the calibration half helped to fit T)
   R2   (calibration half only)
   Q    (the query rows themselves, i.e. a statistic of the very points being tested)
for T = per-feature std and T = full whitening (pseudo-inverse of eigenvalues < 1e-10 x max). Pooled P(p<=t)/t over E evaluations (exact null: <= 1).
usage: python sim7b_exactness.py EVALS THREADS"""
import sys, numpy as np, torch
import fam
from scorer import Ref
E, thr = int(sys.argv[1]), int(sys.argv[2]); torch.set_num_threads(thr)
N, K = 4000, 9
F = fam.highdim(32, h=256, M=20, seed=9); g = np.random.default_rng(99)
def T_std(ref):
    s = ref.std(0); s = np.where(s > 1e-9 * s.max(), s, 1.); return lambda X: X / s
def T_white(ref):
    mu = ref.mean(0, keepdims=True); cov = ((ref - mu).T @ (ref - mu)) / (len(ref) - 1); ev, U = np.linalg.eigh(cov); keep = ev > 1e-10 * ev.max(); P = U[:, keep] / np.sqrt(ev[keep]); return lambda X: (X - mu) @ P
ts = (1e-3, 1e-2, 5e-2, .1, .25)
variants = [('raw', None, None)] + [(f'{kind}_{src}', kind, src) for kind in ('std', 'white') for src in ('R1', 'R', 'R2', 'Q')]
cnt = {v[0]: np.zeros(len(ts)) for v in variants}; tot = 0; anyflag = {v[0]: 0 for v in variants}
from kdlib import bh
for ev in range(E):
    R = F.feat(F.real(N, g)[0]); Q = F.feat(F.real(N, g)[0]); R1, R2 = R[0::2], R[1::2]
    for name, kind, src in variants:
        if kind is None: Tf = lambda X: X
        else:
            ref = {'R1': R1, 'R': R, 'R2': R2, 'Q': Q}[src]; Tf = (T_std if kind == 'std' else T_white)(ref)
        T1, T2 = Tf(R1), Tf(R2); RT = np.empty((len(R), T1.shape[1])); RT[0::2] = T1; RT[1::2] = T2          # keep the parity layout for the split inside Ref
        p = Ref(RT, K, 'torch', workers=thr).p(Tf(Q))
        cnt[name] += [(p <= t).sum() for t in ts]; anyflag[name] += int(bh(p).sum() > 0)
    tot += N
    if ev % 25 == 0: print('  eval', ev, flush=True)
print(f'== exactness of normalised scores | N={N} k={K} features {R.shape[1]} (intrinsic 32) | {E} evaluations | exact null; pooled P(p<=t)/t for t = ' + ' '.join(f'{t:g}' for t in ts) + ' | evaluations with a BH flag')
for name, _, _ in variants:
    print(f'{name:10s} | ' + ' '.join(f'{c / tot / t:6.3f}' for c, t in zip(cnt[name], ts)) + f' | {anyflag[name]}/{E}')

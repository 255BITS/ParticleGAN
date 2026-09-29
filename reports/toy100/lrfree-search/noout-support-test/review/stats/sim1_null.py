"""Q1: null validity of the split-conformal p-values of the shipped support test when the table rows are scored at their CLEAN centres and the
reservoir holds noisy reals. Family x row-scoring mode; table fixed (drawn once), fresh reservoir per evaluation.
usage: python sim1_null.py FAMILY NEVALS [THREADS]      FAMILY in grid100 | unequal | g8 | g32 | g128
modes: iid      rows are fresh draws of the real law (exact conformal null; benchmark)
       clean967 rows at the clean centres of a generator whose output noise is sigma = .967 s (= .029/.03, the declared native noise), scored clean
       noisy967 the same rows scored as noisy draws (clean + sigma eps): the real fake law = real law
       clean500 / noisy500: same with sigma = .5 s (weaker output noise: the clean rows carry half of the spread)"""
import sys, json, time, math, numpy as np, torch
import fam
from scorer import Ref
from kdlib import bh
name, nev = sys.argv[1], int(sys.argv[2]); thr = int(sys.argv[3]) if len(sys.argv) > 3 else 2
torch.set_num_threads(thr)
N, K = 20000, 10
F = {'grid100': fam.grid100, 'unequal': fam.unequal_mixed, 'g8': lambda: fam.highdim(8), 'g32': lambda: fam.highdim(32), 'g128': lambda: fam.highdim(128)}[name]()
backend = 'kd' if name in ('grid100', 'unequal') else 'torch'
g = np.random.default_rng(12345)
s_sig = {'967': .967, '500': .5}
# table rows (fixed for the whole run): one clean set per noise level, one iid set; noisy versions add sigma * eps fresh in every evaluation? no: the
# noisy-scored version scores the rows' fake draws (clean + sigma eps'), redrawn each evaluation as the GAN does (fresh generator noise).
sig_abs = {k: v * float(np.mean(F.s)) if name != 'unequal' else v * .03 for k, v in s_sig.items()}   # sigma_out is a global constant: .967*.03=.029 (or .015)
rows = {}
rows['iid'] = F.real(N, g)[0]
for key, sg in sig_abs.items():
    if name == 'unequal':      # global sigma_out = sg; components with s >= sg only (all are >= .03)
        rows['clean' + key], lab = F.clean(N, g, sg)
    else:
        rows['clean' + key], lab = F.clean(N, g, sg)
modes = ['iid', 'clean967', 'noisy967', 'clean500', 'noisy500']
res = {m: dict(p_hist=[], nflag=[], pmed=[]) for m in modes}
thresholds = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 5e-2, .1, .25, .5]
cnt = {m: np.zeros(len(thresholds)) for m in modes}; tot = {m: 0 for m in modes}; pall = {m: [] for m in modes}
t0 = time.time()
for ev in range(nev):
    R = F.feat(F.real(N, g)[0])
    ref = Ref(R, K, backend=backend, workers=thr)
    for m in modes:
        if m == 'iid': x = rows['iid']
        elif m.startswith('clean'): x = rows[m]
        else:
            key = m[5:]; x = rows['clean' + key] + sig_abs[key] * g.standard_normal(rows['clean' + key].shape)
        p = ref.p(F.feat(x))
        fl = bh(p); res[m]['nflag'].append(int(fl.sum()))
        for i, t in enumerate(thresholds): cnt[m][i] += float((p <= t).sum())
        tot[m] += len(p)
        if ev < 30: pall[m].append(p.astype(np.float32))
    if ev % 50 == 0:
        print(f'  eval {ev}/{nev} {time.time() - t0:.0f}s', flush=True)
out = dict(family=F.name, N=N, k=K, evals=nev, thresholds=thresholds, modes={})
print(f'== {F.name} | N={N} k={K} | {nev} fresh reservoirs (table fixed) | zero-rho reference points in last R1: {ref.n_zero_rho}')
print('mode      | P(p<=t) / t for t = ' + ' '.join(f'{t:g}' for t in thresholds) + ' | evals with >=1 flag (95% CP upper) | mean flagged rows/eval | max | KS(p) first 30 evals')
from scipy.stats import beta
for m in modes:
    fr = cnt[m] / tot[m]; nf = np.array(res[m]['nflag']); anyf = int((nf > 0).sum())
    up = beta.ppf(.975, anyf + 1, nev - anyf) if anyf < nev else 1.
    pp = np.sort(np.concatenate(pall[m])); ks = float(np.max(np.abs(pp - (np.arange(1, len(pp) + 1) / len(pp)))))
    ratio = ' '.join(f'{fr[i] / thresholds[i]:.3f}' for i in range(len(thresholds)))
    print(f'{m:9s} | {ratio} | {anyf}/{nev} ({up:.4f}) | {nf.mean():.3f} | {nf.max()} | {ks:.4f}')
    out['modes'][m] = dict(p_le=fr.tolist(), evals_flag=anyf, mean_flagged=float(nf.mean()), max_flagged=int(nf.max()), ks=ks, nflag=nf.tolist())
json.dump(out, open(f'out/sim1_{name}.json', 'w'))

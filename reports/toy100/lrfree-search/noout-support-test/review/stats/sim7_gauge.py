"""Gauge / metric-normalisation experiment requested with the codex feature-gauge finding (e17-feature-gauge-review).
Synthetic critic features: d-dimensional Gaussian mixtures (50 comps, sd .03/coord; d = 8, 32, 128) or the native-like 2-D lattice (d = 2, 100 modes) pushed through a fixed random
ReLU network (width h). Exact symmetry of a ReLU network: positive rescaling of any hidden unit (features c_i * f_i with c_i = exp(s N(0,1)), next-layer weights divided by c_i;
the critic function is unchanged). Variants of the isolation score (all computed by the SHIPPED arithmetic, only the features are transformed BEFORE the test):
  raw       the features as E17 uses them
  std       each feature divided by its std on the reference half R1 (guard: features with std < 1e-9 max std left alone)
  white     (x - mean_R1) P with P from the eigen-decomposition of the R1 covariance, pseudo-inverse of eigenvalues < 1e-10 x max
  std_full / white_full: the same statistics estimated on the WHOLE reservoir R1+R2 (breaks the exact split-conformal argument: the score function then depends on the calibration half)
Per variant: null (rows = fresh iid draws; rows = clean atoms of a generator with sigma_out = .967 s) false flags per evaluation, P(p<=1e-3)/1e-3 and P(p<=1e-4)/1e-4 of iid rows,
recall of 300 planted clean strays at displacement mu * r0 (r0 = s sqrt(d) is the noise radius) in DATA space, legit rows flagged.
usage: python sim7_gauge.py D EVALS THREADS"""
import sys, json, time, math, numpy as np, torch
import fam
from scorer import Ref
from kdlib import bh
d, E, thr = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); torch.set_num_threads(thr)
N, K, S = 20000, 10, .03
if d == 2:
    from simlib import relu_map
    F = fam.grid100(); f = relu_map(2, 128, seed=5, depth=3, bias_sd=.5); F.feature = lambda x: f(x * 3); F.name = 'grid100 -> ReLU128x3 (2-D manifold in 128-D)'
else:
    F = fam.highdim(d)
g = np.random.default_rng(700 + d)
H = F.feat(F.c[:1]).shape[1]
mult = [.5, 1, 1.5, 2, 3, 4, 6] if d > 2 else [3, 4, 5, 6, 8, 12]
r0 = S * math.sqrt(d) if d > 2 else S            # for d=2: mu in units of sigma
gauge_rng = torch.Generator().manual_seed(11 + d)
z_g = torch.randn(H, generator=gauge_rng, dtype=torch.float64).numpy()
def scaled(X, s): return X * np.exp(s * z_g)[None, :]
def transform(kind, Xs, R):
    """Xs: dict name -> feature array; R: reservoir features (rows alternate R1 R2). statistics from R[0::2] (or all of R for *_full)."""
    ref = R if kind.endswith('_full') else R[0::2]
    base = kind.replace('_full', '')
    if base == 'raw': return Xs
    if base == 'std':
        s = ref.std(0); s = np.where(s > 1e-9 * s.max(), s, 1.); return {k: v / s for k, v in Xs.items()}
    mu = ref.mean(0, keepdims=True); cov = ((ref - mu).T @ (ref - mu)) / (len(ref) - 1); ev, U = np.linalg.eigh(cov); keep = ev > 1e-10 * ev.max()
    P = U[:, keep] / np.sqrt(ev[keep]); return {k: (v - mu) @ P for k, v in Xs.items()}
VARIANTS = [('raw', 0.), ('raw', math.log(4)), ('raw', math.log(16)), ('std', 0.), ('std', math.log(16)), ('white', 0.), ('white', math.log(16)), ('std_full', 0.), ('white_full', 0.)]
acc = {v: dict(nfl_iid=[], nfl_clean=[], p3=0., p4=0., ntot=0, rec={m: 0. for m in mult}, legit=[], pr=[], ed=[]) for v in VARIANTS}
tab_clean = F.clean(N, g, .029 if d == 2 else .967 * S)[0]           # fixed clean atoms
t0 = time.time()
for ev in range(E):
    R = F.feat(F.real(N, g)[0]); Xiid = F.feat(F.real(N, g)[0]); Xcl = F.feat(tab_clean)
    lab = g.integers(0, F.M, 300); u = g.standard_normal((300, F.d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    strays = {m: F.feat(F.c[lab] + m * r0 * u) for m in mult}
    for (kind, s) in VARIANTS:
        Xs = {'R': scaled(R, s), 'iid': scaled(Xiid, s), 'cl': scaled(Xcl, s)}
        for m in mult: Xs[('s', m)] = scaled(strays[m], s)
        T = transform(kind, Xs, Xs['R']); ref = Ref(T['R'], K, 'torch', workers=thr)
        va = T['R'][0::2].var(0); acc[(kind, s)]['pr'].append(float(va.sum() ** 2 / (va ** 2).sum()))           # participation ratio of the feature variances (effective number of dimensions that carry distance)
        acc[(kind, s)]['ed'].append(float(np.median(ref.rho)))                                                     # median leave-one-out k-th neighbour radius (the local scale)
        piid = ref.p(T['iid']); pcl = ref.p(T['cl'])
        a = acc[(kind, s)]; a['nfl_iid'].append(int(bh(piid).sum())); a['nfl_clean'].append(int(bh(pcl).sum()))
        a['p3'] += float((piid <= 1e-3).sum()); a['p4'] += float((piid <= 1e-4).sum()); a['ntot'] += len(piid)
        for m in mult:
            ps = ref.p(T[('s', m)]); p = np.concatenate([piid[:N - 300], ps]); fl = bh(p); a['rec'][m] += fl[N - 300:].mean() / E
            if m == mult[0]: a['legit'].append(int(fl[:N - 300].sum()))
    print(f'  eval {ev + 1}/{E} {time.time() - t0:.0f}s', flush=True)
print(f'== d={d} ({F.name}) feature width {H} | N={N} k={K} | {E} evaluations | gauge: feature i scaled by exp(s z_i), z_i ~ N(0,1) fixed; noise radius r0 = {r0:.4f}')
print('variant (gauge s)      | false flags/eval iid / clean atoms | P(p<=1e-3)/1e-3 P(p<=1e-4)/1e-4 (iid rows) | recall of 300 strays at displacement ' + ' '.join(f'{m:g}' for m in mult) + ' | legit rows flagged/eval | participation ratio of feature variances | median rho')
out = {}
for (kind, s), a in acc.items():
    print(f'{kind:10s} s={s:5.2f}      | {np.mean(a["nfl_iid"]):5.2f} / {np.mean(a["nfl_clean"]):5.2f} (max {max(a["nfl_iid"])}/{max(a["nfl_clean"])})   | {a["p3"] / a["ntot"] / 1e-3:5.2f} {a["p4"] / a["ntot"] / 1e-4:5.2f} | ' + ' '.join(f'{a["rec"][m]:.3f}' for m in mult) + f' | {np.mean(a["legit"]):.2f} | {np.mean(a["pr"]):.1f} | {np.mean(a["ed"]):.3g}')
    out[f'{kind}@{s:.2f}'] = dict(nfl_iid=a['nfl_iid'], nfl_clean=a['nfl_clean'], p3=a['p3'] / a['ntot'], p4=a['p4'] / a['ntot'], rec=a['rec'], legit=a['legit'])
json.dump(out, open(f'out/sim7_gauge_d{d}.json', 'w'))

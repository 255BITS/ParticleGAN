"""Q3: the local-scale ratio. For each 2-D family (isotropic modes, dense core inside a halo, thin elongated Gaussians, ring, spiral):
 (i)  enrichment of small p-values by region under the exact null (rows = fresh draws of the real law): P(p <= 1e-3 | region) / 1e-3 (1 = calibrated)
 (ii) legit (bulk) rows flagged by BH by region when 300 far strays lift the BH threshold (the situation of the training runs)
 (iii) power (recall of planted strays at BH .05) vs distance t (units of the thin-direction noise sd), for 60 and 300 strays
usage: python sim3_local.py EVALS_NULL EVALS_POWER [families...]"""
import sys, json, numpy as np
import fam2d
from scorer import Ref
from kdlib import bh
EN, EP = int(sys.argv[1]), int(sys.argv[2]); only = sys.argv[3:]
N, K = 20000, 10
g = np.random.default_rng(99)
res = {}
for cls in fam2d.ALL:
    F = cls()
    if only and cls.__name__ not in only: continue
    nreg = len(F.names)
    print(f'\n===== {F.name}')
    enr = np.zeros(nreg); cnt = np.zeros(nreg); enr5 = np.zeros(nreg); fl_reg = np.zeros(nreg); bsc = [[] for _ in range(nreg)]; asc = [[] for _ in range(nreg)]
    tot_fl = []; tot_bulk_fl = []
    for ev in range(EN):
        R, _ = F.draw(N, g); ref = Ref(R, K, 'kd', workers=2)
        x, reg = F.draw(N, g); p, s, a, b = ref.p(x, return_all=True)
        for j in range(nreg):
            sel = reg == j; cnt[j] += sel.sum(); enr[j] += (p[sel] <= 1e-3).sum(); enr5[j] += (p[sel] <= 1e-2).sum()
            if ev < 10 and sel.any(): bsc[j].append(np.median(b[sel])); asc[j].append(np.median(s[sel]))
        # (ii) 300 far strays lift the threshold
        st = F.stray(300, 12., g)
        if st is not None:
            x2 = np.concatenate([x[:N - 300], st]); reg2 = reg[:N - 300]
            fl = bh(ref.p(x2)); tot_fl.append(int(fl.sum())); tot_bulk_fl.append(int(fl[:N - 300].sum()))
            for j in range(nreg): fl_reg[j] += fl[:N - 300][reg2 == j].sum()
    print(f'(i) null, {EN} evaluations of {N} rows: P(p<=1e-3 | region)/1e-3 and P(p<=1e-2 | region)/1e-2, median local scale b/sperp, median score')
    for j in range(nreg):
        if cnt[j] == 0: continue
        print(f'   {F.names[j]:24s} n={int(cnt[j]):9d} ({cnt[j] / cnt.sum():.4f}) | enrichment @1e-3: {enr[j] / cnt[j] / 1e-3:6.2f} | @1e-2: {enr5[j] / cnt[j] / 1e-2:5.2f} | median b/sperp {np.median(bsc[j]) / F.sperp:8.3f} | median score {np.median(asc[j]):.3f}')
    if tot_fl:
        print(f'(ii) 300 far strays (t=12) planted; BH flagged {np.mean(tot_fl):.1f} rows/eval of which {np.mean(tot_bulk_fl):.2f} legit rows/eval (FDR proxy {np.mean(np.array(tot_bulk_fl) / np.maximum(tot_fl, 1)):.4f}); legit rows flagged per evaluation by region (share of that region\'s rows):')
        for j in range(nreg):
            if cnt[j]: print(f'   {F.names[j]:24s} {fl_reg[j] / EN:7.3f} rows/eval = {fl_reg[j] / (cnt[j] * (N - 300) / N):.2e} of the region')
    res[cls.__name__] = dict(names=F.names, null_enrichment_1e3=(enr / np.maximum(cnt, 1) / 1e-3).tolist(), flagged_bulk_per_eval=(fl_reg / EN).tolist(),
                             region_fraction=(cnt / cnt.sum()).tolist())
    # (iii) power
    if F.stray(1, 5., g) is not None:
        print(f'(iii) power = recall of planted strays at BH .05 vs distance t (sd units of thin direction = {F.sperp}); {EP} evaluations each; bulk = fresh real draws')
        print('      t:     ' + ' '.join(f'{t:6g}' for t in (2, 3, 4, 5, 6, 8, 12, 16, 24, 32)))
        for m1 in (60, 300, 1000):
            row = []; med_b = []
            for t in (2, 3, 4, 5, 6, 8, 12, 16, 24, 32):
                rec = 0.
                for ev in range(EP):
                    R, _ = F.draw(N, g); ref = Ref(R, K, 'kd', workers=2)
                    x, _ = F.draw(N - m1, g); st = F.stray(m1, t, g)
                    fl = bh(ref.p(np.concatenate([x, st]))); rec += fl[N - m1:].mean()
                row.append(rec / EP)
            print(f'  m1={m1:5d} ' + ' '.join(f'{v:6.3f}' for v in row), flush=True)
            res[cls.__name__]['power_m%d' % m1] = row
json.dump(res, open('out/sim3_local.json', 'w'))

"""Design check of the isolation statistic on synthetic clouds (evaluation side; the statistic is space-agnostic, here the 'space' is the plane).
Three scores for a query point u against a reference half R1 of the real reservoir (k = 10 as in the birth-death):
  raw : a(u) = distance to the k-th nearest reference point
  lof1: a(u) / rho(nn(u)), rho(j) = leave-one-out k-th NN radius of reference point j (local real scale at the nearest real)
  lofk: a(u) / median_j rho(nn_j(u)) over the k nearest reference points
Split-conformal p-value against the other half R2 (exchangeable with fakes under 'fake law = real law'), Benjamini-Hochberg at .05 over the rows.
Test A: a mixture with very unequal component masses (vector_unequal_mass), ideal fakes + 1.4% planted strays.
Test B: equal-mass modes (native-like), ideal fakes + 1.4% strays.  usage: python isolation_stat.py"""
import math, numpy as np, torch
torch.set_num_threads(8)
K = 10
def knn(q, ref, k, exclude_self=False, chunk=2048):
    ds, ids = [], []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], ref)
        if exclude_self: d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float('inf')
        v, ix = d.topk(k, dim=1, largest=False); ds.append(v); ids.append(ix)
    return torch.cat(ds), torch.cat(ids)
def scores(q, R1):
    rho = knn(R1, R1, K, exclude_self=True)[0][:, -1]
    d, ix = knn(q, R1, K)
    a = d[:, -1]
    return dict(raw=a, lof1=a / rho[ix[:, 0]].clamp_min(1e-12), lofk=a / rho[ix].median(1).values.clamp_min(1e-12))
def bh(p, q=.05):
    o = np.argsort(p); ps = p[o]; m = len(p); ok = np.where(ps <= q * np.arange(1, m + 1) / m)[0]
    f = np.zeros(m, bool)
    if len(ok): f[o[:ok.max() + 1]] = True
    return f
def run(name, sample_real, comp_of, n_strays_frac=.014, N=20000, M=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    R = sample_real(M, g); R1, R2 = R[0::2], R[1::2]
    fake, lab = sample_real(N, g, labels=True)
    ns = int(n_strays_frac * N); lo, hi = R.min(0).values - .3, R.max(0).values + .3
    stray = lo + (hi - lo) * torch.rand(ns, 2, generator=g)
    # a planted stray must be >= 6 sigma-equivalents from every reference point: keep only those with a(u) large
    fake = torch.cat([fake[:N - ns], stray]); lab = torch.cat([lab[:N - ns], torch.full((ns,), -1)])
    s_f, s_n = scores(fake, R1), scores(R2, R1)
    print(f'== {name}: {N} fakes, {ns} planted strays (uniform in the bounding box), reservoir {M} (split {len(R1)}/{len(R2)}), k={K}')
    for key in ('raw', 'lof1', 'lofk'):
        null = np.sort(s_n[key].numpy()); sf = s_f[key].numpy()
        p = (1 + (len(null) - np.searchsorted(null, sf, side='left'))) / (1 + len(null))
        fl = bh(p)
        line = f'  {key:5s} flagged {int(fl.sum()):5d} | strays: recall {fl[lab.numpy() == -1].mean():.3f} | legit fakes flagged {fl[lab.numpy() >= 0].mean():.4f}'
        for c in sorted(set(lab.numpy().tolist()) - {-1}):
            sel = lab.numpy() == c
            line += f' | comp{c}: {fl[sel].mean():.4f}'
        print(line)
means = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
def unequal(n, g, labels=False, masses=(.55, .30, .13, .02), sd=.18):
    c = torch.multinomial(torch.tensor(masses), n, replacement=True, generator=g)
    x = means[c] + sd * torch.randn(n, 2, generator=g)
    return (x, c) if labels else x
def equal100(n, g, labels=False, sd=.03):
    co = torch.arange(10.) - 4.5; ce = torch.stack(torch.meshgrid(co, co, indexing='ij'), -1).reshape(-1, 2)
    c = torch.randint(0, 100, (n,), generator=g); x = ce[c] + sd * torch.randn(n, 2, generator=g)
    return (x, c) if labels else x
run('A unequal mass (masses .55/.30/.13/.02, sd .18)', unequal, None)
run('B equal 100 modes (sd .03)', equal100, None)

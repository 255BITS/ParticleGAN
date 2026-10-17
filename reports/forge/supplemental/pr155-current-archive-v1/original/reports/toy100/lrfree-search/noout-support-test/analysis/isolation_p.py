"""Offline check (evaluation side; the statistic itself uses only the critic's feature space): can a per-row isolation p-value, computed in the
critic's feature space against a reservoir of reals passed through the SAME critic, tell strays from bulk rows?
Score of a point = distance (in feature space) to its k-th nearest real. Under 'fake law = real law' a fake's score has the law of a real's leave-one-out score
(exchangeable), so p(f) = (1 + #{reals with score >= score(f)}) / (1 + n) is a valid conformal p-value. Strays (rows > 3 sigma from every centre) are
ground truth for this check only.
usage: python isolation_p.py RUN_DIR TASK [K] [NREAL]"""
import sys, math, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics')
sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E14')
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
from lib import centers, assign, SD
from toy_models import SimpleMLPDiscriminator
torch.set_num_threads(8)
run, task = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
tr = torch.load(f'{run}/final-state.pt', map_location='cpu', weights_only=False)['trainer']
m = tr['models']; z = m['prior']['z'].double(); W = m['G']['weight'].double(); b = m['G']['bias'].double()
sig = float(tr['output_noise']['log_sigma'].exp())
x = (z @ W.T + b).numpy()
ce = centers(task); n, r = assign(x, ce); rad = np.linalg.norm(r, axis=1)
g = np.random.default_rng(0)
xf = torch.tensor(x + sig * g.standard_normal(x.shape), dtype=torch.float32)                      # noisy fakes as the critic sees them
mode = g.integers(0, 100, NR); xr = torch.tensor(ce[mode] + SD * g.standard_normal((NR, 2)), dtype=torch.float32)   # analysis-only real sample
D = SimpleMLPDiscriminator(2, 128, 3, 3); D.load_state_dict({k: v for k, v in m['D'].items()}); D.eval()
head = [mod for mod in D.modules() if isinstance(mod, torch.nn.Linear) and mod.out_features == 1]
print('linear heads with 1 output:', len(head), 'input dim', head[-1].in_features)
cap = {}
head[-1].register_forward_hook(lambda mod, inp, out: cap.__setitem__('f', inp[0].detach()))
@torch.no_grad()
def feats(a, chunk=8192):
    out = []
    for i in range(0, len(a), chunk):
        D(a[i:i + chunk]); out.append(cap['f'].clone())
    return torch.cat(out)
@torch.no_grad()
def kth(q, ref, k, exclude_self=False, chunk=1024):
    res = []
    for i in range(0, len(q), chunk):
        d = torch.cdist(q[i:i + chunk], ref)
        if exclude_self: d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float('inf')
        res.append(d.topk(k, dim=1, largest=False).values[:, -1])
    return torch.cat(res)
def auc(pos_score, neg_score):
    # probability that a random stray has a larger score than a random bulk row
    a = np.sort(neg_score); return float(np.mean(np.searchsorted(a, pos_score) / len(a)))
def bh(p, q=0.05):
    o = np.argsort(p); ps = p[o]; mm = len(p); thr = q * (np.arange(1, mm + 1)) / mm
    ok = np.where(ps <= thr)[0]; flag = np.zeros(mm, bool)
    if len(ok): flag[o[:ok.max() + 1]] = True
    return flag
bulk = rad < 2; stray = rad > 3
print(f'rows {len(x)} | bulk(<2s) {int(bulk.sum())} | strays(>3s) {int(stray.sum())} ({stray.mean():.4f}) | k={K} n_real={NR}')
for name, (qf, rf) in {'feature space (critic input of the score head)': (feats(xf), feats(xr)), 'raw sample space (upper bound; NOT rule-clean, analysis only)': (xf, xr)}.items():
    sf = kth(qf, rf, K).numpy(); sr = kth(rf, rf, K, exclude_self=True).numpy()
    srs = np.sort(sr); p = (1 + (len(srs) - np.searchsorted(srs, sf, side='left'))) / (1 + len(srs))
    fl = bh(p)
    print(f'--- {name}')
    print(f'  score median: bulk {np.median(sf[bulk]):.4g} | strays {np.median(sf[stray]):.4g} | reals (leave-one-out) {np.median(sr):.4g} p99.9 {np.quantile(sr, .999):.4g}')
    print(f'  AUC(stray vs bulk) = {auc(sf[stray], sf[bulk]):.4f} | median p: bulk {np.median(p[bulk]):.3f} strays {np.median(p[stray]):.4f} | fraction p<=.01: bulk {np.mean(p[bulk] <= .01):.4f} strays {np.mean(p[stray] <= .01):.3f}')
    print(f'  BH q=.05 over all rows: flagged {int(fl.sum())} | bulk flagged {int(fl[bulk].sum())} ({fl[bulk].mean():.4f}) | strays flagged {int(fl[stray].sum())} of {int(stray.sum())} (recall {fl[stray].mean():.3f}) | precision {fl[stray].sum() / max(1, fl.sum()):.3f}')
    for lo, hi in ((3, 4), (4, 6), (6, 10), (10, 16), (16, 1e9)):
        sel = (rad > lo) & (rad <= hi)
        if sel.any(): print(f'    band {lo}-{hi} sigma: n={int(sel.sum())} recall {fl[sel].mean():.3f} median p {np.median(p[sel]):.4f}')

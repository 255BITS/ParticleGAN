"""Diagnostic (evaluation-side) anatomy of table rows on a saved final state: where the >3 sigma rows are, and what the critic field looks like there.
usage: python stray_anatomy.py RUN_DIR TASK"""
import sys, math, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics')
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/pkg-seqC')
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
from lib import centers, assign, SD
from toy_models import SimpleMLPDiscriminator
torch.set_num_threads(4)
run, task = sys.argv[1], sys.argv[2]
tr = torch.load(f'{run}/final-state.pt', map_location='cpu', weights_only=False)['trainer']
m = tr['models']; z = m['prior']['z'].double(); W = m['G']['weight'].double(); b = m['G']['bias'].double()
x = (z @ W.T + b).numpy()
ce = centers(task)
n, r = assign(x, ce); rad = np.linalg.norm(r, axis=1)
print('controller latent_bandwidth', tr['controller']['latent_bandwidth'].tolist(), 'mobility', tr['controller']['mobility'], 'log_sigma', float(tr['output_noise']['log_sigma'].exp()))
print('G', W.numpy().round(4).tolist(), b.numpy().round(4).tolist())
bins = [0, 1, 2, 3, 4, 6, 10, 16, 1e9]
h = np.histogram(rad, bins)[0]
print('row count by distance to nearest centre (sigma):', {f'{bins[i]}-{bins[i+1]}': int(h[i]) for i in range(len(h))}, 'stray(>3)=', float((rad > 3).mean()))
# hull membership of strays (interior vs exterior of the lattice hull)
from scipy.spatial import Delaunay
hull = Delaunay(ce)
inside = hull.find_simplex(x) >= 0
st = rad > 3
print('strays inside hull', float((st & inside).sum() / max(1, st.sum())))
# distance of strays to the nearest centre in raw units and their local table density
from scipy.spatial import cKDTree
t = cKDTree(x)
cnt3 = np.array([len(t.query_ball_point(p, 0.09)) - 1 for p in x[st]]) if st.any() else np.array([])
print('strays: median dist (sigma)', float(np.median(rad[st])), 'p90', float(np.percentile(rad[st], 90)), 'table neighbours within 3 sigma: mean', float(cnt3.mean()), 'frac singleton', float((cnt3 == 0).mean()))
# critic field (mean gradient over output noise draws) at rows
D = SimpleMLPDiscriminator(2, 128, 3, 3)
D.load_state_dict({k: v for k, v in tr['models']['D'].items()})
D.eval()
g = torch.Generator().manual_seed(0)
def field(xx, K=16, sig=float(tr['output_noise']['log_sigma'].exp())):
    xt = torch.tensor(xx, dtype=torch.float32)
    G = torch.zeros_like(xt); V = torch.zeros(len(xt)); s2 = torch.zeros(len(xt))
    for _ in range(K):
        q = (xt + sig * torch.randn(xt.shape, generator=g)).requires_grad_(True)
        d = D(q); gr = torch.autograd.grad(d.sum(), q)[0]
        G += gr; V += d.detach(); s2 += gr.square().sum(-1)
    return (G / K).numpy(), (V / K).numpy(), np.sqrt((s2 / K).numpy())
sel_b = np.where(rad < 2)[0]; sel_b = sel_b[np.random.default_rng(0).choice(len(sel_b), min(3000, len(sel_b)), replace=False)]
sel_s = np.where(st)[0]
for name, sel in (('bulk(<2s)', sel_b), ('strays(>3s)', sel_s)):
    if len(sel) == 0: continue
    Gm, Vm, rms = field(x[sel])
    dirc = -r[sel] / np.maximum(np.linalg.norm(r[sel], axis=1, keepdims=True), 1e-12)   # direction to nearest centre
    mg = np.linalg.norm(Gm, axis=1)
    cos = (Gm * dirc).sum(1) / np.maximum(mg, 1e-12)
    print(f'{name:12s} n={len(sel):5d} |mean grad| med {np.median(mg):.5f} p90 {np.percentile(mg,90):.5f}  rms grad(per-noise-draw) med {np.median(rms):.5f}  '
          f'mean/rms med {np.median(mg/np.maximum(rms,1e-12)):.3f}  cos(meangrad, dir to centre) mean {cos.mean():+.3f}  D med {np.median(Vm):+.4f}')
# D profile from strays toward nearest centre: D(s + t*(c-s)), t in 0..1 (mean over strays), and |grad D| along it
if len(sel_s):
    ce_n = ce[n[sel_s]]
    for t_ in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 0.95):
        p = x[sel_s] + t_ * (ce_n - x[sel_s])
        Gm, Vm, rms = field(p, K=4)
        print(f'  t={t_:4.2f}  D mean {Vm.mean():+.4f}  |grad D| med {np.median(np.linalg.norm(Gm,axis=1)):.5f}')

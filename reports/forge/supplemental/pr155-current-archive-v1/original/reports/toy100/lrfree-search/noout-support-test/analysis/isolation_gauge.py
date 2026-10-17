"""Gauge sensitivity of the support test on a REAL critic (native100 final state): codex's audit (reports/toy100/lrfree-search/e17-feature-gauge-review) shows that for a
critic whose head is fed by a purely linear hidden map, rescaling the hidden coordinates (same critic function) changes E17's decisions. Real critics have a nonlinearity before
the head, whose exact symmetry group is a positive rescaling of each hidden unit (ReLU / LeakyReLU): here every one of the head's input features is rescaled by c_i = exp(s * N(0,1)),
then the E17 test (`ParticleBirthDeath._isolated`, called unchanged) is run on (a) the raw features (what E17 does), (b) per-feature standardised features (each feature divided by its
std on the reference half of the reservoir: invariant to that group), (c) fully whitened features (reference covariance; pseudo-inverse of the tiny eigenvalues).
Also: the birth-death density-ratio statistic x_i = d log(r_R / r_F) (rank correlation with the un-rescaled raw x).
usage: python isolation_gauge.py RUN_DIR TASK   (CPU)"""
import sys, importlib.util, math, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics'); sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17')
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
from lib import centers, assign, SD
from toy_models import SimpleMLPDiscriminator
spec = importlib.util.spec_from_file_location('bd17', '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17/particlegan/birth_death.py')
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
torch.set_num_threads(8)
run, task = sys.argv[1], sys.argv[2]
K = 10
class Dummy: Q = .05
tr = torch.load(f'{run}/final-state.pt', map_location='cpu', weights_only=False)['trainer']
m = tr['models']; z = m['prior']['z'].double(); W = m['G']['weight'].double(); b = m['G']['bias'].double(); sig = float(tr['output_noise']['log_sigma'].exp())
x_clean = (z @ W.T + b); ce = centers(task); n, r = assign(x_clean.numpy(), ce); rad = np.linalg.norm(r, axis=1)
g = np.random.default_rng(0)
NR = 20000
xr = torch.tensor(ce[g.integers(0, 100, NR)] + SD * g.standard_normal((NR, 2)), dtype=torch.float32)                      # analysis-only reals
pick = g.integers(0, len(z), len(z))
xf = torch.tensor((x_clean[pick] + sig * torch.randn(len(z), 2, dtype=torch.float64)).numpy(), dtype=torch.float32)        # a fake pool as birth-death draws it
D = SimpleMLPDiscriminator(2, 128, 3, 3); D.load_state_dict({k: v for k, v in m['D'].items()}); D.eval()
head = [mod for mod in D.modules() if isinstance(mod, torch.nn.Linear) and mod.out_features == 1][-1]
cap = {}; head.register_forward_hook(lambda mod, inp, out: cap.__setitem__('f', inp[0].detach()))
@torch.no_grad()
def feats(a, chunk=8192):
    out = []
    for i in range(0, len(a), chunk): D(a[i:i + chunk]); out.append(cap['f'].clone())
    return torch.cat(out).double()
Fq, FR, FF = feats(x_clean.float()), feats(xr), feats(xf)
print(f'{run}: {len(z)} rows, strays(>3 sigma) {int((rad > 3).sum())}, far strays(>6) {int((rad > 6).sum())}, bulk(<2) {int((rad < 2).sum())}; head input width {Fq.shape[1]}; '
      f'feature std quantiles (10/50/90%): {[round(float(v), 4) for v in np.quantile(FR.std(0).numpy(), [.1, .5, .9])]}, dead features {int((FR.std(0) < 1e-9).sum())}')
def normalise(kind, A, B, C):
    """A: table features, B: reservoir features, C: fake-pool features -> transformed copies; statistics of the reference half only."""
    if kind == 'raw': return A, B, C
    ref = B[0::2]
    if kind == 'std':
        s = ref.std(0); s = torch.where(s > 1e-9 * float(s.max()), s, torch.ones_like(s)); return A / s, B / s, C / s
    if kind == 'white':
        mu = ref.mean(0, keepdim=True); cov = ((ref - mu).T @ (ref - mu)) / (len(ref) - 1)
        ev, U = torch.linalg.eigh(cov); keep = ev > 1e-10 * float(ev.max()); P = U[:, keep] / ev[keep].sqrt()
        return (A - mu) @ P, (B - mu) @ P, (C - mu) @ P
def stats(A, B, C):
    Bc = B.mean(0, keepdim=True); A, B, C = A - Bc, B - Bc, C - Bc
    fl = bd.ParticleBirthDeath._isolated(Dummy(), A, B, K, torch.float32)
    rR = bd._knn(A, B, K, shortlist_dtype=torch.float32)[0][:, -1]; rF = bd._knn(A, C, K, shortlist_dtype=torch.float32)[0][:, -1]
    dR = bd._levina_bickel(bd._knn(B, B, K, exclude=torch.arange(len(B)), shortlist_dtype=torch.float32)[0])[0]
    d = min(max(dR, 1.), 2.)
    return fl.numpy(), (d * (rR / rF).log()).numpy()
from scipy.stats import spearmanr
base_flag, base_x = stats(*normalise('raw', Fq, FR, FF))
def line(name, fl, xx):
    bulk, st, far = rad < 2, rad > 3, rad > 6
    jac = (fl & base_flag).sum() / max(1, (fl | base_flag).sum())
    return (f'  {name:30s} flagged {int(fl.sum()):4d} | strays recall {fl[st].mean():.3f} | far(>6) recall {fl[far].mean():.3f} | bulk false {int(fl[bulk].sum()):3d} ({fl[bulk].mean():.4f}) '
            f'| overlap with base raw {jac:.2f} | rank corr of BD x with base raw {spearmanr(xx, base_x)[0]:.3f}')
for s in (0.0, math.log(2), math.log(4), math.log(16)):
    gen = torch.Generator().manual_seed(int(1000 * s) + 7)
    c = torch.exp(s * torch.randn(Fq.shape[1], generator=gen, dtype=torch.float64))
    print(f'gauge: each of the {Fq.shape[1]} features rescaled by exp({s:.2f} N(0,1))  (c spans {float(c.min()):.2f}..{float(c.max()):.2f})')
    for kind in ('raw', 'std', 'white'):
        fl, xx = stats(*normalise(kind, Fq * c, FR * c, FF * c)); print(line(kind, fl, xx))

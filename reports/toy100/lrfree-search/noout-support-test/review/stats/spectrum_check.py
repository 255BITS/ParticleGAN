"""Feature covariance spectra (reference half R1 of 20,000 reals) of (a) the REAL native critic (E14s-rotated100 final state, head input, 128-D) and (b) the synthetic random-ReLU feature families of the
gauge experiment: how many directions carry the variance (participation ratio), how many eigenvalues lie below 1e-4 / 1e-8 / 1e-10 of the largest (the directions whitening amplifies to unit variance),
condition number of the retained spectrum.  usage: python spectrum_check.py"""
import sys, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics'); sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E14'); sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
torch.set_num_threads(2)
from lib import centers, SD
from toy_models import SimpleMLPDiscriminator
import fam
from simlib import relu_map
def spec(X, name):
    R1 = X[0::2]; cov = np.cov(R1.T); ev = np.linalg.eigvalsh(cov)[::-1]; ev = np.clip(ev, 0, None); pr = ev.sum() ** 2 / (ev ** 2).sum(); mx = ev[0]
    std = R1.std(0); prs = (std ** 2).sum() ** 2 / (std ** 4).sum()
    print(f'{name:38s} width {X.shape[1]:4d} | eigen participation ratio {pr:6.1f} | per-feature-variance participation ratio {prs:6.1f} | #eig > 1e-4/1e-8/1e-10 of max: {int((ev > 1e-4 * mx).sum())}/{int((ev > 1e-8 * mx).sum())}/{int((ev > 1e-10 * mx).sum())} | dead features {int((std < 1e-9 * std.max()).sum())} | log10 feature-std spread (p10..p90): {np.log10(np.quantile(std[std > 1e-9 * std.max()], .1)):.2f}..{np.log10(np.quantile(std[std > 1e-9 * std.max()], .9)):.2f} | top-2 eigen share {ev[:2].sum() / ev.sum():.3f} top-10 {ev[:10].sum() / ev.sum():.3f}')
g = np.random.default_rng(0)
tr = torch.load('/ml2/hypergan/gan-attempts/noout-20260928/runs/E14s-rotated100/final-state.pt', map_location='cpu', weights_only=False)['trainer']['models']
D = SimpleMLPDiscriminator(2, 128, 3, 3); D.load_state_dict(tr['D']); D.eval(); cap = {}
head = [m for m in D.modules() if isinstance(m, torch.nn.Linear) and m.out_features == 1][-1]; head.register_forward_hook(lambda m, i, o: cap.__setitem__('f', i[0].detach()))
ce = centers('rotated100'); xr = torch.tensor(ce[g.integers(0, 100, 20000)] + SD * g.standard_normal((20000, 2)), dtype=torch.float32)
with torch.no_grad(): D(xr)
spec(cap['f'].double().numpy(), 'REAL native critic (E14s-rotated100)')
F2 = fam.grid100(); f = relu_map(2, 128, seed=5, depth=3, bias_sd=.5); F2.feature = lambda x: f(x * 3); spec(F2.feat(F2.real(20000, g)[0]), 'synthetic d=2 (lattice -> ReLU128x3)')
for d in (8, 32, 128):
    F = fam.highdim(d); spec(F.feat(F.real(20000, g)[0]), f'synthetic d={d} (50 comps -> ReLU256x2)')

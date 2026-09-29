"""Real critic, real table: what does the SHIPPED `_isolated` do on the final states of E14s (no isolation; 1.6% strays) and E17dbg (isolation on) when the table rows are
scored (a) at their CLEAN centres G(z_i) (what maybe_apply does) or (b) as noisy draws G(z_i) + sigma eps (what analysis/isolation_p.py scored)?
Reals: analysis-only fresh samples of the task law (100 Gaussians, sd .03, lattice of the task) through the same critic (feature = input of the 1-output Linear head).
Ground truth (analysis only): distance of the clean row to the nearest centre in sigma units. usage: python real_state_check.py THREADS"""
import sys, math, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics'); sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E14'); sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
from lib import centers, assign, SD
from toy_models import SimpleMLPDiscriminator
from simlib import shipped_flags, isolation_scores, bh_flags, K_of
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
W = '/ml2/hypergan/gan-attempts/noout-20260928/runs/'
def load(run):
    tr = torch.load(W + run + '/final-state.pt', map_location='cpu', weights_only=False)['trainer']; m = tr['models']
    z = m['prior']['z'].double(); Wg = m['G']['weight'].double(); b = m['G']['bias'].double(); sig = float(tr['output_noise']['log_sigma'].exp())
    D = SimpleMLPDiscriminator(2, 128, 3, 3); D.load_state_dict(m['D']); D.eval()
    head = [mod for mod in D.modules() if isinstance(mod, torch.nn.Linear) and mod.out_features == 1][-1]; cap = {}
    head.register_forward_hook(lambda mod, inp, out: cap.__setitem__('f', inp[0].detach()))
    @torch.no_grad()
    def feats(a, chunk=8192):
        out = []
        for i in range(0, len(a), chunk):
            D(a[i:i + chunk].float()); out.append(cap['f'].clone())
        return torch.cat(out).double()
    return z, Wg, b, sig, feats
def report(tag, fl, rad):
    bands = ((0, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 1e9)); s = f'{tag}: flagged {int(fl.sum()):4d} |'
    for lo, hi in bands:
        sel = (rad > lo) & (rad <= hi) if lo > 0 else (rad <= hi)
        s += f' {lo:g}-{hi:g}s: {int(fl[sel].sum())}/{int(sel.sum())}' if sel.any() else ''
    return s
for run, task in (('E14s-rotated100', 'rotated100'), ('E14s-grid100', 'grid100'), ('E14s-staggered100', 'staggered100'), ('E17dbg-rotated100', 'rotated100'), ('E17a-grid100', 'grid100')):
    try: z, Wg, b, sig, feats = load(run)
    except Exception as e: print(run, 'skip', repr(e)[:80]); continue
    x = (z @ Wg.T + b); ce = centers(task); _, r = assign(x.numpy(), ce); rad = np.linalg.norm(r, axis=1)
    strays = rad > 3
    print(f'\n=== {run} ({task}) | rows {len(x)} | strays (clean position > 3 sigma from every centre): {int(strays.sum())} ({strays.mean():.4f}) | output sigma {sig:.4f} | k={K_of(len(x))}')
    for rep in range(3):
        g = np.random.default_rng(100 + rep)
        xr = torch.tensor(ce[g.integers(0, 100, 20000)] + SD * g.standard_normal((20000, 2)), dtype=torch.float32)
        R = feats(xr)
        q_clean = feats(x.float()); q_noisy = feats((x + sig * torch.randn(x.shape, generator=torch.Generator().manual_seed(rep), dtype=torch.float64)).float())
        R = R - R.mean(0, keepdim=True); mu = feats(xr).mean(0, keepdim=True)
        R_raw = feats(xr); mu = R_raw.mean(0, keepdim=True)
        fc = shipped_flags(q_clean - mu, R_raw - mu); fn = shipped_flags(q_noisy - mu, R_raw - mu)
        print(report(f'  rep{rep} CLEAN centres (shipped)', fc.numpy(), rad)); print(report(f'  rep{rep} NOISY draws (analysis/isolation_p.py)', fn.numpy(), rad))
        if rep == 0:
            p, s = isolation_scores((q_clean - mu), (R_raw - mu), 10); pn, sn = isolation_scores((q_noisy - mu), (R_raw - mu), 10)
            print('     median p: bulk<2s clean %.3f noisy %.3f | strays>3s clean %.2e noisy %.2e | P(p<=1e-3): bulk clean %.4f noisy %.4f' % (
                float(p[torch.as_tensor(rad < 2)].median()), float(pn[torch.as_tensor(rad < 2)].median()), float(p[torch.as_tensor(strays)].median()), float(pn[torch.as_tensor(strays)].median()),
                float((p[torch.as_tensor(rad < 2)] <= 1e-3).double().mean()), float((pn[torch.as_tensor(rad < 2)] <= 1e-3).double().mean())))

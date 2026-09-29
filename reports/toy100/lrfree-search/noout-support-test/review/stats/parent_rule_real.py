"""Q6 on a REAL table state: flag the rows of the E14s final state (rotated100, 1.6% strays) with the shipped `_isolated` (clean centres, real critic features), then draw
parents with the SHIPPED `_isolation_pick` arithmetic (stub trainer) and compare with two alternative rules. Report the distance (sigma units, analysis only) of the parents.
Rules: BALL (shipped E17: uniform among unflagged rows within 2x the nearest unflagged distance in z), NEAR1 (nearest unflagged row), BALLP (ball, weight = the row's own conformal p),
PONLY (ball restricted to unflagged rows with p >= .05 = Q)   [BALLP / PONLY use only the test's own p-values, no new constant except that PONLY reuses Q].
usage: python parent_rule_real.py THREADS"""
import sys, math, numpy as np, torch, types
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics'); sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E14'); sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
from lib import centers, assign, SD
from toy_models import SimpleMLPDiscriminator
from simlib import bd17, shipped_flags, isolation_scores, bh_flags, K_of
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
W = '/ml2/hypergan/gan-attempts/noout-20260928/runs/'
for run, task in (('E14s-rotated100', 'rotated100'), ('E14s-grid100', 'grid100'), ('E14s-staggered100', 'staggered100')):
    tr = torch.load(W + run + '/final-state.pt', map_location='cpu', weights_only=False)['trainer']; m = tr['models']
    z = m['prior']['z'].float(); Wg = m['G']['weight'].double(); b = m['G']['bias'].double(); x = (z.double() @ Wg.T + b)
    D = SimpleMLPDiscriminator(2, 128, 3, 3); D.load_state_dict(m['D']); D.eval(); cap = {}
    head = [mod for mod in D.modules() if isinstance(mod, torch.nn.Linear) and mod.out_features == 1][-1]; head.register_forward_hook(lambda mod, i, o: cap.__setitem__('f', i[0].detach()))
    def feats(a):
        with torch.no_grad(): out = []; [ (D(a[i:i + 8192].float()), out.append(cap['f'].clone())) for i in range(0, len(a), 8192)]; return torch.cat(out).double()
    ce = centers(task); _, r = assign(x.numpy(), ce); rad = np.linalg.norm(r, axis=1)
    g = np.random.default_rng(7); xr = torch.tensor(ce[g.integers(0, 100, 20000)] + SD * g.standard_normal((20000, 2)), dtype=torch.float32)
    R = feats(xr); q = feats(x.float()); mu = R.mean(0, keepdim=True)
    p, _ = isolation_scores(q - mu, R - mu, 10); fl = bh_flags(p)
    print(f'\n=== {run}: rows {len(z)}; strays(>3 sigma) {int((rad > 3).sum())}; flagged {int(fl.sum())} (of which >3 sigma: {int((fl.numpy() & (rad > 3)).sum())}); rows by class: <2 {int((rad < 2).sum())}, 2-3 {int(((rad >= 2) & (rad < 3)).sum())}, 3-4 {int(((rad >= 3) & (rad < 4)).sum())}, 4-6 {int(((rad >= 4) & (rad < 6)).sum())}, >=6 {int((rad >= 6).sum())}; unflagged 3-6 sigma rows {int(((~fl.numpy()) & (rad >= 3) & (rad < 6)).sum())}')
    # shipped parent rule via the real method with a stub
    class Stub: pass
    S = Stub(); S.Q = .05; S.N = len(z); S.counters = dict(iso_evals=0, iso_flagged=0, iso_acted=0, iso_skipped=0, iso_moves=0); S.iso_log = []; S.last = {}; S.stream = torch.Generator().manual_seed(1)
    tstub = types.SimpleNamespace(completed_steps=0, prior=types.SimpleNamespace(z=z))
    child = torch.zeros(0, dtype=torch.long)
    cls = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 1e9)]
    def summarize(name, par):
        pr = rad[par.numpy()]; return f'  {name:6s}: ' + ' | '.join(f'{lo:g}-{hi:g}s {np.mean((pr >= lo) & (pr < hi)):.3f}' for lo, hi in cls) + f' | mean {pr.mean():.2f}'
    if int(fl.sum()) > 0.05 * len(z) or not int(fl.sum()):
        print('  guard would skip this state'); continue
    dead, par = bd17.ParticleBirthDeath._isolation_pick(S, tstub, fl, child)
    print(f'  flagged rows re-drawn {len(dead)}; distance of the flagged rows themselves: ' + ' | '.join(f'{lo:g}-{hi:g}s {np.mean((rad[dead.numpy()] >= lo) & (rad[dead.numpy()] < hi)):.3f}' for lo, hi in cls))
    print('  parent distance classes (share of the parents), one draw per flagged row:'); print(summarize('BALL', par))
    keep = (~fl).nonzero().flatten(); dist = torch.cdist(z[dead], z[keep]); pk = p[keep]
    near1 = keep[dist.argmin(1)]; print(summarize('NEAR1', near1))
    ball = dist <= 2 * dist.min(1, keepdim=True).values
    w = ball.double() * pk[None, :]; ballp = keep[torch.multinomial(w, 1, generator=torch.Generator().manual_seed(2)).squeeze(1)]; print(summarize('BALLP', ballp))
    w2 = (ball & (pk[None, :] >= .05)).double(); w2[w2.sum(1) == 0] = ball.double()[w2.sum(1) == 0]; ponly = keep[torch.multinomial(w2, 1, generator=torch.Generator().manual_seed(3)).squeeze(1)]; print(summarize('PONLY', ponly))
    print('  rows in the ball per flagged row: median %.0f (min %d max %d); share of them at >= 3 sigma: %.3f' % (float(ball.sum(1).double().median()), int(ball.sum(1).min()), int(ball.sum(1).max()), float((ball & torch.as_tensor(rad[keep.numpy()] >= 3)[None, :]).sum().double() / ball.sum().double())))

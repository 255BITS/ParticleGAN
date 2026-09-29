"""Property test of the SHIPPED `_isolation_pick` (called on a stub trainer; tests/test_E15.py never asserts the ball semantics: E15/E16/E17 all pass its move checks).
Checks: (1) every parent is unflagged and not an ordinary-move child; (2) dist(child, parent) <= 2 * dist(child, nearest unflagged row); (3) parent draws are uniform over the ball
(chi-square on 4000 repeated draws for one flagged row); (4) the guard: nothing happens when 0 flagged or flagged > Q*N; (5) flagged rows that the ordinary moves took are not re-drawn;
(6) shortest edge case: dist 0 to an unflagged duplicate -> the ball collapses to the duplicates. usage: python test_ball_property.py"""
import types, numpy as np, torch
from simlib import bd17
torch.set_num_threads(1)
P = bd17.ParticleBirthDeath
def stub(N, seed=0):
    S = types.SimpleNamespace(Q=.05, N=N, counters=dict(iso_evals=0, iso_flagged=0, iso_acted=0, iso_skipped=0, iso_moves=0), iso_log=[], last={}, stream=torch.Generator().manual_seed(seed)); return S
bad = 0
def check(name, ok, detail=''):
    global bad; bad += (not ok); print(('[ok]   ' if ok else '[FAIL] ') + name + (' | ' + detail if detail else ''))
g = torch.Generator().manual_seed(0)
N = 4000
z = torch.cat([torch.randn(N - 60, 2, generator=g) * .1 + torch.tensor([[float(i % 4), float(i // 4 % 4)]]) for i in range(1)] + [torch.rand(60, 2, generator=g) * 6 - 1])
flagged = torch.zeros(N, dtype=torch.bool); flagged[N - 60:] = True
tstub = types.SimpleNamespace(completed_steps=0, prior=types.SimpleNamespace(z=z))
child = torch.tensor([N - 3, N - 2, 5, 6])          # two flagged rows and two unflagged rows taken by the ordinary moves
dead, par = P._isolation_pick(stub(N), tstub, flagged, child)
keep = ~flagged; keep[child] = False
check('flagged rows that the ordinary moves took are not re-drawn', not bool(torch.isin(dead, child).any()), f'{len(dead)} re-drawn of {int(flagged.sum())} flagged')
check('parents are unflagged and not ordinary-move children', bool(keep[par].all()))
d_all = torch.cdist(z[dead], z[keep.nonzero().flatten()]); dmin = d_all.min(1).values
d_par = (z[dead] - z[par]).norm(dim=1)
check('every parent is within 2x the distance of the nearest unflagged row', bool((d_par <= 2 * dmin + 1e-6).all()), f'max d_par/dmin {float((d_par / dmin.clamp_min(1e-12)).max()):.3f}')
# uniformity over the ball for one flagged row
from collections import Counter
sizes = torch.stack([((torch.cdist(z[r:r + 1], z[(~torch.nn.functional.one_hot(torch.tensor(r), N).bool()).nonzero().flatten()])[0]) <= 2 * torch.cdist(z[r:r + 1], z[(~torch.nn.functional.one_hot(torch.tensor(r), N).bool()).nonzero().flatten()])[0].min()).sum() for r in dead[:10].tolist()])
row = dead[int(sizes.argmax()):int(sizes.argmax()) + 1]; fl1 = torch.zeros(N, dtype=torch.bool); fl1[row] = True
kall = (~fl1).nonzero().flatten(); dist = torch.cdist(z[row], z[kall])[0]; ball_idx = kall[dist <= 2 * dist.min()]
S2 = stub(N, 5); cnt = Counter()
for t in range(4000):
    _, p1 = P._isolation_pick(S2, tstub, fl1, child[:0]); cnt[int(p1[0])] += 1
inside = set(ball_idx.tolist()); counts = np.array([cnt.get(i, 0) for i in ball_idx.tolist()], dtype=float)
exp = counts.sum() / len(ball_idx); chi = float(((counts - exp) ** 2 / exp).sum()); dof = len(ball_idx) - 1
check(f'parent draw is uniform over the ball ({len(ball_idx)} rows): chi2 {chi:.1f} on {dof} dof; draws outside the ball {sum(v for i, v in cnt.items() if i not in inside)}',
      abs(chi - dof) < 5 * (2 * dof) ** .5 and set(cnt) <= inside)
# guard
S3 = stub(N); d0, p0 = P._isolation_pick(S3, tstub, torch.zeros(N, dtype=torch.bool), child[:0]); check('guard: nothing flagged -> nothing done', len(d0) == 0 and S3.counters['iso_acted'] == 0)
big = torch.zeros(N, dtype=torch.bool); big[:int(.05 * N) + 1] = True; d1, p1 = P._isolation_pick(stub(N), tstub, big, child[:0]); check('guard: flagged > Q N -> skipped', len(d1) == 0)
ok = torch.zeros(N, dtype=torch.bool); ok[:int(.05 * N)] = True; d2, p2 = P._isolation_pick(stub(N), tstub, ok, child[:0]); check('guard: flagged == Q N acts (boundary is inclusive)', len(d2) == int(.05 * N))
# duplicate at distance 0
z2 = z.clone(); z2[N - 1] = z2[0]; t2 = types.SimpleNamespace(completed_steps=0, prior=types.SimpleNamespace(z=z2)); f2 = torch.zeros(N, dtype=torch.bool); f2[N - 1] = True
d3, p3 = P._isolation_pick(stub(N), t2, f2, child[:0]); check('duplicate of an unflagged row: the ball collapses onto the duplicates (parent at distance 0)', float((z2[p3[0]] - z2[N - 1]).norm()) < 1e-6)
print('ALLOK' if not bad else f'FAILED {bad}')

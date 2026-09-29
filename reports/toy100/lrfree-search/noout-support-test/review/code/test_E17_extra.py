"""Additional tests for the isolation mechanism (review_E17_code): written to kill mutants that tests/test_E15.py lets survive.
usage: python test_E17_extra.py <pkg root>      (CPU, 2 threads, about 1 minute)
A. differential test of `_isolated` against an independent brute-force float64 implementation of the docstring's specification (exact mask equality);
B. `_isolation_pick` on crafted tables: ball rule, child exclusion, guard boundary, chunking, live-table use, private stream;
C. `maybe_apply` / trainer coupling: flagged set = reference on the critic's features, stale-site resets with the iso sites, tester rebase and evidence reset receive
   the iso rows; flag on without isolated rows == flag off bit for bit."""
import sys, json, math, hashlib, torch, torch.nn as nn
import os; torch.set_num_threads(int(os.environ.get("NT", 2)))
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make, OVERRIDES
PKG = sys.argv[1]
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += (not ok); print(('[ok]   ' if ok else '[FAIL] ') + name + (' | ' + detail if detail else ''), flush=True)
def fresh():
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]
BASE = dict(birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none')
def trainer(n, flag=True):
    fresh(); build, real, digest = make(PKG, **BASE, num_particles=n, **({'birth_death_isolation': True} if flag else {})); t = build(); return t, t.birth_death, real, digest

# ============================================================== A. `_isolated` against an independent reference
def ref_scores(q, R, k):
    """(scores of q, scores of R2) of the docstring's statistic, brute force in float64 (no shortlist, counts by broadcasting)."""
    q, R = q.double(), R.double()
    R1, R2 = R[0::2], R[1::2]
    dm = lambda A, B: torch.cdist(A, B, compute_mode='donot_use_mm_for_euclid_dist')
    D11 = dm(R1, R1); D11.fill_diagonal_(float('inf'))
    rho = D11.sort(dim=1).values[:, k - 1]                                   # leave-one-out k-th neighbour radius
    pos = rho[rho > 0]
    floor = (pos.sort().values[(len(pos) - 1) // 2].item() * 1e-3) if len(pos) else 1.
    def score(U):
        vals, ix = dm(U, R1).sort(dim=1)
        return vals[:, k - 1] / rho[ix[:, :k]].sort(dim=1).values[:, (k - 1) // 2].clamp_min(floor)
    return score(q), score(R2)
def ref_isolated(q, R, k, Q=.05):
    s, null = ref_scores(q, R, k)
    p = (1. + (null[None, :] >= s[:, None]).sum(1).double()) / (1. + len(null))
    m = len(p); ps = p.sort().values
    ok = (ps <= torch.arange(1, m + 1, dtype=torch.float64) * Q / m).nonzero()
    return (p <= ps[int(ok[-1])]) if len(ok) else torch.zeros(m, dtype=torch.bool)

t, bd, real, digest = trainer(2000)
MEANS = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 2.5, 1.]], dtype=torch.float64)
SDS = torch.tensor([.05, .2, .2], dtype=torch.float64)
def mixture(n, g, masses=(.7, .28, .02)):
    c = torch.multinomial(torch.tensor(masses), n, replacement=True, generator=g)
    return MEANS[c] + SDS[c][:, None] * torch.randn(n, 3, generator=g, dtype=torch.float64), c
def graded_queries(g, R):
    """legit draws + strays at graded distances (p-values spread over the whole BH region) + exact copies of the most extreme R2 rows (ties at the cut)."""
    legit, _ = mixture(800, g)
    comp = torch.randint(0, 3, (900,), generator=g); ang = torch.randn(900, 3, generator=g, dtype=torch.float64); ang = ang / ang.norm(dim=1, keepdim=True)
    dist = torch.linspace(.15, 2.2, 900, dtype=torch.float64)
    strays = MEANS[comp] + dist[:, None] * ang
    R2 = R[1::2]
    ties = R2[(torch.cdist(R2, R2).topk(6, dim=1, largest=False).values[:, -1]).topk(12).indices].repeat(4, 1)      # copies of the 12 most isolated R2 rows
    return torch.cat([legit, strays, ties, R[0::2][:40]])
def compare(name, R, q, k, min_flag=10):
    got = bd._isolated(q, R, k, torch.float32); ref = ref_isolated(q, R, k, bd.Q)
    diff = int((got != ref).sum())
    check(f'A {name}: mask == independent reference ({int(got.sum())} flagged of {len(q)}, {diff} rows differ)', diff == 0 and int(ref.sum()) >= min_flag)
for seed in (1, 2):
    g = torch.Generator().manual_seed(seed)
    R, _ = mixture(2400, g); q = graded_queries(g, R)
    for k in (9, 10):
        compare(f'seed {seed} k={k} |R|=2400', R, q, k)
g = torch.Generator().manual_seed(5); R, comp = mixture(2401, g); q = graded_queries(g, R)
compare('odd reservoir |R|=2401 (R1 has one row more), k=9', R, q, 9)
compare('reservoir sorted by component, k=9', R[comp.argsort()], q, 9)        # the loader's order must not matter
# BH boundary: rows tied at the smallest possible p-value 1/(1+|R2|) are flagged iff there are at least m / (Q (1+|R2|)) of them: |R| = 6000 (|R2| = 3000), m = 2400 queries -> 16 rows
# (16 x .05 / 2400 = 3.3333e-4 >= 1/3001 = 3.3322e-4 > 15 x .05 / 2400); the other queries are copies of the 2384 least extreme R2 rows (p >= .2, never flagged)
g = torch.Generator().manual_seed(21); Rb, _ = mixture(6000, g)
_, null_b = ref_scores(Rb[:2], Rb, 10)
low_rows = Rb[1::2][null_b.argsort()[:2385]]
for f, expect in ((16, 16), (15, 0)):
    far = MEANS[0] + 50. * torch.nn.functional.normalize(torch.randn(f, 3, generator=g, dtype=torch.float64), dim=1)
    qb = torch.cat([far, low_rows[:2400 - f]])
    got = bd._isolated(qb, Rb, 10, torch.float32); ref = ref_isolated(qb, Rb, 10, bd.Q)
    check(f'A BH boundary: {f} far strays tied at p = 1/3001 among m = 2400 rows: {int(got.sum())} flagged (expected {expect}; reference {int(ref.sum())})', int(got.sum()) == expect and bool((got == ref).all()) and (expect == 0 or bool(got[:f].all())))
# scale invariance (the score is a ratio, the floor is relative to the data): the same mask at scale 1e-6 and 1e6
g = torch.Generator().manual_seed(8); R, _ = mixture(2400, g); q = graded_queries(g, R)
base = bd._isolated(q, R, 9, torch.float32)
for c in (1e-6, 1e6):
    got = bd._isolated(q * c, R * c, 9, torch.float32)
    check(f'A scale invariance: mask at scale {c:g} == mask at scale 1 ({int((got != base).sum())} rows differ of {int(base.sum())} flagged) and == reference', int((got != base).sum()) <= 1 and int(base.sum()) > 300 and int((got != ref_isolated(q, R, 9, bd.Q)).sum()) <= 1)
# dtype contract: the shortlist may be float32, the exact recompute (the inputs of _knn) must be float64
bdm = sys.modules[type(bd).__module__]; seen = []; orig_knn = bdm._knn
def spy_knn(query, points, k, *a, **kw): seen.append((query.dtype, points.dtype)); return orig_knn(query, points, k, *a, **kw)
bdm._knn = spy_knn
try: bd._isolated(q, R, 9, torch.float32)
finally: bdm._knn = orig_knn
check('A dtype contract: every _knn call of the support test gets float64 features (%d calls: %s)' % (len(seen), sorted(set(map(str, seen)))), len(seen) == 3 and all(d == (torch.float64, torch.float64) for d in seen))
# the floor: 30% exact copies of one point (radius 0), 50% a very dense cluster (sd .01), 20% a sparse one (sd 1): the median of the positive radii is 7e-3, their mean 0.26.  Rows displaced by
# 1.2e-4 from the copied point have k neighbours of radius 0: the local scale is the floor (1e-3 x the median = 7e-6): score 17, far beyond the null (max 2.2): flagged.  With the mean
# instead of the median the floor is 36x larger, the score 0.47: not flagged.
g = torch.Generator().manual_seed(31)
Rf_ = torch.cat([torch.zeros(720, 3, dtype=torch.float64), torch.tensor([5., 0, 0], dtype=torch.float64) + .01 * torch.randn(1200, 3, generator=g, dtype=torch.float64),
                 torch.tensor([0., 8., 0.], dtype=torch.float64) + 1.0 * torch.randn(480, 3, generator=g, dtype=torch.float64)])
Rf_ = Rf_[torch.randperm(2400, generator=g)]
disp = 1.2e-4 * torch.nn.functional.normalize(torch.randn(100, 3, generator=g, dtype=torch.float64), dim=1)
_, null_f = ref_scores(Rf_[:2], Rf_, 9)
qf_ = torch.cat([disp, Rf_[1::2][null_f.argsort()[:900]]])
got = bd._isolated(qf_, Rf_, 9, torch.float32); ref = ref_isolated(qf_, Rf_, 9, bd.Q)
check(f'A floor (duplicate cluster + two scales): the 100 rows displaced by 1.2e-4 are flagged ({int(got[:100].sum())} of 100), nothing else ({int(got[100:].sum())}); == reference ({int((got != ref).sum())} differ)',
      int(got[:100].sum()) == 100 and int(got[100:].sum()) == 0 and bool((got == ref).all()))
# duplicates and zero radii: 4 distinct points x 500 copies (a memorised finite data set); table rows either at those points or displaced from them
g = torch.Generator().manual_seed(6)
pts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
Rd = pts[torch.randint(0, 4, (2000,), generator=g)]
qd = torch.cat([pts[torch.randint(0, 4, (1800,), generator=g)], pts[torch.randint(0, 4, (200,), generator=g)] + .3 * torch.randn(200, 3, generator=g, dtype=torch.float64)])
got = bd._isolated(qd, Rd, 9, torch.float32); ref = ref_isolated(qd, Rd, 9, bd.Q)
check(f'A finite data set (4 points x 500 copies): copies not flagged ({int(got[:1800].sum())}), displaced rows flagged ({int(got[1800:].sum())} of 200); == reference ({int((got != ref).sum())} differ)',
      int(got[:1800].sum()) == 0 and int(got[1800:].sum()) >= 190 and bool((got == ref).all()))

# ============================================================== B. `_isolation_pick` on crafted tables
t, bd, real, digest = trainer(20000)
N = 20000; gg = torch.Generator().manual_seed(11)
CENTRES = torch.tensor([[0., 0.], [30., 0.], [0., 30.], [-30., 0.], [0., -30.]])
RADII = torch.tensor([1.0, 1.4, 1.8, 2.2, 3.0, 5.0])
Z = torch.zeros(N, 2); role = torch.full((N,), -1); grp = torch.full((N,), -1); rad = torch.zeros(N)
row = 0
for c in range(5):                                        # 200 flagged rows exactly at each centre
    Z[row:row + 200] = CENTRES[c]; role[row:row + 200] = 1; grp[row:row + 200] = c; row += 200
near = {}
for c in range(5):                                        # six unflagged rows at known distances around each centre
    for j, r in enumerate(RADII):
        a = float(torch.rand(1, generator=gg)) * 2 * math.pi
        Z[row] = CENTRES[c] + r * torch.tensor([math.cos(a), math.sin(a)]); role[row] = 0; grp[row] = c; rad[row] = r; near[(c, j)] = row; row += 1
far = N - row; a = torch.rand(far, generator=gg) * 2 * math.pi; r = 60 + 40 * torch.rand(far, generator=gg)
Z[row:] = torch.stack([r * a.cos(), r * a.sin()], 1); role[row:] = 0; grp[row:] = -2
flagged = role == 1
NOCHILD = torch.zeros(0, dtype=torch.long)
def prep():
    with torch.no_grad(): t.prior.z.copy_(Z); t.ema_prior.z.copy_(torch.randn(N, 2) * 40)      # the EMA table is junk: parents must be chosen in the LIVE table
    bd.stream.manual_seed(123)
    for k in [k for k in bd.counters if k.startswith('iso')]: bd.counters[k] = 0
sizes = []
orig_cdist = torch.cdist
def spy_cdist(a, b, *args, **kw):
    sizes.append((a.shape[0], b.shape[0])); return orig_cdist(a, b, *args, **kw)
prep(); torch.cdist = spy_cdist
try: dead, parent = bd._isolation_pick(t, flagged, NOCHILD)
finally: torch.cdist = orig_cdist
inball = (rad > 0) & (rad <= 1.9)                                               # the 1.0, 1.4, 1.8 rows
ok_len = len(dead) == 1000 and len(parent) == 1000 and bool(torch.equal(dead, flagged.nonzero().flatten()))
check('B1 n_flag == Q N (1000 of 20000) acts; every flagged row gets one parent (dead == flagged rows, aligned across chunks)', ok_len, f'{len(dead)} dead, {len(parent)} parents')
in_ball = torch.tensor([bool(inball[p]) and int(grp[p]) == int(grp[d]) for d, p in zip(dead.tolist(), parent.tolist())]) if ok_len else torch.zeros(1, dtype=torch.bool)
check('B1 parents are unflagged rows of the row\'s own ball (radius 2 x nearest = 2.0: the 1.0, 1.4, 1.8 rows): %d of %d' % (int(in_ball.sum()), len(in_ball)), bool(in_ball.all()) and ok_len)
freq = torch.stack([torch.tensor([float(((parent == near[(c, j)]) & (grp[dead] == c)).sum()) / 200 for j in range(3)]) for c in range(5)]) if ok_len else torch.zeros(5, 3)
check('B1 the ball is sampled uniformly: all three ball rows used, share in [.22, .45] (min %.3f max %.3f)' % (float(freq.min()), float(freq.max())), float(freq.min()) >= .22 and float(freq.max()) <= .45)
biggest = max([a * b for a, b in sizes] or [0])
check('B1 chunking: no [chunk, keep] block above 2^24 entries (largest %d, %d blocks)' % (biggest, len(sizes)), biggest <= (1 << 24) and len(sizes) >= 2)
check('B1 counters', bd.counters['iso_acted'] == 1 and bd.counters['iso_moves'] == 1000 and bd.counters['iso_flagged'] == 1000, str({k: v for k, v in bd.counters.items() if k.startswith('iso')}))
# B2 the stream: private (global RNG irrelevant), reproducible, advanced only when acting
prep(); s0 = bd.stream.get_state(); torch.manual_seed(1); d1, p1 = bd._isolation_pick(t, flagged, NOCHILD); s1 = bd.stream.get_state()
prep(); torch.manual_seed(999); torch.rand(17); d2, p2 = bd._isolation_pick(t, flagged, NOCHILD)
check('B2 the parent draw comes from the private stream (a different global RNG state gives identical parents; the stream advanced)', torch.equal(p1, p2) and not torch.equal(s0, s1))
prep(); s0 = bd.stream.get_state()
bd._isolation_pick(t, torch.zeros(N, dtype=torch.bool), NOCHILD)                              # nothing flagged: no draw
big = flagged.clone(); big[(~flagged).nonzero().flatten()[:1]] = True                       # 1001 flagged: guard, no draw
res_big = bd._isolation_pick(t, big, NOCHILD)
check('B2 guard: 0 flagged and 1001 flagged (> Q N = 1000) do nothing and draw nothing (stream state unchanged, iso_skipped 1, iso_acted 0)',
      torch.equal(s0, bd.stream.get_state()) and len(res_big[0]) == 0 and bd.counters['iso_skipped'] == 1 and bd.counters['iso_acted'] == 0, str({k: v for k, v in bd.counters.items() if k.startswith('iso')}))
# B3 exclusion of rows the ordinary birth-death just moved
prep()
child = torch.cat([flagged.nonzero().flatten()[:50], torch.tensor([near[(1, 0)]])])      # 50 flagged rows of centre 0 + the nearest unflagged row (r = 1.0) of centre 1
dead, parent = bd._isolation_pick(t, flagged, child)
check('B3 rows just moved by the ordinary birth-death are not re-drawn (dead has %d rows: 950 expected) and are not parents' % len(dead),
      len(dead) == 950 and not bool(torch.isin(dead, child).any()) and not bool(torch.isin(parent, child).any()))
c1 = (grp[dead] == 1); allowed = torch.tensor([near[(1, 1)], near[(1, 2)], near[(1, 3)]])
check('B3 with the r=1.0 row excluded the ball of centre 1 becomes 2 x 1.4 = 2.8: parents in {1.4, 1.8, 2.2} (%d of %d)' % (int(torch.isin(parent[c1], allowed).sum()), int(c1.sum())),
      int(c1.sum()) == 200 and bool(torch.isin(parent[c1], allowed).all()))
# B4 degenerate inputs: everything flagged (guard lifted), no unflagged row left, everything already moved
prep(); Q0 = bd.Q; bd.Q = 2.
try:
    r_all = bd._isolation_pick(t, torch.ones(N, dtype=torch.bool), NOCHILD); r_child = bd._isolation_pick(t, flagged, torch.arange(N)); r_keep0 = bd._isolation_pick(t, flagged, (~flagged).nonzero().flatten())
finally: bd.Q = Q0
check('B4 flagged set = whole table, all rows already moved, no unflagged row left: empty answers, no error', all(len(a) == 0 and len(b) == 0 for a, b in (r_all, r_child, r_keep0)))

# ============================================================== C. maybe_apply / trainer coupling on a 2000-row ring table
def ring_table(n_stray, seed=0, flag=True):
    t, bd, real, digest = trainer(2000, flag)
    for i in range(40): t.step(real(i))
    with torch.no_grad(): t.G.weight.copy_(torch.eye(2)); t.G.bias.zero_()
    gg = torch.Generator().manual_seed(seed)
    for key in [k for k in bd.counters if k.startswith('iso')]: bd.counters[key] = 0
    mode = torch.randint(0, 8, (2000,), generator=gg); ang = mode.float() * math.pi / 4
    z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(2000, 2, generator=gg)
    a = torch.rand(n_stray, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(n_stray, generator=gg)
    if n_stray: z[:n_stray] = torch.stack([r * a.cos(), r * a.sin()], 1)
    with torch.no_grad(): t.prior.z.copy_(z); t.ema_prior.z.copy_(z)
    bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0
    bd.observe_real(torch.cat([real(1000 + i) for i in range(32)])[:2000]); return t, bd, real, mode
# C1 the flagged set of a real evaluation == the reference on the critic's own features
t, bd, real, mode = ring_table(60)
with torch.no_grad():
    t.G.eval(); qraw = t.G(t.prior.z.detach()); t.G.train()
    qf, Rf = bd._features(t, qraw), bd._features(t, bd.reservoir); cen = Rf.mean(0, keepdim=True)
ref = ref_isolated(qf - cen, Rf - cen, bd.k, bd.Q)
last = bd.maybe_apply(t, .03)
mv = bd.moved_rows
ordinary = last['moves'] - last['iso_moves']
check('C1 flagged set of maybe_apply == reference on the critic\'s features (%d flagged; reference %d; ordinary moves %d; moved %s)' % (last['iso_flagged'], int(ref.sum()), ordinary, None if mv is None else len(mv)),
      last['iso_flagged'] == int(ref.sum()) and ordinary == 0 and mv is not None and torch.equal(mv.sort().values, ref.nonzero().flatten()))
# C2 stale resets: after the evaluation a row that was not moved is restarted (n = 0) exactly when its anchor ball holds a site (the old place or the new place of a moved row);
#    all bulk rows are in an excursion at this point (the table was just re-placed), so the set is fully determined; the oracle is recomputed in float64 from the pre-move table.
t, bd, real, mode = ring_table(60)
seen = []; orig_move = bd._move
def spy(trainer, child, parent): seen.append((child.clone(), parent.clone())); return orig_move(trainer, child, parent)
bd._move = spy
z0 = t.prior.z.detach().clone()
last = bd.maybe_apply(t, .03)
child, parent = seen[-1]; par = parent.unique()
sites = torch.cat([z0[child], z0[parent]]).double()
inside = (torch.cdist(bd.anchor.double(), sites, compute_mode='donot_use_mm_for_euclid_dist') <= bd.radius.double()[:, None]).any(1)
rest = torch.ones(2000, dtype=torch.bool); rest[child] = False
mis = int(((bd.n == 0) != inside)[rest].sum())
check('C2 stale reset == float64 oracle for the %d unmoved rows (%d rows differ; %d restarted; the %d parents restarted: %d); stale_resets counter %d' %
      (int(rest.sum()), mis, int((bd.n[rest] == 0).sum()), len(par), int((bd.n[par] == 0).sum()), bd.counters['stale_resets']),
      last['moves'] == last['iso_moves'] and mis <= 15 and len(par) >= 3 and bool((bd.n[par] == 0).all()))
# C6 a re-drawn row is a clone of its parent in every per-row structure: live row + jitter, EMA row + the same jitter, Adam/AMSGrad rows, A2 latent history
t, bd, real, mode = ring_table(60)
gh = torch.Generator().manual_seed(77)
with torch.no_grad():
    t.ema_prior.z.add_(.01 * torch.randn(2000, 2, generator=gh))                         # EMA rows differ from the live rows
    t.opt_g.latent_history.copy_(torch.randn(2000, 2, generator=gh))
    for key, v in t.opt_g.state[t.prior.z].items():
        if isinstance(v, torch.Tensor) and v.shape == t.prior.z.shape: v.copy_(torch.randn(2000, 2, generator=gh).abs())
pre = dict(z=t.prior.z.detach().clone(), ema=t.ema_prior.z.detach().clone(), hist=t.opt_g.latent_history.clone(), st={k: v.clone() for k, v in t.opt_g.state[t.prior.z].items() if isinstance(v, torch.Tensor) and v.shape == t.prior.z.shape})
seen = []; orig_move = bd._move
def spy(trainer, child, parent): seen.append((child.clone(), parent.clone())); return orig_move(trainer, child, parent)
bd._move = spy
bd.maybe_apply(t, .03)
child, parent = seen[-1]; rest = torch.ones(2000, dtype=torch.bool); rest[child] = False
jit = t.prior.z.detach()[child] - pre['z'][parent]
ok_ema = torch.allclose(t.ema_prior.z.detach()[child], pre['ema'][parent] + jit, atol=1e-6)
ok_hist = torch.equal(t.opt_g.latent_history[child], pre['hist'][parent]) and torch.equal(t.opt_g.latent_history[rest], pre['hist'][rest])
ok_st = all(torch.equal(t.opt_g.state[t.prior.z][k][child], v[parent]) and torch.equal(t.opt_g.state[t.prior.z][k][rest], v[rest]) for k, v in pre['st'].items())
check('C6 re-drawn rows are clones of their parents: EMA row + same jitter (%s), A2 latent history (%s), optimizer rows (%s, %d state tensors); unmoved rows untouched' % (ok_ema, ok_hist, ok_st, len(pre['st'])), ok_ema and ok_hist and ok_st and len(pre['st']) == 3)
# C7 moved rows restart their evidence whatever the stale test says (a NaN anchor radius blinds the stale test: `dist <= nan` is False): the explicit reset of S, W, n must still zero them
t, bd, real, mode = ring_table(60)
with torch.no_grad():
    z0 = t.prior.z.detach().clone(); bd.anchor[:60] = z0[:60]; bd.radius[:60] = float('nan'); bd.n[:60] = 3; bd.W[:60] = .1; bd.S[:60] = .1
last = bd.maybe_apply(t, .03); mv = bd.moved_rows
check('C7 moved rows have S = W = n = 0 even when the stale test cannot see them (NaN radius planted on the 60 strays: %s moved)' % (None if mv is None else len(mv)),
      mv is not None and len(mv) >= 55 and bool((bd.n[mv] == 0).all() and (bd.W[mv] == 0).all() and (bd.S[mv] == 0).all()))
# C8 the call site: maybe_apply hands `_isolated` the centred critic features of the clean table centres and of the real reservoir, the birth-death's own k and the float32 shortlist
t, bd, real, mode = ring_table(60)
with torch.no_grad():
    t.G.eval(); qraw = t.G(t.prior.z.detach()); t.G.train()
    qf, Rf = bd._features(t, qraw), bd._features(t, bd.reservoir); cen = Rf.mean(0, keepdim=True)
calls = []; orig_iso = bd._isolated
def spy_iso(q, R, k, fast): calls.append((q.clone(), R.clone(), k, fast)); return orig_iso(q, R, k, fast)
bd._isolated = spy_iso
bd.maybe_apply(t, .03)
ok_call = len(calls) == 1 and calls[0][2] == bd.k and calls[0][3] == torch.float32 and torch.allclose(calls[0][0], qf - cen, atol=1e-9) and torch.allclose(calls[0][1], Rf - cen, atol=1e-9)
check('C8 _isolated is called once with (centred features of G(z), centred features of the reservoir, k = %d, float32 shortlist)' % bd.k, ok_call, '' if not calls else 'k=%s fast=%s' % (calls[0][2], calls[0][3]))
# C3 the trainer hands the iso rows to the tester and to the row evidence
t, bd, real, mode = ring_table(60); tester = t._table_tester(); got = {}
orig_rebase, orig_reset = tester.rebase, t.row_evidence.reset
tester.rebase = lambda params, rows: (got.setdefault('rebase', []).append(rows.clone()), orig_rebase(params, rows))[1]
t.row_evidence.reset = lambda rows: (got.setdefault('reset', []).append(rows.clone()), orig_reset(rows))[1]
t.step(real(5000)); mv = bd.moved_rows
have = mv is not None and len(mv) >= 55
check('C3 trainer.step after isolation moves: tester.rebase and row_evidence.reset both received exactly the moved rows (%s)' % ('%d moved, rebase %s, reset %s' % (0 if mv is None else len(mv), [len(x) for x in got.get('rebase', [])], [len(x) for x in got.get('reset', [])])),
      have and len(got.get('rebase', [])) == 1 and len(got.get('reset', [])) == 1 and torch.equal(got['rebase'][0].sort().values, mv.sort().values) and torch.equal(got['reset'][0].sort().values, mv.sort().values))
# C4 flag on without isolated rows == flag off (bit for bit), through three evaluations
def digest_of(t):
    h = hashlib.sha256()
    for m in (t.G, t.D, t.prior, t.ema_G, t.ema_prior):
        for p in m.parameters(): h.update(p.detach().contiguous().numpy().tobytes())
    h.update(t.birth_death.stream.get_state().numpy().tobytes())
    return h.hexdigest()[:16]
def run_state(flag):
    t, bd, real, mode = ring_table(0, flag=flag)
    for i in range(70): t.step(real(6000 + i))
    return t, digest_of(t)
ton, don = run_state(True); toff, doff = run_state(False)
check('C4 flag on, isolation never acting (nothing flagged or the guard blocks: flagged %d, acted 0) == flag off, parameters and private stream, %d evaluations, %d ordinary moves (%s %s)' % (ton.birth_death.counters['iso_flagged'], ton.birth_death.counters['iso_evals'], ton.birth_death.counters['moves'], don, doff),
      don == doff and ton.birth_death.counters['iso_evals'] >= 2 and ton.birth_death.counters['iso_acted'] == 0 and ton.birth_death.counters['moves'] > 0)
# C5 features far from the origin (a critic whose last hidden layer carries a large constant): the statistic is translation invariant, only the float32 shortlist is not; the
#    trainer centres the features first, so the flagged set must equal the float64 reference on the very same features
t, bd, real, mode = ring_table(60)
with torch.no_grad(): t.D[2].bias.add_(1e4)
with torch.no_grad():
    t.G.eval(); qraw = t.G(t.prior.z.detach()); t.G.train()
    qf, Rf = bd._features(t, qraw), bd._features(t, bd.reservoir); cen = Rf.mean(0, keepdim=True)
ref = ref_isolated(qf - cen, Rf - cen, bd.k, bd.Q)
last = bd.maybe_apply(t, .03); mv = bd.moved_rows
check('C5 critic features offset by 1e4: flagged set == float64 reference (%d flagged, reference %d; moved %s; ordinary moves %d)' % (last['iso_flagged'], int(ref.sum()), None if mv is None else len(mv), last['moves'] - last['iso_moves']),
      int(ref.sum()) >= 55 and last['iso_flagged'] == int(ref.sum()) and mv is not None and torch.equal(mv.sort().values, ref.nonzero().flatten()))
print('ALLOK' if not bad else 'FAILED %d' % bad)

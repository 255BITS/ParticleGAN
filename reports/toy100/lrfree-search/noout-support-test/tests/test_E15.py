"""pkg-E15 CPU tests: `birth_death_isolation` (split-conformal support test of the table rows in the critic's feature space + re-draw of unsupported rows).
Flag off = pkg-E14 bit for bit; the recipe refuses the flag outside critic-space birth-death; the local real scale keeps a rare component from being flagged for
being sparse; a planted stray population is found and re-drawn with the moves' full bookkeeping; a table that is mostly unsupported (in transit) is left alone;
a tiny table is inert. usage: python test_E15.py"""
import sys, json, math, hashlib, torch, torch.nn as nn
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make, OVERRIDES
E14 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E14'; E15 = sys.argv[1] if len(sys.argv) > 1 else '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E15'   # the package under test (E15: uniform parents; E16: nearest unflagged rows)
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += (not ok); print(('[ok]   ' if ok else '[FAIL] ') + name + (' | ' + detail if detail else ''))
def fresh():
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]
BASE = dict(birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none')

# ---- 1. validation
fresh(); sys.path.insert(0, E15)
import particlegan as package
ov = json.load(open(OVERRIDES)); ov.update(num_particles=200, z_dim=2, batch_size=64)
def refused(**kw):
    try: package.get_recipe(**{**ov, **kw}); return False
    except ValueError: return True
check('recipe: isolation with data-space birth-death is refused', refused(birth_death_space='data', birth_death_isolation=True))
check('recipe: isolation must be a boolean', refused(birth_death_space='critic', birth_death_isolation=1))
check('recipe: isolation without birth-death is refused', refused(particle_birth_death=False, birth_death_space='critic', birth_death_isolation=True))
check('recipe: isolation with critic-space birth-death is accepted', not refused(birth_death_space='critic', birth_death_isolation=True))

# ---- 2. flag off = pkg-E14 exactly; flag on in a 200-row table is inert (the test cannot reach BH resolution) and therefore also identical
def run(pkg, steps, **extra):
    fresh(); build, real, digest = make(pkg, **BASE, **extra); t = build()
    for i in range(steps): t.step(real(i))
    return t, digest(t)
t14, d14 = run(E14, 80)
t15, d15 = run(E15, 80)
t15on, d15on = run(E15, 80, birth_death_isolation=True)
check('flag off: E15 trajectory == E14 trajectory (80 steps, critic-space birth-death, gate, anchor)', d14 == d15, f'{d14} {d15}')
c = t15on.birth_death.counters
check('flag on, 200-row table: evaluations were scored, nothing flagged or moved -> identical to flag off', d15on == d15 and c['iso_evals'] > 0 and c['iso_moves'] == 0,
      f"evals {c['iso_evals']} flagged {c['iso_flagged']} moves {c['iso_moves']}")

# ---- 3. the statistic on synthetic features (called on a real ParticleBirthDeath object; k as in the trainer)
fresh(); build, real, digest = make(E15, **BASE, birth_death_isolation=True, num_particles=2000); t = build(); bd = t.birth_death
K = 10
means4 = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
def mix(n, g, masses, sd):
    c = torch.multinomial(torch.tensor(masses), n, replacement=True, generator=g)
    return means4[c] + torch.tensor(sd)[c][:, None] * torch.randn(n, 2, generator=g), c
def raw_flags(q, R, k, level=.05):     # the unnormalised score, for contrast: distance to the k-th real neighbour only
    R1, R2 = R[0::2], R[1::2]
    a = lambda u: torch.cdist(u, R1).topk(k, dim=1, largest=False).values[:, -1]
    null = a(R2).sort().values; s = a(q)
    p = (1. + (len(null) - torch.searchsorted(null, s)).double()) / (1. + len(null)); ps = p.sort().values
    ok = (ps <= torch.arange(1, len(p) + 1, dtype=p.dtype) * level / len(p)).nonzero()
    return (p <= ps[int(ok[-1])]) if len(ok) else torch.zeros(len(p), dtype=torch.bool)
g = torch.Generator().manual_seed(3)
R, _ = mix(20000, g, (.70, .29, .008, .002), (.05, .18, .18, .18))
fake, comp = mix(19720, g, (.70, .29, .008, .002), (.05, .18, .18, .18))
lo, hi = R.min(0).values - .3, R.max(0).values + .3
stray = lo + (hi - lo) * torch.rand(280, 2, generator=g)
q = torch.cat([fake, stray]); label = torch.cat([comp, torch.full((280,), -1)])
d_near = torch.cdist(stray, R).min(1).values           # planted points that fall inside real support are not strays: score recall on the far ones
far = d_near > 0.3
flag = bd._isolated(q.double(), R.double(), K, torch.float32); rawf = raw_flags(q.double(), R.double(), K)
rare2, rare3 = label == 2, label == 3
check('statistic: far planted strays found (recall %.3f), legitimate fakes flagged %.4f' % (float(flag[19720:][far].float().mean()), float(flag[:19720].float().mean())),
      float(flag[19720:][far].float().mean()) >= .85 and float(flag[:19720].float().mean()) <= .003)
check('statistic: the .8%% and .2%% components are not flagged for being sparse (%.4f, %.4f); the unnormalised score flags %.3f of the .2%% component'
      % (float(flag[rare2].float().mean()), float(flag[rare3].float().mean()), float(rawf[rare3].float().mean())),
      float(flag[rare2].float().mean()) <= .02 and float(flag[rare3].float().mean()) <= .02 and float(rawf[rare3].float().mean()) > .10)
# duplicates in the reservoir (a finite data set): no NaN, the test stays defined
Rd = torch.cat([R[:1000], R[:1000]]).double(); qd = torch.cat([fake[:500], stray[:100]]).double()
fd = bd._isolated(qd, Rd, K, torch.float32)
check('statistic: a reservoir with exact duplicates gives a defined answer (%d of %d flagged)' % (int(fd.sum()), len(fd)), fd.dtype == torch.bool and len(fd) == len(qd))

# ---- 4. moves on a controlled 2000-row table (8 modes on a circle, sd .03, as in the fixture)
def ring_table(n_stray, seed=0):
    fresh(); build, real, digest = make(E15, **BASE, birth_death_isolation=True, num_particles=2000); t = build()
    for i in range(40): t.step(real(i))                 # a few real updates so that optimizer state exists for the copy check
    with torch.no_grad(): t.G.weight.copy_(torch.eye(2)); t.G.bias.zero_()      # the warm-up moved G a little: the table is placed in data coordinates
    bd = t.birth_death; gg = torch.Generator().manual_seed(seed)
    for key in [k for k in bd.counters if k.startswith('iso')]: bd.counters[key] = 0         # the warm-up evaluation (table in transit: everything flagged, skipped) is not under test
    mode = torch.randint(0, 8, (2000,), generator=gg); ang = mode.float() * math.pi / 4
    z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(2000, 2, generator=gg)
    a = torch.rand(n_stray, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(n_stray, generator=gg)       # strays: an inner ring, >= 30 sigma from every mode
    z[:n_stray] = torch.stack([r * a.cos(), r * a.sin()], 1)
    with torch.no_grad(): t.prior.z.copy_(z); t.ema_prior.z.copy_(z)
    reals = torch.cat([real(1000 + i) for i in range(32)])[:2000]
    bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0
    bd.observe_real(reals); return t, bd, real
t, bd, real = ring_table(60)
seen = {}
orig = bd._move
def spy(trainer, child, parent):
    seen.setdefault('calls', []).append((child.clone(), parent.clone(), t.prior.z.detach().clone(), t.ema_prior.z.detach().clone()))
    return orig(trainer, child, parent)
bd._move = spy
state_before = {k: v.clone() for k, v in t.opt_g.state.get(t.prior.z, {}).items() if isinstance(v, torch.Tensor) and v.shape == t.prior.z.shape}
last = bd.maybe_apply(t, .03)
strays = torch.arange(60)
moved = bd.moved_rows
found = int(torch.isin(strays, moved).sum()) if moved is not None else 0
check('moves: the 60 planted strays are found and re-drawn (%d of 60), extra bulk rows moved: %d; last: flagged %s moves %s' % (found, (len(moved) if moved is not None else 0) - found, last.get('iso_flagged'), last.get('moves')),
      found >= 55 and (len(moved) - found) <= 3 and last.get('moves') == len(moved))
child, parent, z_pre, ema_pre = seen['calls'][-1]
iso = torch.isin(child, strays)
check('moves: parents are rows that are neither flagged nor moved', not bool(torch.isin(parent, moved).any()) and not bool(torch.isin(parent, strays).any()))
zp, ep = t.prior.z.detach(), t.ema_prior.z.detach()
check('moves: the new row is the parent plus one jitter, the averaged row the same (max |dz - dema| %.2e)' % float(((zp[child] - z_pre[parent]) - (ep[child] - ema_pre[parent])).abs().max()),
      float(((zp[child] - z_pre[parent]) - (ep[child] - ema_pre[parent])).abs().max()) < 1e-6)
st = t.opt_g.state.get(t.prior.z, {})
same = all(bool(torch.equal(v[child], state_before[k][parent])) for k, v in st.items() if k in state_before)
check('moves: the optimizer rows of the new rows are copies of the parents\' (%d state tensors)' % len(state_before), len(state_before) > 0 and same)
check('moves: the re-drawn strays now sit at a mode (distance to the nearest of the 8 centres, max %.3f)' % float((zp[child[iso]].norm(dim=1) - 3).abs().max()),
      float((zp[child[iso]].norm(dim=1) - 3).abs().max()) < .2)
check('moves: evidence of the moved rows restarted (S, W, n = 0)', bool((bd.S[moved] == 0).all() and (bd.W[moved] == 0).all() and (bd.n[moved] == 0).all()))
check('moves: counters', bd.counters['iso_moves'] == last['iso_moves'] and bd.counters['iso_acted'] == 1 and bd.counters['iso_evals'] == 1, str({k: v for k, v in bd.counters.items() if k.startswith('iso')}))
# a fresh evaluation right after: nothing left to flag, nothing moves
bd.observe_real(torch.cat([real(2000 + i) for i in range(32)])[:2000]); last2 = bd.maybe_apply(t, .03)
check('moves: the next evaluation finds (almost) nothing (flagged %s)' % last2.get('iso_flagged'), last2.get('iso_flagged', 99) <= 3)
# guard: a table that is mostly unsupported is not resampled
t, bd, real = ring_table(300)
z0 = t.prior.z.detach().clone(); last = bd.maybe_apply(t, .03)
check('guard: 300 of 2000 rows unsupported (15%% > Q): flagged %s, nothing moved, skipped counted' % last.get('iso_flagged'),
      last.get('iso_flagged', 0) >= 250 and last.get('iso_moves') == 0 and torch.equal(z0, t.prior.z.detach()) and bd.counters['iso_skipped'] == 1 and bd.counters['iso_moves'] == 0)
# dry run
t, bd, real = ring_table(60); bd.dry_run = True; z0 = t.prior.z.detach().clone(); bd.maybe_apply(t, .03)
check('dry run: nothing moves', torch.equal(z0, t.prior.z.detach()) and bd.counters['iso_moves'] == 0)

# ---- 5. checkpoints and determinism
t, bd, real = ring_table(60); bd.maybe_apply(t, .03); sd = t.state_dict()
fresh(); build, real2, digest = make(E15, **BASE, birth_death_isolation=True, num_particles=2000); t2 = build(); t2.load_state_dict(sd)
check('checkpoint: the isolation counters survive a round trip', {k: v for k, v in t2.birth_death.counters.items() if k.startswith('iso')} == {k: v for k, v in bd.counters.items() if k.startswith('iso')})
ta, ba, _ = ring_table(60); ba.maybe_apply(ta, .03); tb, bb, _ = ring_table(60); bb.maybe_apply(tb, .03)
check('determinism: two identical evaluations re-draw identically', torch.equal(ta.prior.z, tb.prior.z) and torch.equal(ba.moved_rows, bb.moved_rows))
# the trainer consumes the moves: tester re-anchor / evidence reset run without error when only isolation moves happened
t, bd, real = ring_table(60); t.birth_death.observe_real(real(3000))
try:
    for i in range(30): t.step(real(4000 + i))
    ok = True
except Exception as e:
    ok = False; print(repr(e))
check('trainer: steps run after isolation moves (tester re-anchor, evidence reset)', ok)
# ---- 6. CUDA under the harness's deterministic-algorithms setting (the first launch of E15a died on torch.median-with-indices: not deterministic on CUDA)
if torch.cuda.is_available():
    torch.use_deterministic_algorithms(True)
    try:
        import os; os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        fresh(); build, real, digest = make(E15, **BASE, birth_death_isolation=True, num_particles=2000); t = build(); bd = t.birth_death
        cpu_flags = bd._isolated(q.double(), R.double(), K, torch.float32)
        dev_flags = bd._isolated(q.double().cuda(), R.double().cuda(), K, torch.float32).cpu()
        check('CUDA + deterministic algorithms: the support test runs and agrees with the CPU flags (%d vs %d flagged, %d differ)' % (int(cpu_flags.sum()), int(dev_flags.sum()), int((cpu_flags != dev_flags).sum())),
              int((cpu_flags != dev_flags).sum()) <= 3)
    except Exception as e:
        check('CUDA + deterministic algorithms: the support test runs', False, repr(e)[:200])
    finally:
        torch.use_deterministic_algorithms(False)
else:
    print('[skip] no CUDA device')
print('ALLOK' if not bad else 'FAILED %d' % bad)

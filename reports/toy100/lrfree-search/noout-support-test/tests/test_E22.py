"""pkg-E22 (= pkg-E19 + the duplicate guard) CPU tests: (a) identity: with duplicate-free reservoirs E22 follows E19's trajectory bit for bit (training run with the support test acting and the
feature scale on); (b) the guard: a reservoir with more than Q exact copies stops the test (nothing flagged, counted), a duplicate-free one does not. usage: python test_E22.py [pkg-E22]"""
import sys, json, math, hashlib, torch
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make, OVERRIDES
E19 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E19'; E22 = sys.argv[1] if len(sys.argv) > 1 else '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E22'
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += (not ok); print(('[ok]   ' if ok else '[FAIL] ') + name + (' | ' + detail if detail else ''), flush=True)
def fresh():
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]
BASE = dict(birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none', birth_death_isolation=True, birth_death_feature_scale='std')
def run(pkg, steps=60, n=2000):
    """warm-up, 60 planted strays on an inner ring, one support-test evaluation that acts, then more training steps"""
    fresh(); build, real, digest = make(pkg, **BASE, num_particles=n); t = build()
    for i in range(40): t.step(real(i))
    with torch.no_grad(): t.G.weight.copy_(torch.eye(2)); t.G.bias.zero_()
    bd = t.birth_death; gg = torch.Generator().manual_seed(0)
    mode = torch.randint(0, 8, (n,), generator=gg); ang = mode.float() * math.pi / 4
    z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(n, 2, generator=gg)
    a = torch.rand(60, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(60, generator=gg); z[:60] = torch.stack([r * a.cos(), r * a.sin()], 1)
    with torch.no_grad(): t.prior.z.copy_(z); t.ema_prior.z.copy_(z)
    bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0; bd.observe_real(torch.cat([real(1000 + i) for i in range(32)])[:n])
    bd.maybe_apply(t, .03)
    for i in range(steps): t.step(real(3000 + i))
    return digest(t), {k: v for k, v in bd.counters.items() if k != 'iso_dup_skips'}, bd
d19, c19, _ = run(E19); d22, c22, bd22 = run(E22)
check('identity: warm-up, 60 planted strays re-drawn by the support test, 60 more steps: parameters identical (%s %s)' % (d19, d22), d19 == d22)
check('identity: birth-death counters identical, the support test acted (iso moves %d) and the guard never fired (dup skips %d)' % (c22['iso_moves'], bd22.counters['iso_dup_skips']), c19 == c22 and c22['iso_moves'] >= 55 and bd22.counters['iso_dup_skips'] == 0)
# (b) the guard on a finite pool
def pool_run(dup):
    fresh(); build, real, digest = make(E22, **BASE, num_particles=2000); t = build(); bd = t.birth_death
    for i in range(40): t.step(real(i))
    for key in [k for k in bd.counters if k.startswith('iso')]: bd.counters[key] = 0
    reals = torch.cat([real(1000 + i) for i in range(32)])[:2000]
    if dup is not None: reals = reals[torch.randint(0, dup, (2000,), generator=torch.Generator().manual_seed(1))]
    bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0; bd.observe_real(reals); return t, bd
t, bd = pool_run(200); bd.maybe_apply(t, .03)
check('guard: a reservoir of 200 distinct rows (90%% copies) stops the test: dup skips %d, iso flagged %s, iso moves %d' % (bd.counters['iso_dup_skips'], bd.last.get('iso_flagged'), bd.counters['iso_moves']), bd.counters['iso_dup_skips'] == 1 and bd.counters['iso_moves'] == 0)
t, bd = pool_run(None); bd.maybe_apply(t, .03)
check('guard: a duplicate-free reservoir runs the test (dup skips %d)' % bd.counters['iso_dup_skips'], bd.counters['iso_dup_skips'] == 0)
print('ALLOK' if not bad else 'FAILED %d' % bad)

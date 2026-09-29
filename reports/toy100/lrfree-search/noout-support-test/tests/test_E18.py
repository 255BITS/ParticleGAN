"""pkg-E18 CPU tests: `birth_death_feature_scale` ("none" | "std"). "none" = pkg-E17 bit for bit. "std" divides every critic feature by its std on the reference half of the real reservoir:
invariant to the positive rescaling of any hidden unit of a ReLU-type critic (same critic function, different Euclidean distances: codex's E17 audit, reports/toy100/lrfree-search/
e17-feature-gauge-review), while the raw metric is not. usage: python test_E18.py [pkg-E18 path]"""
import sys, json, math, torch, torch.nn as nn
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make, OVERRIDES
E17 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17'; E18 = sys.argv[1] if len(sys.argv) > 1 else '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E18'
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += (not ok); print(('[ok]   ' if ok else '[FAIL] ') + name + (' | ' + detail if detail else ''))
def fresh():
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]
BASE = dict(birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none', birth_death_isolation=True)
# 1. validation
fresh(); sys.path.insert(0, E18)
import particlegan as package
ov = json.load(open(OVERRIDES)); ov.update(num_particles=200, z_dim=2, batch_size=64)
def refused(**kw):
    try: package.get_recipe(**{**ov, **kw}); return False
    except ValueError: return True
check('recipe: unknown scale refused', refused(birth_death_space='critic', birth_death_feature_scale='whiten'))
check('recipe: std with data-space birth-death refused', refused(birth_death_space='data', birth_death_feature_scale='std'))
check('recipe: std with critic-space birth-death accepted', not refused(birth_death_space='critic', birth_death_feature_scale='std'))
# 2. "none" == pkg-E17 exactly
def run(pkg, steps, **extra):
    fresh(); build, real, digest = make(pkg, **BASE, **extra); t = build()
    for i in range(steps): t.step(real(i))
    return digest(t)
d17 = run(E17, 80); d18 = run(E18, 80)
check('scale none: E18 trajectory == E17 trajectory (80 steps)', d17 == d18, f'{d17} {d18}')
# 3. gauge twin on a 2000-row table with planted strays: the same critic function in two parametrisations
def ring_table(scale, c=None, n_stray=60, seed=0):
    fresh(); build, real, digest = make(E18, **BASE, birth_death_feature_scale=scale, num_particles=2000); t = build()
    for i in range(40): t.step(real(i))
    with torch.no_grad(): t.G.weight.copy_(torch.eye(2)); t.G.bias.zero_()
    bd = t.birth_death; gg = torch.Generator().manual_seed(seed)
    for key in [k for k in bd.counters if k.startswith('iso')]: bd.counters[key] = 0
    mode = torch.randint(0, 8, (2000,), generator=gg); ang = mode.float() * math.pi / 4
    z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(2000, 2, generator=gg)
    a = torch.rand(n_stray, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(n_stray, generator=gg)
    z[:n_stray] = torch.stack([r * a.cos(), r * a.sin()], 1)
    with torch.no_grad(): t.prior.z.copy_(z); t.ema_prior.z.copy_(z)
    bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0
    bd.observe_real(torch.cat([real(1000 + i) for i in range(32)])[:2000])
    x = torch.randn(64, 2) * 2
    before = t.D(x).detach()
    if c is not None:            # rescale hidden unit i of the last hidden layer by c_i > 0 and its outgoing weight by 1/c_i: ReLU is positively homogeneous, so D is unchanged
        with torch.no_grad():
            t.D[2].weight.mul_(c[:, None]); t.D[2].bias.mul_(c); t.D[4].weight.div_(c[None, :])
    after = t.D(x).detach()
    return t, bd, float((before - after).abs().max())
g = torch.Generator().manual_seed(11)
c = torch.exp(1.4 * torch.randn(32, generator=g))
res = {}
for scale in ('none', 'std'):
    ta, ba, _ = ring_table(scale); tb, bb, dev = ring_table(scale, c)
    check(f'gauge twin ({scale}): the critic function is unchanged by the rescaling (max |dD| {dev:.1e}; unit scales span {float(c.min()):.2f}..{float(c.max()):.2f})', dev < 1e-4)
    la = ba.maybe_apply(ta, .03); lb = bb.maybe_apply(tb, .03)
    fa, fb = la.get('iso_flagged'), lb.get('iso_flagged')
    ma = set(ba.moved_rows.tolist()) if ba.moved_rows is not None else set(); mb = set(bb.moved_rows.tolist()) if bb.moved_rows is not None else set()
    res[scale] = (fa, fb, len(ma ^ mb))
    print(f'       {scale:5s}: flagged base {fa}, rescaled {fb}, moved-row sets differ in {len(ma ^ mb)} rows')
check('std: flags and moves identical under the rescaling of the hidden units', res['std'][0] == res['std'][1] and res['std'][2] == 0)
print(f"       (raw metric under the same rescaling: flagged {res['none'][0]} vs {res['none'][1]}, {res['none'][2]} moved rows differ; informative only)")
# 3b. the standardised feature matrices themselves are identical under the rescaling (the strong form of the invariance; the flags above are the weak form)
ta, ba, _ = ring_table('std'); tb, bb, _ = ring_table('std', c)
def standardised(t, bd):
    Rf = bd._features(t, bd.reservoir); qf = bd._features(t, t.G(t.prior.z.detach())); Ff = qf.clone(); mu = Rf.mean(0, keepdim=True)
    return bd._standardise(qf - mu, Ff - mu, Rf - mu)
sa, sb = standardised(ta, ba), standardised(tb, bb)
dev = max(float((x - y).abs().max()) / float(x.abs().max()) for x, y in zip(sa, sb))          # relative: the critic runs in float32
rawa = ba._features(ta, ta.G(ta.prior.z.detach())); rawb = bb._features(tb, tb.G(tb.prior.z.detach()))
check('std: standardised features identical under the rescaling (max relative diff %.1e; raw features differ by %.2f)' % (dev, float((rawa - rawb).abs().max())), dev < 1e-5 and float((rawa - rawb).abs().max()) > .1)
# 4. dead units: no NaN, a defined answer
fresh(); build, real, digest = make(E18, **BASE, birth_death_feature_scale='std', num_particles=2000); t = build()
for i in range(40): t.step(real(i))
with torch.no_grad(): t.D[2].weight[:8].zero_(); t.D[2].bias[:8].fill_(-5.)          # 8 dead units on every input
try:
    t.birth_death.observe_real(torch.cat([real(2000 + i) for i in range(32)])[:2000]); last = t.birth_death.maybe_apply(t, .03); ok = last is not None
except Exception as e:
    ok = False; print(repr(e))
check('std: dead hidden units do not break the evaluation', ok)
print('ALLOK' if not bad else 'FAILED %d' % bad)

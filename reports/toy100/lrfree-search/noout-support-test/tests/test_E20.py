"""pkg-E20 CPU tests: (a) persistence: rows are re-drawn only when flagged in two consecutive evaluations; a row that is not flagged again restarts its count; the count is checkpointed
and a resumed run is bit-identical; (b) the duplicate guard: a reservoir with more than Q exact copies of its own rows switches the test off; (c) the p-weighted parent draw is
covered by test_E20_extra.py (B1). usage: python test_E20.py [pkg-E20 path]"""
import sys, json, math, torch, torch.nn as nn
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make, OVERRIDES
PKG = sys.argv[1] if len(sys.argv) > 1 else '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E20'
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += (not ok); print(('[ok]   ' if ok else '[FAIL] ') + name + (' | ' + detail if detail else ''), flush=True)
def fresh():
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]
BASE = dict(birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none', birth_death_isolation=True)
def ring_table(n_stray=60, seed=0, dup=None):
    fresh(); build, real, digest = make(PKG, **BASE, num_particles=2000); t = build()
    for i in range(40): t.step(real(i))
    with torch.no_grad(): t.G.weight.copy_(torch.eye(2)); t.G.bias.zero_()
    bd = t.birth_death; gg = torch.Generator().manual_seed(seed)
    for key in [k for k in bd.counters if k.startswith('iso')]: bd.counters[key] = 0
    bd.iso_run.zero_()
    mode = torch.randint(0, 8, (2000,), generator=gg); ang = mode.float() * math.pi / 4
    z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(2000, 2, generator=gg)
    a = torch.rand(n_stray, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(n_stray, generator=gg)
    z[:n_stray] = torch.stack([r * a.cos(), r * a.sin()], 1)
    with torch.no_grad(): t.prior.z.copy_(z); t.ema_prior.z.copy_(z)
    def refill(j):
        reals = torch.cat([real(1000 + 40 * j + i) for i in range(32)])[:2000]
        if dup is not None: reals = reals[torch.randint(0, dup, (2000,), generator=torch.Generator().manual_seed(j))]      # only `dup` distinct rows: a finite pool
        bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0; bd.observe_real(reals)
    refill(0)
    return t, bd, refill
# ---- (a) persistence
t, bd, refill = ring_table(60)
l1 = bd.maybe_apply(t, .03)
check('persistence: first evaluation flags the 60 strays (%s) and re-draws none (moves %s, moved_rows %s); their run count is 1' % (l1.get('iso_flagged'), l1.get('iso_moves'), None if bd.moved_rows is None else len(bd.moved_rows)),
      l1.get('iso_flagged', 0) >= 55 and l1.get('iso_moves') == 0 and bd.moved_rows is None and int((bd.iso_run[:60] == 1).sum()) >= 55 and bd.counters['iso_moves'] == 0)
sd = t.state_dict(); check('persistence: the run count is part of the checkpoint (key iso_run, %d rows flagged once)' % int((sd['birth_death']['iso_run'] > 0).sum()), 'iso_run' in sd['birth_death'] and int((sd['birth_death']['iso_run'] > 0).sum()) >= 55)
fresh(); build2, real2, dig2 = make(PKG, **BASE, num_particles=2000); t2 = build2(); t2.load_state_dict(sd); bd2 = t2.birth_death
check('persistence: a restored trainer has the same run counts', bool(torch.equal(bd2.iso_run, bd.iso_run)))
refill(1); l2 = bd.maybe_apply(t, .03)
moved = bd.moved_rows if bd.moved_rows is not None else torch.zeros(0, dtype=torch.long)
check('persistence: second evaluation re-draws the strays (flagged %s, moves %s, of the 60 planted: %d) and restarts their run count' % (l2.get('iso_flagged'), l2.get('iso_moves'), int(torch.isin(torch.arange(60), moved).sum())),
      int(torch.isin(torch.arange(60), moved).sum()) >= 55 and bool((bd.iso_run[moved] == 0).all()))
# the restored trainer, given the same reservoir refill, continues bit-identically
def refill_into(bd_, j):
    reals = torch.cat([real2(1000 + 40 * j + i) for i in range(32)])[:2000]
    bd_.fill, bd_.cursor, bd_.rows_since_eval = 0, 0, 0; bd_.observe_real(reals)
refill_into(bd2, 1); bd2.maybe_apply(t2, .03)
check('persistence: the restored trainer\'s second evaluation moves the same rows to the same places (prior table identical: %s, moved rows identical: %s)' % (bool(torch.equal(t.prior.z, t2.prior.z)), bool(torch.equal(bd.moved_rows, bd2.moved_rows))),
      bool(torch.equal(t.prior.z, t2.prior.z)) and bd.moved_rows is not None and bd2.moved_rows is not None and bool(torch.equal(bd.moved_rows, bd2.moved_rows)))
# a row that is not flagged in the second evaluation is not re-drawn and restarts
t, bd, refill = ring_table(60); bd.maybe_apply(t, .03)
with torch.no_grad():
    ang = (torch.arange(60) % 8).float() * math.pi / 4; t.prior.z[:60] = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .01; t.ema_prior.z[:60] = t.prior.z[:60]        # the strays walked home (onto the modes)
refill(1); l2 = bd.maybe_apply(t, .03)
check('persistence: rows that are no longer flagged in the second evaluation are not re-drawn and their run count restarts (moves %s, max run %d)' % (l2.get('iso_moves'), int(bd.iso_run[:60].max())),
      l2.get('iso_moves') == 0 and int(bd.iso_run[:60].max()) == 0)
# ---- (b) duplicate guard: a finite pool of 200 distinct real rows (90% copies)
t, bd, refill = ring_table(60, dup=200)
l1 = bd.maybe_apply(t, .03); refill(1); l2 = bd.maybe_apply(t, .03)
check('duplicate guard: reservoir with 90%% exact copies -> the test does not run (dup skips %d), nothing flagged (flagged %s), nothing re-drawn (moves %d)' % (bd.counters['iso_dup_skips'], l2.get('iso_flagged'), bd.counters['iso_moves']),
      bd.counters['iso_dup_skips'] >= 2 and l2.get('iso_flagged') in (None, 0) and bd.counters['iso_moves'] == 0 and int(bd.iso_run.sum()) == 0)
t, bd, refill = ring_table(60, dup=1500)
bd.maybe_apply(t, .03)
check('duplicate guard: a reservoir drawn with replacement from 1500 distinct rows (about 45%% copies, more than Q) also stops the test (dup skips %d)' % bd.counters['iso_dup_skips'], bd.counters['iso_dup_skips'] == 1)
t, bd, refill = ring_table(60)
bd.maybe_apply(t, .03)
check('duplicate guard: a duplicate-free reservoir runs the test (dup skips %d, flagged %s)' % (bd.counters['iso_dup_skips'], bd.last.get('iso_flagged')), bd.counters['iso_dup_skips'] == 0 and bd.last.get('iso_flagged', 0) >= 55)
print('ALLOK' if not bad else 'FAILED %d' % bad)

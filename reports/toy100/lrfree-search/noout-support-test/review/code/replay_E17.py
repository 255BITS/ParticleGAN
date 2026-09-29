"""Task 4: bit-exact checkpoint replay with the isolation flag ON (serve_average = 4, row_evidence_gate True and False), CPU.
A 2000-row ring table (8 modes, sd .02) with planted strays is trained; strays are planted again when the step counter reaches 70 and 100 (identically in every run) so that
isolation moves really happen in three evaluations (steps 41, 73, 105; a fourth at 137 runs on a repaired table).  Runs are compared with a full recursive bit comparison of
trainer.state_dict() (parameters, EMA, both optimizers, RNG streams, tester, birth-death and row-evidence state):
  R0  uninterrupted, 140 steps
  R1  checkpoint at step 40 (after planting, before the first evaluation), load into a fresh trainer, continue to 140
  R2  checkpoint at step 43 (after the first isolation evaluation)
  R3  checkpoint at step 72 (after the second planting, before the second acting evaluation)
plus: flag off pkg-E17 == pkg-E14 (whole state_dict without the recipe), flag on with never-acting isolation (no strays) == flag off.
usage: python replay_E17.py [--forced-serve]   (forced-serve: the table tester's STATIONARY verdict is forced so that the averaged model really is served between steps)"""
import sys, math, torch
import os; torch.set_num_threads(int(os.environ.get("NT", 2)))
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make
E14 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E14'; E17 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17'
FORCED = '--forced-serve' in sys.argv
N = 2000
def fresh():
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]

def build(pkg, flag, gate, null='scaled', serve=4.0):
    fresh()
    extra = dict(birth_death_space='critic', row_evidence_gate=gate, row_evidence_null=null, table_release_rule='anchor', reopen_signal='none', serve_average=serve, num_particles=N)
    if flag: extra['birth_death_isolation'] = True
    b, real, digest = make(pkg, **extra); t = b()
    if FORCED: t._serve_settled = lambda: True
    return t, real

def plant(t, real, n_stray, seed, reservoir):
    """place the whole table on the ring with n_stray strays (deterministic in `seed`), G = identity; reservoir=True also refills the real reservoir.
    The planting goes into the TRAINING iterate (the averaged model, when served, is swapped out first and back in afterwards)."""
    bd = t.birth_death; gg = torch.Generator().manual_seed(seed)
    mode = torch.randint(0, 8, (N,), generator=gg); ang = mode.float() * math.pi / 4
    z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(N, 2, generator=gg)
    a = torch.rand(n_stray, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(n_stray, generator=gg)
    z[:n_stray] = torch.stack([r * a.cos(), r * a.sin()], 1)
    t._serve_release()
    with torch.no_grad():
        t.G.weight.copy_(torch.eye(2)); t.G.bias.zero_(); t.prior.z.copy_(z); t.ema_prior.z.copy_(z)
    if reservoir:
        bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0
        bd.observe_real(torch.cat([real(1000 + i) for i in range(32)])[:N])
    t._serve_apply()

SCHEDULE = [(40, dict(n_stray=60, seed=0, reservoir=True)), (70, dict(n_stray=60, seed=1, reservoir=False)), (100, dict(n_stray=50, seed=2, reservoir=False))]
TOTAL = 140
CLEAN = [(40, dict(n_stray=0, seed=0, reservoir=True))]          # the same ring table without strays
def advance(t, real, start, stop, schedule=SCHEDULE):
    for s in range(start, stop):
        t.step(real(s))
        for at, kw in schedule:
            if t.completed_steps == at: plant(t, real, **kw)

def run(pkg, flag, gate, stop_at=TOTAL, schedule=SCHEDULE):
    t, real = build(pkg, flag, gate); advance(t, real, 0, stop_at, schedule); return t

def resume(pkg, flag, gate, sd, start, schedule=SCHEDULE):
    t, real = build(pkg, flag, gate); t.load_state_dict(sd); advance(t, real, start, TOTAL, schedule); return t

def diff(a, b, path='', out=None, skip=()):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in skip and path == '': continue
            if k not in a or k not in b: out.append(f'{path}/{k}: key only on one side')
            else: diff(a[k], b[k], f'{path}/{k}', out, skip)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): out.append(f'{path}: length {len(a)} vs {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)): diff(x, y, f'{path}[{i}]', out, skip)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            nan_equal = a.shape == b.shape and a.is_floating_point() and bool(torch.equal(torch.nan_to_num(a.cpu(), nan=1234.5), torch.nan_to_num(b.cpu(), nan=1234.5)))
            if not nan_equal: out.append(f'{path}: tensors differ (max abs {float((a.double() - b.double()).abs().max()) if a.shape == b.shape else "shape"})')
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        out.append(f'{path}: {a!r} vs {b!r}')
    return out

allok = True
def report(name, d):
    global allok
    allok &= not d
    print(('[BIT-EXACT] ' if not d else '[DIFFER]    ') + name + ('' if not d else ' | ' + '; '.join(d[:4]) + (' ...' if len(d) > 4 else '')), flush=True)

for gate in (True, False):
    tag = f'flag ON, serve_average 4, row_evidence_gate {gate}' + (', served average forced' if FORCED else '')
    t0 = run(E17, True, gate); sd0 = t0.state_dict(); c = t0.birth_death.counters; tt = t0._table_tester()
    print(f'--- {tag}: uninterrupted run: evaluations {c["evals"]}, isolation evaluations {c["iso_evals"]}, acted {c["iso_acted"]}, skipped {c["iso_skipped"]}, iso moves {c["iso_moves"]}, ordinary moves {c["moves"]}, '
          f'stale resets {c["stale_resets"]}; averaged model currently served: {t0._fast is not None}; last_decisive {getattr(tt, "last_decisive", None)}; tester rebases {tt.counts.get("rebases") if tt is not None else None}; '
          f'row-evidence resets {t0.row_evidence.counters["resets"] if t0.row_evidence else None}; iso log {t0.birth_death.iso_log}', flush=True)
    for label, stop in (('R1 checkpoint at step 40 (before the first evaluation)', 40), ('R2 checkpoint at step 43 (after the first isolation evaluation)', 43), ('R3 checkpoint at step 72 (before the second acting evaluation)', 72)):
        ta = run(E17, True, gate, stop_at=stop); sd = ta.state_dict()
        tb = resume(E17, True, gate, sd, stop)
        report(f'{tag}: {label} -> resume -> step {TOTAL}', diff(sd0, tb.state_dict()))
    tE17 = run(E17, False, gate); tE14 = run(E14, False, gate); cc = tE17.birth_death.counters
    report(f'flag OFF, gate {gate}: pkg-E17 == pkg-E14 over {TOTAL} steps, same planted table (ordinary moves {cc["moves"]}, evaluations {cc["evals"]})', diff(tE17.state_dict(), tE14.state_dict(), skip=('recipe',)))
    tOn = run(E17, True, gate, schedule=CLEAN); tOff = run(E17, False, gate, schedule=CLEAN); con = tOn.birth_death.counters
    d = [x for x in diff(tOn.state_dict(), tOff.state_dict(), skip=('recipe',)) if not x.startswith('/birth_death/counters') and not x.startswith('/birth_death/last')]
    report(f'flag ON, ring table without strays (iso evaluations {con["iso_evals"]}, acted {con["iso_acted"]}, flagged {con["iso_flagged"]}; ordinary moves {con["moves"]}) == flag OFF, gate {gate}', d)
print('ALL BIT-EXACT' if allok else 'SOME DIFFER')

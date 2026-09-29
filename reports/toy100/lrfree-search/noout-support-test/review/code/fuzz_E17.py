"""Fuzz: (1) `_isolated` == independent float64 reference on random small problems (dimensions, k, ties, duplicates, heavy tails, planted strays, exact copies of reservoir rows);
(2) `_isolation_pick` invariants on random tables/flag sets/child sets (legal parents, ball membership by brute force, dead = flagged minus child).  usage: python fuzz_E17.py [PKG] [trials]"""
import sys, math, os, types, torch
torch.set_num_threads(int(os.environ.get('NT', 2)))
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixture import make
PKG = sys.argv[1] if len(sys.argv) > 1 else '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17'
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_E17_extra.py')).read()
exec("import torch\n" + src[src.index("def ref_scores"): src.index("t, bd, real, digest = trainer(2000)")])       # ref_scores, ref_isolated
BASE = dict(birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none')
build, real, digest = make(PKG, **BASE, birth_death_isolation=True, num_particles=300); t = build(); bd = t.birth_death

def gen(kind, n, d, g):
    if kind == 'gauss':
        m = torch.randint(1, 5, (1,), generator=g).item(); c = 4 * torch.randn(m, d, generator=g, dtype=torch.float64); s = 10 ** (2 * torch.rand(m, generator=g, dtype=torch.float64) - 1.5)
        w = torch.multinomial(torch.rand(m, generator=g) + .05, n, replacement=True, generator=g)
        return c[w] + s[w][:, None] * torch.randn(n, d, generator=g, dtype=torch.float64)
    if kind == 'lattice':
        x = torch.randint(0, 6, (n, d), generator=g).double()
        return x + float(os.environ.get('LATTICE_JITTER', 0)) * torch.randn(n, d, generator=g, dtype=torch.float64)
    if kind == 'dups':
        pts = torch.randn(max(2, n // 40), d, generator=g, dtype=torch.float64); return pts[torch.randint(0, len(pts), (n,), generator=g)]
    if kind == 'heavy': return torch.distributions.StudentT(2., 0., 1.).sample((n, d)).double()
    raise ValueError(kind)

print('--- (1) _isolated vs reference'); bad = 0; nflag = []; kinds = {}
for trial in range(TRIALS):
    g = torch.Generator().manual_seed(9000 + trial)
    kind = ['gauss', 'lattice', 'dups', 'heavy'][trial % 4]; d = [1, 2, 3, 8, 32][int(torch.randint(0, 5, (1,), generator=g))]
    nR = int(torch.randint(60, 900, (1,), generator=g)); k = int(torch.randint(4, 13, (1,), generator=g)); k = min(k, nR // 2 - 2)
    torch.manual_seed(trial)
    R = gen(kind, nR, d, g)
    nq = int(torch.randint(20, 900, (1,), generator=g))
    parts = [gen(kind, nq // 2, d, g), R[torch.randint(0, nR, (nq // 4,), generator=g)], R.mean(0) + 8 * R.std(0).clamp_min(1e-3) * torch.randn(nq - nq // 2 - nq // 4, d, generator=g, dtype=torch.float64)]
    q = torch.cat(parts)
    try:
        got = bd._isolated(q, R, k, torch.float32); ref = ref_isolated(q, R, k, bd.Q)
    except Exception as e:
        bad += 1; print(f'  trial {trial} {kind} d={d} nR={nR} nq={len(q)} k={k}: EXCEPTION {type(e).__name__}: {str(e)[:100]}'); continue
    diff = int((got != ref).sum()); nflag.append(int(ref.sum())); kinds.setdefault(kind, [0, 0]); kinds[kind][0] += 1
    if diff: bad += 1; kinds[kind][1] += 1; print(f'  trial {trial} {kind} d={d} nR={nR} nq={len(q)} k={k}: {diff} rows differ (code {int(got.sum())} flagged, reference {int(ref.sum())})')
print(f'{TRIALS} trials, {bad} with a difference; by kind (trials, differing): {kinds}; trials with >0 flagged rows: {sum(1 for x in nflag if x)}')

print('--- (2) _isolation_pick invariants'); bad2 = 0; acted = 0
Q0 = bd.Q; bd.Q = 1.0
for trial in range(TRIALS):
    g = torch.Generator().manual_seed(500 + trial)
    N = int(torch.randint(8, 300, (1,), generator=g)); dz = [1, 2, 4, 16][int(torch.randint(0, 4, (1,), generator=g))]
    kind = trial % 3
    z = torch.randn(N, dz, generator=g) if kind == 0 else (torch.randint(0, 4, (N, dz), generator=g).float() if kind == 1 else torch.randn(N // 3 + 1, dz, generator=g).repeat(3, 1)[:N])
    flagged = torch.rand(N, generator=g) < torch.rand(1, generator=g).item() * .6
    child = (torch.rand(N, generator=g) < torch.rand(1, generator=g).item() * .3).nonzero().flatten()
    stub = types.SimpleNamespace(prior=types.SimpleNamespace(z=z), completed_steps=0)
    bd.N = N; bd.stream.manual_seed(trial); bd.counters = {**bd.counters}
    dead, parent = bd._isolation_pick(stub, flagged, child)
    isch = torch.zeros(N, dtype=torch.bool); isch[child] = True
    exp_dead = (flagged & ~isch).nonzero().flatten(); keep = ~flagged & ~isch
    problems = []
    n_flag = int(flagged.sum())
    if n_flag == 0 or not len(exp_dead) or not int(keep.sum()):
        if len(dead) or len(parent): problems.append('non-empty answer where nothing can be done')
    else:
        acted += 1
        if not torch.equal(dead, exp_dead): problems.append('dead != flagged minus child')
        if len(parent) != len(dead): problems.append('length')
        else:
            if not bool(keep[parent].all()): problems.append('a parent is flagged or a child')
            d64 = torch.cdist(z[dead].double(), z[keep].double(), compute_mode='donot_use_mm_for_euclid_dist'); dmin = d64.min(1).values
            dpar = (z[dead].double() - z[parent].double()).norm(dim=1)
            tol = 4e-3 * float(z.abs().max()) + 1e-6
            if not bool((dpar <= 2 * dmin + tol).all()): problems.append('parent outside the ball (max excess %.4f)' % float((dpar - 2 * dmin).max()))
    bd.stream.manual_seed(trial); dead2, parent2 = bd._isolation_pick(stub, flagged, child)
    if not (torch.equal(dead, dead2) and torch.equal(parent, parent2)): problems.append('not reproducible from the same stream state')
    if problems: bad2 += 1; print(f'  trial {trial} N={N} dz={dz} kind={kind}: {problems}')
bd.Q = Q0
print(f'{TRIALS} trials ({acted} with something to do), {bad2} with a violated invariant')

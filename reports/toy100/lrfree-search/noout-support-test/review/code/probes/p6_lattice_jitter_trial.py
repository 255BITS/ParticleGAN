import sys, os, math, torch
torch.set_num_threads(2)
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
PKG = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17'
TRIAL_ARG = sys.argv[1]; __file__ = os.path.abspath('fuzz_E17.py'); sys.argv = ['x', PKG]
os.environ['LATTICE_JITTER'] = '1e-3'
src = open('fuzz_E17.py').read()
exec(src[:src.index("print('--- (1)")])
trial = int(TRIAL_ARG)
g = torch.Generator().manual_seed(9000 + trial)
kind = ['gauss', 'lattice', 'dups', 'heavy'][trial % 4]; d = [1, 2, 3, 8, 32][int(torch.randint(0, 5, (1,), generator=g))]
nR = int(torch.randint(60, 900, (1,), generator=g)); k = int(torch.randint(4, 13, (1,), generator=g)); k = min(k, nR // 2 - 2)
torch.manual_seed(trial)
R = gen(kind, nR, d, g)
nq = int(torch.randint(20, 900, (1,), generator=g))
parts = [gen(kind, nq // 2, d, g), R[torch.randint(0, nR, (nq // 4,), generator=g)], R.mean(0) + 8 * R.std(0).clamp_min(1e-3) * torch.randn(nq - nq // 2 - nq // 4, d, generator=g, dtype=torch.float64)]
q = torch.cat(parts)
print(kind, 'd', d, 'nR', nR, 'nq', len(q), 'k', k)
bdm = sys.modules[type(bd).__module__]
R1, R2 = R[0::2], R[1::2]
# knn accuracy of the code path (float32 shortlist + exact recompute) for the three kinds of queries used by the statistic
for name, U, excl in (('R1 vs R1 (leave-one-out)', R1, True), ('R2 vs R1', R2, False), ('q vs R1', q, False)):
    dd, ix = bdm._knn(U, R1, k, exclude=torch.arange(len(R1)) if excl else None, shortlist_dtype=torch.float32)
    D = torch.cdist(U, R1, compute_mode='donot_use_mm_for_euclid_dist')
    if excl: D.fill_diagonal_(float('inf'))
    truth = D.sort(dim=1).values[:, :k]
    bad = ((dd - truth).abs() > 1e-12 * (1 + truth)).any(1)
    print(f'  {name}: rows whose k nearest distances differ from brute force: {int(bad.sum())} of {len(U)}')

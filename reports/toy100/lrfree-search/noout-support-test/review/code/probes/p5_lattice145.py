import sys, os, math, torch
torch.set_num_threads(2)
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make
PKG = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17'
src = open('fuzz_E17.py').read()
TRIAL_ARG = sys.argv[1] if len(sys.argv) > 1 else '145'; __file__ = os.path.abspath('fuzz_E17.py'); sys.argv = ['x', PKG]
exec(src[:src.index("print('--- (1)")])       # imports, ref functions, gen()
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
got = bd._isolated(q, R, k, torch.float32); ref = ref_isolated(q, R, k, bd.Q)
print('code flagged', int(got.sum()), 'ref flagged', int(ref.sum()))
s, null = ref_scores(q, R, k)
n2 = len(null)
p = (1. + (null[None, :] >= s[:, None]).sum(1).double()) / (1. + n2)
ps = p.sort().values; m = len(p)
line = torch.arange(1, m + 1, dtype=torch.float64) * bd.Q / m
ok = (ps <= line).nonzero()
print('n2', n2, 'm', m, 'p_min', 1 / (1 + n2), 'c_min', m * (1 / (1 + n2)) / bd.Q, ' number of rows with p == p_min:', int((p == p.min()).sum()), ' last passing rank', int(ok[-1]) + 1 if len(ok) else None)
print('smallest p-values:', ps[:5].tolist(), ' line at ranks 1..5:', line[:5].tolist())
# which rows does the code flag?
idx = got.nonzero().flatten()
print('code flags rows (first 10):', idx[:10].tolist(), ' their reference p:', p[idx[:10]].tolist())
print('reference scores of code-flagged rows (first 5):', s[idx[:5]].tolist(), ' null max', null.max().item())
# is the difference tie-breaking?  recompute scores with the code's own pieces
bdm = sys.modules[type(bd).__module__]
R1, R2 = R[0::2], R[1::2]
rho_c = bdm._knn(R1, R1, k, exclude=torch.arange(len(R1)), shortlist_dtype=torch.float32)[0][:, -1]
D11 = torch.cdist(R1, R1, compute_mode='donot_use_mm_for_euclid_dist'); D11.fill_diagonal_(float('inf')); rho_r = D11.sort(dim=1).values[:, k - 1]
print('rho code vs ref: max abs diff', float((rho_c - rho_r).abs().max()), ' zero radii', int((rho_r == 0).sum()), 'of', len(rho_r))
# --- elementwise comparison of the null scores: code (shortlist + exact recompute, its own tie-breaking) vs reference (full sort)
def score_code(U):
    dd, ix = bdm._knn(U, R1, k, shortlist_dtype=torch.float32)
    positive = rho_c[rho_c > 0]; floor = float(positive.median()) * 1e-3
    return dd[:, -1] / rho_c[ix].sort(dim=1).values[:, (k - 1) // 2].clamp_min(floor), dd, ix
sc_c, dd_c, ix_c = score_code(R2)
sc_r, _ = ref_scores(R2, R, k)[1], None
sc_r = ref_scores(R2[:2], R, k)[1]        # reference null (scores of R2)
diff = (sc_c - sc_r).abs() > 1e-9 * (1 + sc_r.abs())
print('null entries that differ code vs reference:', int(diff.sum()), 'of', len(sc_r), ' code null max', float(sc_c.max()), 'ref null max', float(sc_r.max()))
for i in diff.nonzero().flatten()[:5].tolist():
    dd_r = torch.cdist(R2[i:i + 1], R1, compute_mode='donot_use_mm_for_euclid_dist')[0]
    srt = dd_r.sort()
    kth = srt.values[k - 1]
    ties = int(((dd_r - kth).abs() < 1e-12).sum()); below = int((dd_r < kth - 1e-12).sum())
    print(f'  row {i}: code score {float(sc_c[i]):.6g}, ref {float(sc_r[i]):.6g}; k-th distance {float(kth):.4f}: {below} points strictly closer, {ties} tied at the k-th distance (k={k}); rho of tied candidates: {sorted(set(round(float(x), 4) for x in rho_r[(dd_r - kth).abs() < 1e-12].tolist()))}')

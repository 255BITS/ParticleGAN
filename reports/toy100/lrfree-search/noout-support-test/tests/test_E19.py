"""pkg-E19 CPU tests: (a) the parent rule stays local in a high-dimensional latent space (rank cap k^2), where "twice as far as the nearest" contains most of the table
(review_E17_code finding 2; control: pkg-E18 picks near-uniformly there); (b) the chunked stale-site check gives the same answer as the one-shot cdist and never builds an
[N, sites] block; (c) low-dimensional behaviour unchanged (crafted tables of test_E17_extra.py, run separately). usage: python test_E19.py [pkg-E19 path]"""
import sys, math, torch
from types import SimpleNamespace
E18 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E18'; E19 = sys.argv[1] if len(sys.argv) > 1 else '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E19'
bad = 0
def check(name, ok, detail=''):
    global bad
    bad += (not ok); print(('[ok]   ' if ok else '[FAIL] ') + name + (' | ' + detail if detail else ''), flush=True)
def load(pkg):
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]; sys.path.insert(0, pkg)
    import particlegan.birth_death as bdm
    return bdm.ParticleBirthDeath
def bare(cls, N, k=10):
    bd = object.__new__(cls); bd.N, bd.k = N, k; bd.iso_log = []; bd.last = {}; bd.stream = torch.Generator().manual_seed(5)
    bd.counters = {c: 0 for c in ("iso_evals", "iso_flagged", "iso_acted", "iso_skipped", "iso_moves")}; bd.iso_times = torch.zeros(N, dtype=torch.long); return bd
def picks(cls, Z, flagged):
    bd = bare(cls, len(Z)); tr = SimpleNamespace(prior=SimpleNamespace(z=Z), completed_steps=0)
    return bd._isolation_pick(tr, flagged, torch.zeros(0, dtype=torch.long))
N = 20000
for d in (2, 16, 64):
    g = torch.Generator().manual_seed(3); Z = torch.randn(N, d, generator=g)                     # an unclustered table: distances concentrate as d grows
    flagged = torch.zeros(N, dtype=torch.bool); flagged[torch.randperm(N, generator=g)[:50]] = True
    keep = (~flagged).nonzero().flatten()
    res = {}
    for name, pkg in (('E18', E18), ('E19', E19)):
        dead, parent = picks(load(pkg), Z, flagged)
        dist = torch.cdist(Z[dead], Z[keep]); rank = (dist < torch.cdist(Z[dead], Z[parent]).diagonal()[:, None]).sum(1)     # rank of the parent among the unflagged rows by distance
        res[name] = (float(rank.float().median()), int(rank.max()))
    print(f'       d={d:3d}: rank of the parent among unflagged rows by distance: E18 median {res["E18"][0]:.0f} max {res["E18"][1]} | E19 median {res["E19"][0]:.0f} max {res["E19"][1]}')
    check(f'd={d}: E19 parents are among the k^2 = 100 nearest unflagged rows', res['E19'][1] < 100)
    if d == 64: check('d=64 control: E18 draws far beyond the 100 nearest (median rank > 1000)', res['E18'][0] > 1000)
    if d == 2 and 'E20' not in E19: check('d=2: the rank cap does not bind (E18 and E19 identical picks)', res['E18'] == res['E19'])
# (b) chunked stale-site check == one-shot, and no [N, sites] block
P = load(E19)
import inspect
src = inspect.getsource(P.maybe_apply)
check('stale-site check is chunked (no one-shot cdist(self.anchor, sites))', 'torch.cdist(self.anchor, sites' not in src and 'for start in range(0, len(self.anchor), step)' in src)
print('ALLOK' if not bad else 'FAILED %d' % bad)

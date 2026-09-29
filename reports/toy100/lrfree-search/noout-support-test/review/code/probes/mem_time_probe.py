"""One maybe_apply evaluation on a fixture trainer with N table rows (ring table + strays), flag on/off; prints peak RSS increment, wall time and the kNN cell counts.
usage: python mem_time_probe.py PKG N FRAC_STRAY FLAG(0/1) [dz] ; run each configuration in its own process (fresh peak-RSS baseline)."""
import sys, json, math, time, os, torch
torch.set_num_threads(2)
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import make
pkg, N, frac, flag = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
dz = int(sys.argv[5]) if len(sys.argv) > 5 else 2
BASE = dict(birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none')
build, real, digest = make(pkg, **BASE, num_particles=N, **({'birth_death_isolation': True} if flag else {}))
t = build(); bd = t.birth_death
for i in range(3): t.step(real(i))
with torch.no_grad(): t.G.weight.copy_(torch.eye(2)); t.G.bias.zero_()
gg = torch.Generator().manual_seed(0)
mode = torch.randint(0, 8, (N,), generator=gg); ang = mode.float() * math.pi / 4
z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(N, 2, generator=gg)
n_stray = int(frac * N)
a = torch.rand(n_stray, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(n_stray, generator=gg)
if n_stray: z[:n_stray] = torch.stack([r * a.cos(), r * a.sin()], 1)
with torch.no_grad(): t.prior.z.copy_(z); t.ema_prior.z.copy_(z)
bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0
bd.observe_real(torch.cat([real(1000 + i) for i in range(math.ceil(N / 64))])[:N])
import importlib; bdm = sys.modules['particlegan.birth_death']
calls = []
orig = bdm._knn
def spy(query, points, k, exclude=None, chunk=2048, shortlist_dtype=None):
    calls.append((len(query), len(points), query.shape[1])); return orig(query, points, k, exclude=exclude, chunk=chunk, shortlist_dtype=shortlist_dtype)
bdm._knn = spy
cd = []
orig_cdist = torch.cdist
def spy_cdist(a, b, *args, **kw):
    cd.append((a.shape[0], b.shape[0], a.shape[1])); return orig_cdist(a, b, *args, **kw)
torch.cdist = spy_cdist
def status(key):
    for line in open('/proc/self/status'):
        if line.startswith(key): return int(line.split()[1]) / 1024.
open('/proc/self/clear_refs', 'w').write('5')          # reset the peak-RSS counter to the current RSS
base = status('VmRSS:'); t0 = time.time()
last = bd.maybe_apply(t, .03)
secs = time.time() - t0
peak = status('VmHWM:')
cells = sum(a * b for a, b, d in calls); cells_w = sum(a * b * d for a, b, d in calls)
print(json.dumps(dict(N=N, flag=flag, strays=n_stray, dz=dz, secs=round(secs, 2), base_mb=round(base), peak_mb=round(peak), extra_mb=round(peak - base),
                      knn_calls=len(calls), knn_cells_over_N2=round(cells / N ** 2, 3), knn_cellsdim_over_N2=round(cells_w / N ** 2, 1),
                      iso_flagged=last.get('iso_flagged'), iso_moves=last.get('iso_moves'), moves=last.get('moves'),
                      sites=None if bd.moved_rows is None else 2 * len(bd.moved_rows))))
print('knn calls (nq, np, d):', calls)
big = sorted(cd, key=lambda c: -c[0] * c[1])[:3]
print('largest direct cdist calls (nq, np, d):', big, ' = %.1f MB float32 each' % (big[0][0] * big[0][1] * 4 / 2 ** 20) if big else '')

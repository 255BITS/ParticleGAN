"""Offline comparison of parent rules for the re-drawn rows on a saved native final state (evaluation side: uses the true centres only to measure where parents sit).
Rules: ball (E19: uniform among unflagged rows within 2x the nearest distance and among the k^2 nearest), cap (uniform among the k^2 nearest unflagged rows),
capw (E20: among the k^2 nearest unflagged rows, probability proportional to the row's own isolation p-value). Shipped statistic (E20 `_isolation_p`, std features, clean centres).
usage: python parent_rules.py RUN_DIR TASK"""
import sys, importlib.util, math, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics'); sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E20')
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
from lib import centers, assign, SD
from toy_models import SimpleMLPDiscriminator
spec = importlib.util.spec_from_file_location('bd20', '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E20/particlegan/birth_death.py')
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
torch.set_num_threads(8); K = 10
class Dummy:
    Q = .05
run, task = sys.argv[1], sys.argv[2]
tr = torch.load(f'{run}/final-state.pt', map_location='cpu', weights_only=False)['trainer']; m = tr['models']
z = m['prior']['z'].float(); W = m['G']['weight'].double(); b = m['G']['bias'].double(); x_clean = (z.double() @ W.T + b)
ce = centers(task); rad = np.linalg.norm(assign(x_clean.numpy(), ce)[1], axis=1)
g = np.random.default_rng(0); NR = len(z)
xr = torch.tensor(ce[g.integers(0, 100, NR)] + SD * g.standard_normal((NR, 2)), dtype=torch.float32)
D = SimpleMLPDiscriminator(2, 128, 3, 3); D.load_state_dict({k: v for k, v in m['D'].items()}); D.eval()
head = [mod for mod in D.modules() if isinstance(mod, torch.nn.Linear) and mod.out_features == 1][-1]
cap = {}; head.register_forward_hook(lambda mod, inp, out: cap.__setitem__('f', inp[0].detach()))
@torch.no_grad()
def feats(a):
    out = []
    for i in range(0, len(a), 8192): D(a[i:i + 8192]); out.append(cap['f'].clone())
    return torch.cat(out).double()
Fq, FR = feats(x_clean.float()), feats(xr); mu = FR.mean(0, keepdim=True); Fq, FR = Fq - mu, FR - mu
Fq, _, FR = bd.ParticleBirthDeath._standardise(Fq, Fq, FR)
p = bd.ParticleBirthDeath._isolation_p(Dummy(), Fq, FR, K, torch.float32)
flagged = bd.ParticleBirthDeath._bh(Dummy(), p)
print(f'{run}: rows {len(z)}, flagged {int(flagged.sum())} (strays >3 sigma {int((rad > 3).sum())}); p of unflagged rows: median {float(p[~flagged].median()):.3f}; share of unflagged rows with p <= .05: {float((p[~flagged] <= .05).double().mean()):.4f}')
keep = (~flagged).nonzero().flatten(); dead = flagged.nonzero().flatten()
if len(dead) == 0: sys.exit(0)
zk = z[keep]; dist = torch.cdist(z[dead], zk); cap_n = min(K * K, len(keep))
near = dist.topk(cap_n, dim=1, largest=False).indices
gen = torch.Generator().manual_seed(1)
def draw(mask, weights=None, reps=20):
    out = []
    for _ in range(reps):
        u = torch.rand(mask.shape, generator=gen, dtype=torch.float64).clamp(1e-12, 1 - 1e-12); key = -torch.log(-torch.log(u))
        if weights is not None: key = key + weights.log()
        key = torch.where(mask, key, torch.full_like(key, -float('inf')))
        out.append(key.argmax(1))
    return torch.cat(out)
n = len(dead)
ball_mask = dist <= torch.minimum(2. * dist.min(1, keepdim=True).values, dist.topk(cap_n, dim=1, largest=False).values[:, -1:])
capmask = torch.zeros_like(dist, dtype=torch.bool); capmask.scatter_(1, near, True)
pk = p[keep].double()[None, :].expand_as(dist)
rules = {'ball (E19)': draw(ball_mask), 'cap uniform': draw(capmask), 'cap weighted by p (E20)': draw(capmask, pk)}
bins = [0, 2, 3, 4, 6, 1e9]
for name, idx in rules.items():
    par = keep[idx]; pr = rad[par.numpy()]; pp = p[par].double()
    h = np.histogram(pr, bins)[0] / len(pr)
    print(f'  {name:26s} parent distance to a centre (sigma) shares {dict(zip(["0-2","2-3","3-4","4-6",">6"], [round(float(v), 3) for v in h]))} | mean p of parents {float(pp.mean()):.3f} | share of parents with p <= .05: {float((pp <= .05).double().mean()):.3f}')

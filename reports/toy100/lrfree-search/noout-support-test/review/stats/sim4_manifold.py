"""Q4c: a low-dimensional manifold inside a high-dimensional feature space, and unused feature dimensions.
(a) the 100-mode 2-D law pushed through random ReLU nets of ambient width h = 16 / 128 / 512 (intrinsic dimension 2, ambient h): false flags of iid rows and clean-scored atoms,
    recall of clean planted strays at 3..12 sigma (300 strays).
(b) 2-D law + p 'nuisance' feature dimensions: real rows carry N(0, tau^2) noise in these dimensions, generator rows carry NONE (the critic sees texture noise in reals
    that the generator's clean centres do not have) / or generator rows carry the SAME noise (iid, exact null). tau in units of sigma.
usage: python sim4_manifold.py EVALS THREADS"""
import sys, numpy as np, torch
import fam
from scorer import Ref
from simlib import relu_map
from kdlib import bh
E, thr = int(sys.argv[1]), int(sys.argv[2]); torch.set_num_threads(thr)
N, K, S = 20000, 10, .03
g = np.random.default_rng(31)
F0 = fam.grid100()
def strays(m, t):
    th = g.uniform(0, 6.283, m); return F0.c[g.integers(0, 100, m)] + t * S * np.stack([np.cos(th), np.sin(th)], 1)
print('(a) 2-D law, ambient width h (random ReLU depth 3): iid rows / clean atoms false flags per eval; recall of 300 clean strays at t sigma')
for h in (16, 128, 512):
    f = relu_map(2, h, seed=5, depth=3, bias_sd=.5); feat = lambda x: f(torch.as_tensor(x * 3, dtype=torch.float32)).double().numpy()
    tab = F0.clean(N, g, .029)[0]; nfl = {'iid': [], 'clean': []}; rec = {t: 0. for t in (3, 4, 5, 6, 8, 12)}
    for ev in range(E):
        R = feat(F0.real(N, g)[0]); ref = Ref(R, K, 'torch', workers=thr)
        nfl['iid'].append(int(bh(ref.p(feat(F0.real(N, g)[0]))).sum())); nfl['clean'].append(int(bh(ref.p(feat(tab))).sum()))
        for t in rec:
            bulk = F0.real(N - 300, g)[0]; st = strays(300, t); fl = bh(ref.p(feat(np.concatenate([bulk, st])))); rec[t] += fl[N - 300:].mean() / E
    print(f'  h={h:4d}: false flags/eval iid {np.mean(nfl["iid"]):.2f} (max {max(nfl["iid"])}), clean {np.mean(nfl["clean"]):.2f} (max {max(nfl["clean"])}) | recall t=3..12: ' + ' '.join(f'{t}:{v:.3f}' for t, v in rec.items()), flush=True)
print('(b) 2-D law (as features, scaled x10) + 126 nuisance dimensions with sd tau*S*10 in the reals')
for tau in (0., .5, 1., 2., 5.):
    nfl = {'gen-without-noise': [], 'gen-with-noise': []}; rec = {t: 0. for t in (4, 6, 8, 12)}
    for ev in range(max(3, E // 3)):
        def embed(x, noisy): return np.concatenate([x * 10, (tau * S * 10) * g.standard_normal((len(x), 126)) if noisy else np.zeros((len(x), 126))], 1)
        R = embed(F0.real(N, g)[0], True); ref = Ref(R, K, 'torch', workers=thr)
        nfl['gen-with-noise'].append(int(bh(ref.p(embed(F0.real(N, g)[0], True))).sum()))
        nfl['gen-without-noise'].append(int(bh(ref.p(embed(F0.clean(N, g, .029)[0], False))).sum()))
        for t in rec:
            bulk = embed(F0.real(N - 300, g)[0], True); st = embed(strays(300, t), True); fl = bh(ref.p(np.concatenate([bulk, st]))); rec[t] += fl[N - 300:].mean() / max(3, E // 3)
    print(f'  tau={tau:3.1f}: false flags/eval gen-with-noise {np.mean(nfl["gen-with-noise"]):.2f}, gen-without-noise (clean) {np.mean(nfl["gen-without-noise"]):.2f} | recall (noisy strays) t=4..12: ' + ' '.join(f'{t}:{v:.3f}' for t, v in rec.items()), flush=True)

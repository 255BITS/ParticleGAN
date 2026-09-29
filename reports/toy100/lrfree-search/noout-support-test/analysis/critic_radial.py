"""Critic value and radial force around the true mode centres in a native final state (evaluation side; uses the true centres). Tests the 'wall' reading of the churn:
E14s keeps strays in the far field as negatives; the support-test runs remove them. usage: python critic_radial.py RUN_DIR TASK [RUN_DIR TASK ...]"""
import sys, math, numpy as np, torch
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/analysis/forensics'); sys.path.insert(0, '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17')
sys.path.insert(0, '/ml2/hypergan/lrfree-20260926/harness/hosts/native100')
from lib import centers, SD
from toy_models import SimpleMLPDiscriminator
torch.set_num_threads(4)
args = sys.argv[1:]
for run, task in zip(args[0::2], args[1::2]):
    tr = torch.load(f'{run}/final-state.pt', map_location='cpu', weights_only=False)['trainer']['models']
    D = SimpleMLPDiscriminator(2, 128, 3, 3); D.load_state_dict({k: v for k, v in tr['D'].items()}); D.eval()
    ce = centers(task); g = np.random.default_rng(0)
    out = []
    for r in (0, 1, 2, 3, 4, 5, 6, 8, 10, 14):
        c = ce[g.integers(0, 100, 4000)]; a = g.uniform(0, 2 * math.pi, 4000)
        x = torch.tensor(c + r * SD * np.stack([np.cos(a), np.sin(a)], 1), dtype=torch.float32, requires_grad=True)
        d = D(x); gr = torch.autograd.grad(d.sum(), x)[0].numpy()
        radial = (gr * np.stack([np.cos(a), np.sin(a)], 1)).sum(1)          # outward component of the critic gradient (positive = D rises outward = generator step pushes outward)
        out.append((r, float(d.detach().mean()), float(np.median(radial)), float((radial > 0).mean())))
    base = out[0][1]
    print(run.split('/')[-1], '| r(sigma): D - D(0) ; median outward gradient ; share of points with outward gradient')
    print('   ' + ' | '.join(f'{r}: {d - base:+.4f} ; {m:+.4f} ; {s:.2f}' for r, d, m, s in out))

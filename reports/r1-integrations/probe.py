"""R1 in the e22-* integration loops: settle, turn the target 30 degrees, compare R1 with pre-R1.

usage: python reports/r1-integrations/probe.py <loop: external|sites|replay|support> <r1: 0|1> --turn T --steps S
One JSON line every 50 updates (metric, R1 fires, ladder reopens); 'turn' and 'done' events.
external: 4-corner toy of examples/e22_external_loop.py, metric = share of 2,048 served samples within .087 of a
          (rotated) corner. routed loops: held-out clean RMSE; the paired edit (target - neutral host) turns.
"""
import argparse, json, math, sys, runpy, time
ROOT = __import__('pathlib').Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / 'examples'))
import torch
from torch import nn
import particlegan
from particlegan import get_recipe, init
assert particlegan.__file__.startswith(str(ROOT))
EX = str(ROOT / 'examples') + '/'
ap = argparse.ArgumentParser(); ap.add_argument('loop'); ap.add_argument('r1', type=int)
ap.add_argument('--turn', type=int, default=400); ap.add_argument('--steps', type=int, default=700)
ap.add_argument("--deg", type=float, default=30.)
a = ap.parse_args()
DEG = a.deg
torch.set_num_threads(1)
OFF = {"reopen_signal": "none", "reopen_anchor": "hold"}
over = {} if a.r1 else OFF

def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])

if a.loop == 'external':
    m = runpy.run_path(EX + 'e22_external_loop.py')
    recipe = get_recipe("e22", num_particles=256, z_dim=2, batch_size=64, output_noise_std=0.029, **over)
    G = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(.2), nn.Linear(32, 2))
    D = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(.2), nn.Linear(32, 1))
    prior = recipe.make_prior()
    init.deterministic_orthogonal_(G, seed=0); init.deterministic_orthogonal_(D, seed=1); init.deterministic_orthogonal_(prior, seed=2)
    loop = m['make_loop'](recipe, G, D, prior)
    policy = loop.policy
    rng = torch.Generator().manual_seed(42)
    base = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    centers = base.clone()
    def step():
        ids = torch.randint(4, (64,), generator=rng)
        real = centers[ids] + .029 * torch.randn(64, 2, generator=rng)
        with torch.autograd.set_multithreading_enabled(False):
            m['update'](loop, real)
    def turn(theta):
        centers.copy_(base @ rot(theta).T)
    def metric():
        x = policy.served_model().sample(2048, generator=torch.Generator().manual_seed(100), output_noise=False)
        return round(float((torch.cdist(x, centers).min(1).values < .087).float().mean()), 4)
else:
    name = {'sites': 'e22_routed_sites.py', 'replay': 'e22_routed_sites.py', 'support': 'e22_routed_support.py'}[a.loop]
    m = runpy.run_path(EX + name)
    if a.loop == 'support':
        loop = m['make_loop'](tokens=16, particles=32, recipe_overrides=over)
        update = runpy.run_path(EX + 'e22_routed_sites.py')['update']
    else:
        loop = m['make_loop'](initialization='api', recipe_overrides=over)
        update = m['update']
    gen = None
    if a.loop == 'replay':
        gen = runpy.run_path(EX + 'e22_routed_replay.py')['checkpointed_generate']
    policy = loop.policy
    neutral = m['neutral']
    edits = {k: getattr(loop, k + '_targets') - neutral(getattr(loop, k + '_context')) for k in ('fit', 'guard', 'test')}
    def step():
        with torch.autograd.set_multithreading_enabled(False):
            if gen is None:
                update(loop)
            else:
                update(loop, generator_forward=gen)
    def turn(theta):
        for k, e in edits.items():
            getattr(loop, k + '_targets').copy_(neutral(getattr(loop, k + '_context')) + e @ rot(theta).T)
    def metric():
        with torch.no_grad():
            y = policy.served_model().routed_forward(loop.test_context)
        return round(float((y - loop.test_targets).square().mean().sqrt()), 5)

def fires():
    return 0 if policy.surprise is None else policy.surprise.fires
def reopens():
    return sum(t.counts.get('reopens', 0) for row in policy.lr_settle.testers for t in row if t is not None)

t0 = time.time()
for i in range(1, a.steps + 1):
    if i == a.turn + 1:
        turn(math.radians(DEG)); print(json.dumps(dict(event='turn', step=i)), flush=True)
    step()
    if i % 50 == 0 or i == 5:
        print(json.dumps(dict(step=i, metric=metric(), fires=fires(), reopens=reopens(), sec=round(time.time() - t0))), flush=True)
print(json.dumps(dict(event='done', loop=a.loop, r1=a.r1, metric=metric(), fires=fires(),
                      fire_log=[] if policy.surprise is None else policy.surprise.log)), flush=True)

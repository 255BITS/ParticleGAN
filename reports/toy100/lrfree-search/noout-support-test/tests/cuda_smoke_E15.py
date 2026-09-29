"""CUDA smoke of the isolation moves under the harness's deterministic-algorithms setting (the CPU suite cannot see CUDA-only nondeterministic ops).
usage: CUDA_VISIBLE_DEVICES=<gpu> python cuda_smoke_E15.py"""
import os, sys, json, math, torch, torch.nn as nn
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.use_deterministic_algorithms(True)
E15 = sys.argv[1] if len(sys.argv) > 1 else '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E15'
sys.path.insert(0, E15); sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
import particlegan as package
from particlegan.particle_prior import ParticlePrior
from fixture import OVERRIDES
ov = json.load(open(OVERRIDES)); ov.update(num_particles=2000, z_dim=2, batch_size=64, birth_death_space='critic', birth_death_isolation=True, row_evidence_gate=True,
                                          table_release_rule='anchor', reopen_signal='none', serve_average=4.0)
dev = torch.device('cuda:0')
recipe = package.get_recipe(**ov); torch.manual_seed(0)
prior = ParticlePrior(2000, 2).to(dev)
with torch.no_grad(): prior.z.uniform_(-5, 5)
G = nn.Linear(2, 2).to(dev)
with torch.no_grad(): G.weight.copy_(torch.eye(2)); G.bias.zero_()
D = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
t = package.GANTrainer(recipe, G, D, prior=prior, seed=0, optimizer_options={'foreach': False, 'fused': False})
def real(i):
    g = torch.Generator().manual_seed(1000 + i); k = torch.randint(0, 8, (64,), generator=g); ang = k.float() * math.pi / 4
    return (torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .03 * torch.randn(64, 2, generator=g)).to(dev)
for i in range(40): t.step(real(i))
bd = t.birth_death
gg = torch.Generator().manual_seed(0); mode = torch.randint(0, 8, (2000,), generator=gg); ang = mode.float() * math.pi / 4
z = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(2000, 2, generator=gg)
a = torch.rand(60, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(60, generator=gg); z[:60] = torch.stack([r * a.cos(), r * a.sin()], 1)
with torch.no_grad():
    G.weight.copy_(torch.eye(2)); G.bias.zero_(); t.prior.z.copy_(z.to(dev)); t.ema_prior.z.copy_(z.to(dev))
bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0
bd.observe_real(torch.cat([real(1000 + i) for i in range(32)])[:2000])
last = bd.maybe_apply(t, .03)
print('flagged', last.get('iso_flagged'), 'moves', last.get('moves'), 'iso_moves', last.get('iso_moves'), '| deterministic algorithms on:', torch.are_deterministic_algorithms_enabled())
for i in range(60): t.step(real(3000 + i))
print('60 more steps ran; iso counters', {k: v for k, v in bd.counters.items() if k.startswith('iso')})
print('OK' if last.get('iso_moves', 0) >= 55 else 'CHECK')

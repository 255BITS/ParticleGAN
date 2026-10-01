"""Record E22 on a frozen native100 task with dense snapshots, for the README animation.

Rebuilds the harness's frozen native host (screen.py run_native: seed 1234, 20k-row table, batch 2048,
identity nn.Linear G, SimpleMLPDiscriminator(2,128x3,fourier=3), serialized backward, the same real-batch
stream order) with the PR's merged package, whose critic init is explicit
(init.deterministic_orthogonal_(critic, seed=1), bit-identical to the recipe-owned init the E22 runs used).
Every --every steps it draws the served model with FIXED latent/noise seeds (side-effect free: forked RNGs,
own latent stream), so each plotted point is the same particle across frames. At the end it scores a 20k
draw with the harness's evaluation seeds so the run can be checked against the archived E22a result.

usage: record_e22.py <task> <out.npz> [--every 25] [--points 4096]
"""
import argparse, json, math, os, sys
from contextlib import nullcontext

REPO = '/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/pkg-RA15-partial-recovery'
HOSTS = '/ml2/hypergan/lrfree-20260926/harness/hosts'
sys.path.insert(0, REPO)
sys.path.insert(1, HOSTS)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
import particlegan
from particlegan import GANTrainer, get_recipe, init
from particlegan.particle_prior import ParticlePrior
from native100 import metrics, problems, toy_models

ap = argparse.ArgumentParser()
ap.add_argument('task'); ap.add_argument('out')
ap.add_argument('--every', type=int, default=25); ap.add_argument('--points', type=int, default=4096)
ap.add_argument('--steps', type=int, default=7000)
ap.add_argument('--gate', action='store_true', help='score a 20k draw at the end of every period and print a PASS/FAIL verdict');
ap.add_argument('--rotate-every', type=int, default=0); ap.add_argument('--rotate-deg', type=float, default=0.)
args = ap.parse_args()
assert particlegan.__file__.startswith(REPO)

# screen.py deterministic()
torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.2, 0); torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest'); torch.manual_seed(0)

seed, device, batch, n_rows = 1234, 'cuda:0', 2048, 20000
options = json.load(open('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930/configs/RA14-replay.json'))
recipe = get_recipe(**{**options, 'num_particles': n_rows, 'z_dim': 2, 'batch_size': batch})

with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.device(device):
        prior = ParticlePrior(n_rows, 2)
        with torch.no_grad():
            prior.z.uniform_(-5.0, 5.0)
        G = nn.Linear(2, 2)
        with torch.no_grad():
            G.weight.copy_(torch.eye(2)); G.bias.zero_()
        D = toy_models.SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=3)
        for layer in D.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    init.deterministic_orthogonal_(D, seed=1)
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=seed, serial_backward=True,
                         optimizer_options={'foreach': False, 'fused': False})
torch.manual_seed(seed)
stream = torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def draw(n, latent_seed, noise_seed):
    """screen.py draw(): live (served) model, clean + output noise; training RNGs untouched."""
    model, table = trainer.G, trainer.prior
    modes = [(m, m.training) for root in (model, table) for m in root.modules()]
    try:
        model.eval(); table.eval()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(noise_seed)
            latent_stream = torch.Generator(device=device).manual_seed(latent_seed)
            latent, indices = table.sample(n, generator=latent_stream)
            clean = trainer._generate(model, latent, 0., latent_stream, indices=indices)
            sigma = float(trainer.output_sigma())
            return clean + sigma * torch.randn_like(clean)
    finally:
        for module, flag in modes:
            module.training = flag


frames, steps = [], []
CENTERS = problems._centers(args.task, device=device, dtype=torch.float32)
def snap(step):
    pts = draw(args.points, 77, 78)
    frames.append(pts.cpu().numpy().astype(np.float16)); steps.append(step)
    angles.append(angle(step))
    if step % 100 == 0 or step in (1000 * k + 25 for k in range(8)):
        # live metrics against the centres in force now (tail -f rot-<task>.log)
        c = CENTERS @ rotation(angle(step)).T
        d, nearest = torch.cdist(pts.float(), c).min(1)
        hq = d <= .09
        lrs = [[round(float(g['lr']), 7) for g in o.param_groups] for o in (trainer.opt_g, trainer.opt_d)]
        print(json.dumps(dict(task=args.task, step=step, target_deg=round(math.degrees(angle(step))),
                              modes=int((torch.bincount(nearest[hq], minlength=100) >= 10).sum()),
                              hq=round(float(hq.float().mean()), 4), sigma=round(float(trainer.output_sigma()), 4),
                              lr_g_prior_d=lrs)), flush=True)
    if args.gate and step and step % args.rotate_every == 0:
        pts20k = draw(20000, seed + 403, seed + 402)
        c = CENTERS @ rotation(angle(step)).T
        d, nearest = torch.cdist(pts20k.float(), c).min(1)
        hq20 = d <= .09
        row = dict(task=args.task, period_end=step, target_deg=round(math.degrees(angle(step))),
                   modes=int((torch.bincount(nearest[hq20], minlength=100) >= 10).sum()), hq=round(float(hq20.float().mean()), 4))
        gate_rows.append(row)
        print('GATE ' + json.dumps(row), flush=True)
        torch.save(trainer.state_dict(), str(owned_output / f'checkpoint-{step:06d}.pt'))
        print('MECHANISM ' + json.dumps(dict(step=step, surprise=None if trainer.policy.surprise is None else trainer.policy.surprise.diagnostics(), backend_selection=trainer.policy._feature_selection.state_dict(), reopen_guard=trainer.policy.reopen_guard.state_dict())), flush=True)
    if step and step % 1000 == 0:   # partial results survive an interrupted run
        np.savez_compressed(args.out + '.partial.npz', frames=np.stack(frames), steps=np.array(steps), angles=np.array(angles))

import math
def angle(step):
    # target rotation (radians) in force for the update that completes `step` (step 0 = before training)
    return 0. if args.rotate_every <= 0 else math.radians(args.rotate_deg) * ((max(step, 1) - 1) // args.rotate_every)
def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(((c, -s), (s, c)), device=device)
def real_batch(step):
    x = problems.sample_real(args.task, batch, device=device, generator=stream)
    return x if args.rotate_every <= 0 else x @ rotation(angle(step)).T
angles = []
gate_rows = []
serial = torch.autograd.set_multithreading_enabled
assert args.task == 'rotated100' and args.steps == 1500 and args.every == 500
assert args.points == 4096 and args.gate and args.rotate_every == 500 and args.rotate_deg == 30
assert round(math.degrees(angle(1001))) == 60 and round(math.degrees(angle(1500))) == 60
support.resume(trainer, real_batch, stream, gate_rows, owned_output)
for step in range(1001, args.steps + 1):
    real_d = real_batch(step)
    cache = []
    def generator_real():
        if not cache:
            cache.append(real_batch(step))
        return cache[0]
    with serial(False):
        trainer.step(real_d, generator_real=generator_real)
    if not cache:
        generator_real()
    support.note_update(trainer, step, stream)
    if step % args.every == 0:
        snap(step)
    if step % 1000 == 0:
        print(f'{args.task} step {step}', flush=True)

final = draw(20000, seed + 403, seed + 402)   # the harness's evaluation seeds
if args.rotate_every <= 0:
    m = metrics.evaluate_samples(final, args.task)
    check = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()
             if k in ('modes', 'hq', 'precision', 'mass_tv')}
else:   # rotating target: hq / modes against the centres in force at the end
    c = problems._centers(args.task, device=device, dtype=torch.float32) @ rotation(angle(args.steps)).T
    d, nearest = torch.cdist(final.float(), c).min(1)
    hq = d <= .09
    check = dict(modes=int((torch.bincount(nearest[hq], minlength=100) >= 10).sum()), hq=float(hq.float().mean()))
print('FINAL', args.task, json.dumps(check), flush=True)
centers = problems._centers(args.task, device='cpu', dtype=torch.float32).numpy()
if args.gate:
    base = gate_rows[0]['hq']
    ok = [r['modes'] >= 95 and r['hq'] >= .9 * base for r in gate_rows[1:]]
    verdict = dict(task=args.task, status='PASS' if all(ok) else 'FAIL', rule='after each turn: modes >= 95 and hq >= .9 x pre-turn hq',
                   pre_turn_hq=base, periods=gate_rows, passed_periods=sum(ok), turns=len(ok))
    print('VERDICT ' + json.dumps(verdict), flush=True)
    assert len(gate_rows) == 3 and len(ok) == 2, 'rotation gate requires both turns'
    json.dump(verdict, open(args.out + '.verdict.json', 'w'), indent=1)
np.savez_compressed(args.out, frames=np.stack(frames), steps=np.array(steps), centers=centers, angles=np.array(angles),
                    final=json.dumps(check))

support.finish(trainer, stream, 'candidate')

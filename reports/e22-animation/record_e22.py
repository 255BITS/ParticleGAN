"""Record E22 on a frozen native100 task with dense snapshots, for the README animation.

Rebuilds the frozen native host of the lrfree harness (screen.py run_native: seed 1234, 20k-row table, batch 2048,
identity nn.Linear G, SimpleMLPDiscriminator(2,128x3,fourier=3), serialized backward, the same real-batch
stream order) with the PR's merged package, whose critic init is explicit
(init.deterministic_orthogonal_(critic, seed=1), bit-identical to the recipe-owned init the E22 runs used).
Every --every steps it draws the served model with FIXED latent/noise seeds (side-effect free: forked RNGs,
own latent stream), so each plotted point is the same particle across frames. At the end it scores a 20k
draw with the harness's evaluation seeds so the run can be checked against the archived E22a result.

usage (CUDA; about 12 min per task alone): python reports/e22-animation/record_e22.py <task> frames-<task>.npz
"""
import argparse, json, os, sys
from contextlib import nullcontext

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
import particlegan
from particlegan import GANTrainer, get_recipe, init
from particlegan.particle_prior import ParticlePrior
from benchmarks.toy100 import metrics, problems   # the frozen native host (same files the harness verifies)
from lib import toy_models

ap = argparse.ArgumentParser()
ap.add_argument('task'); ap.add_argument('out')
ap.add_argument('--every', type=int, default=25); ap.add_argument('--points', type=int, default=4096)
ap.add_argument('--steps', type=int, default=7000)
args = ap.parse_args()
assert particlegan.__file__.startswith(REPO)

# screen.py deterministic()
torch.cuda.set_device(0); torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest'); torch.manual_seed(0)

seed, device, batch, n_rows = 1234, 'cuda:0', 2048, 20000
options = json.load(open(f'{REPO}/configs/100gaussians/e22-noout.json'))
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
            latent, _ = table.sample(n, generator=latent_stream)
            clean = trainer._generate(model, latent, 0., latent_stream)
            sigma = float(trainer.output_sigma())
            return clean + sigma * torch.randn_like(clean)
    finally:
        for module, flag in modes:
            module.training = flag


frames, steps = [], []
def snap(step):
    frames.append(draw(args.points, 77, 78).cpu().numpy().astype(np.float16)); steps.append(step)

serial = torch.autograd.set_multithreading_enabled
snap(0)
for step in range(1, args.steps + 1):
    real_d = problems.sample_real(args.task, batch, device=device, generator=stream)
    cache = []
    def generator_real():
        if not cache:
            cache.append(problems.sample_real(args.task, batch, device=device, generator=stream))
        return cache[0]
    with serial(False):
        trainer.step(real_d, generator_real=generator_real)
    if not cache:
        generator_real()
    if step % args.every == 0:
        snap(step)
    if step % 1000 == 0:
        print(f'{args.task} step {step}', flush=True)

final = draw(20000, seed + 403, seed + 402)   # the harness's evaluation seeds
m = metrics.evaluate_samples(final, args.task)
check = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()
         if k in ('modes', 'hq', 'precision', 'mass_tv')}
print('FINAL', args.task, json.dumps(check), flush=True)
centers = problems._centers(args.task, device='cpu', dtype=torch.float32).numpy()
np.savez_compressed(args.out, frames=np.stack(frames), steps=np.array(steps), centers=centers,
                    final=json.dumps(check))

#!/usr/bin/env python
"""Frozen-checkpoint endpoint sensitivity diagnostics for row-EM.

Run in the original candidate environment. No training batches are stepped and
no checkpoint is written.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPORT = Path(__file__).resolve().parent
ATTEMPT = Path('/ml2/hypergan/gan-attempts')
CANDIDATE = ATTEMPT / 'row-em-renew-20260928/candidate/package'
CAL_RUNS = ATTEMPT / 'row-em-renew-20260928/runs'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
sys.path.insert(0, str(CANDIDATE))
from particlegan import GANTrainer, get_recipe


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


models = load_module('native100_models', HARNESS / 'hosts/native100/toy_models.py')
problems = load_module('native100_problems', HARNESS / 'hosts/native100/problems.py')


def build(task):
    path = CAL_RUNS / task / 'final-state.pt'
    device = torch.device('cuda:0')
    state = torch.load(path, map_location=device, weights_only=False)['trainer']
    overrides = json.loads((REPORT / 'overrides.json').read_text())
    overrides.update(num_particles=20_000, z_dim=2, batch_size=2048)
    recipe = get_recipe(**overrides)
    prior = recipe.make_prior().to(device)
    generator = torch.nn.Linear(2, 2).to(device)  # frozen native host: affine_square_v1
    critic = models.SimpleMLPDiscriminator(2, 128, 3, 3).to(device)
    trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=1234,
                         optimizer_options={'foreach': False, 'fused': False},
                         serial_backward=True)
    trainer.load_state_dict(state)
    trainer.G.eval()
    trainer.D.eval()
    return trainer, state


def real(task, n, seed):
    g = torch.Generator(device='cuda:0').manual_seed(seed)
    return problems.sample_real(task, n, device='cuda:0', generator=g)


def outputs(trainer, n, seed, weights, sigma, *, differentiable=False):
    """Draw by inverse CDF so every law shares uniform draws, jitter, and noise."""
    device = trainer.device
    rng = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand(n, device=device, generator=rng)
    cdf = torch.cumsum(weights, dim=0)
    idx = torch.searchsorted(cdf, u.contiguous()).clamp_max(len(weights) - 1)
    latent = trainer.prior(idx)
    latent = trainer.controller.perturb_latent(latent, rng, trainer.prior)
    clean = trainer.G(latent)
    eps = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=rng)
    return clean + sigma * eps


def ci(values):
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float('nan')
    return {'mean': mean, 'ci95': [mean - 1.96 * se, mean + 1.96 * se], 'se': se}


def block_ci(x, block=500):
    x = np.asarray(x, dtype=np.float64)
    n = len(x) // block
    return ci(x[:n * block].reshape(n, block).mean(1))


def paired_losses(trainer, left, right):
    with torch.no_grad():
        sr, sf = trainer.D(left), trainer.D(right)
        delta = sr - sf
        ld = F.softplus(-delta)
        lg = F.softplus(delta)
        auc = (delta > 0).float() + .5 * (delta == 0).float()
    return {'left_minus_right_score': block_ci(delta.cpu().numpy()),
            'rank_left_gt_right': block_ci(auc.cpu().numpy()),
            'relativistic_left_loss': block_ci(ld.cpu().numpy()),
            'relativistic_right_loss': block_ci(lg.cpu().numpy())}, (ld, lg)


def kernel(x, y, bandwidth):
    d2 = torch.cdist(x, y).square()
    return torch.exp(-d2 / (2 * bandwidth * bandwidth))


def mmd2(x, y, bandwidths):
    n, m = len(x), len(y)
    xx, yy = torch.cdist(x, x).square(), torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()
    result = x.new_zeros(())
    for h in bandwidths:
        kxx = torch.exp(-xx / (2 * h * h))
        kyy = torch.exp(-yy / (2 * h * h))
        kxy = torch.exp(-xy / (2 * h * h))
        result = result + (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
        result = result + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
        result = result - 2 * kxy.mean()
    return result / len(bandwidths)


def mmd_bandwidths(reference):
    from scipy.spatial import cKDTree
    x = reference.detach().cpu().numpy()
    distances, _ = cKDTree(x).query(x, k=9, workers=1)
    h = float(np.median(distances[:, -1]))
    return [h / 2, h, h * 2], h


def endpoint(trainer, task):
    device = trainer.device
    n = 20_000
    r = real(task, n, 91001)
    real_a, real_b = real(task, n, 91002), real(task, n, 91003)
    uniform = torch.full_like(trainer.prior.row_weights, 1 / len(trainer.prior.row_weights))
    calibrated = trainer.prior.row_weights.detach().clone()
    sigma_train = float(trainer._output_sigma(.02))
    sigma_cal = float(trainer.row_em.sampling_sigma)
    laws = {
        'uniform_training_width': (uniform, sigma_train),
        'calibrated_mass_training_width': (calibrated, sigma_train),
        'uniform_mass_calibrated_width': (uniform, sigma_cal),
        'calibrated_mass_calibrated_width': (calibrated, sigma_cal),
    }
    scored, losses, clouds = {}, {}, {}
    with torch.no_grad():
        for name, (weights, sigma) in laws.items():
            f = outputs(trainer, n, 92001, weights, sigma)
            scored[name], losses[name] = paired_losses(trainer, r, f)
            clouds[name] = f
    controls = {}
    with torch.no_grad():
        controls['real_vs_real'], _ = paired_losses(trainer, real_a, real_b)
        fa = outputs(trainer, n, 93001, uniform, sigma_train)
        fb = outputs(trainer, n, 93002, uniform, sigma_train)
        controls['baseline_vs_independent_baseline'], _ = paired_losses(trainer, fa, fb)

    # Fixed characteristic Gaussian-kernel MMD; bandwidth comes from a separate
    # real-only reference sample's eighth-neighbor distance.
    ref = real(task, 4096, 94001)
    bandwidths, base_h = mmd_bandwidths(ref)
    mmd_real = real(task, 4096, 94002)
    with torch.no_grad():
        fakes = {k: outputs(trainer, 4096, 95001, *v) for k, v in laws.items()}
        null_real = real(task, 4096, 94003)
        null_fake = outputs(trainer, 4096, 95002, uniform, sigma_train)
    def block_mmd(x, y):
        vals = []
        for start in range(0, 4096, 256):
            vals.append(float(mmd2(x[start:start+256], y[start:start+256], bandwidths).detach()))
        return ci(vals)
    mmd = {k: block_mmd(mmd_real, v) for k, v in fakes.items()}
    mmd['real_vs_real_null'] = block_mmd(mmd_real, null_real)
    mmd['baseline_vs_baseline_null'] = block_mmd(fakes['uniform_training_width'], null_fake)

    paired_deltas = {}
    base_lg = losses['uniform_training_width'][1]
    for name, (_, lg) in losses.items():
        paired_deltas[name] = block_ci((lg - base_lg).cpu().numpy())
    return {'sample_count': n, 'training_sigma': sigma_train,
            'calibrated_sigma': sigma_cal, 'laws': scored, 'paired_G_loss_delta_vs_uniform_training': paired_deltas,
            'null_controls': controls, 'mmd': {'bandwidths': bandwidths, 'reference_knn_median': base_h,
                                              'values_by_256_block': mmd}, 'clouds': clouds,
            'real_mmd': mmd_real}


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required to reconstruct and score the native checkpoints')
    torch.set_num_threads(1)
    report = {'protocol': 'frozen-native100-critic-gap-v1', 'tasks': {}}
    for task in ('grid100', 'rotated100', 'staggered100'):
        trainer, state = build(task)
        end = endpoint(trainer, task)
        # Keep only scalar summaries in JSON; do not archive generated clouds.
        end.pop('clouds')
        end.pop('real_mmd')
        report['tasks'][task] = {**end, 'saved_steps': state['completed_steps'],
                                 'critic_loss': 'relativistic paired softplus; raw score has no sigmoid meaning'}
        print(task, 'endpoint complete', flush=True)
        del trainer
    out = REPORT / 'critic-gap-results.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()

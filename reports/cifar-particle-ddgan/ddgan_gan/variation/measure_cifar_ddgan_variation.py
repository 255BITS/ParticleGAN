#!/usr/bin/env python
"""Frozen-checkpoint numerical variation at fixed DDGAN noisy input; no grids."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_cifar_particle_ddgan import rng, state_hash, write_json
from experiments.measure_cifar_particle_variation import pair_metrics, cosine
from lib.image_particle_autoencoder import DirectGenerator, ImageRoutingEncoder
from lib.image_particle_ddgan import ParticleDDGenerator
from particlegan import MoGParticlePrior


@torch.no_grad()
def audit(run, out):
    from torchvision.datasets import CIFAR10
    from lib.cifar_metrics import uint8_images
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    out.mkdir(parents=True, exist_ok=False)
    path = run / 'checkpoint.pt'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    for name, expected in ck['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    diffusion = cfg['model'] == 'ddgan'
    bounded = cfg['arm'] == 'bounded'
    assert diffusion or bounded
    g = (ParticleDDGenerator(cfg['z_dim'], cfg['width'], cfg['alpha_bar']) if diffusion
         else DirectGenerator(cfg['z_dim'], cfg['width']))
    e = ImageRoutingEncoder(cfg['z_dim'], cfg['width'])
    prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=cfg['z_dim'],
                             sigma_rel=cfg['sigma_rel'], generator=rng(cfg['seed'] + 1, 'cpu'))
    for module, key in ((g, 'ema_G'), (e, 'ema_E'), (prior, 'ema_prior')):
        module.load_state_dict(ck[key])
        module.cuda().eval().requires_grad_(False)
    before = state_hash([g, e, prior])
    extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).cuda().eval().requires_grad_(False)
    dataset = CIFAR10(cfg['data_dir'], train=False)
    n, draws, batch = 512, 8, 128
    real8 = torch.from_numpy(dataset.data[:n]).permute(0, 3, 1, 2).contiguous().cuda()
    real = real8.float() / 127.5 - 1
    means = prior.means()
    codes = torch.cat([e(x, means, prior.sigma, cfg['temperature'])[0] for x in real.split(batch)]) if bounded else None
    original_features = torch.cat([extractor(x)[0].cpu() for x in real8.split(batch)])
    noise = torch.randn((n, draws, cfg['z_dim']), device='cuda', generator=rng(cfg['seed'] + 30000))
    prior_z = prior.sample(n * draws, rng(cfg['seed'] + 30001))[0]
    context_rng = rng(cfg['seed'] + 30002)
    conditions = [('encoded', a) for a in (0., .5, 1.)] if bounded else []
    if diffusion:
        conditions += [('prior', 1.)]
    source_paths = [Path(__file__), ROOT / 'experiments/measure_cifar_particle_variation.py']
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    for p in source_paths:
        (out / p.name).write_bytes(p.read_bytes())
    protocol = {'inputs': n, 'draws': draws, 'pairs_per_input': 28, 'model': cfg['model'], 'arm': cfg['arm'],
                'test_indices': 'first512 CIFAR10 test images', 'checkpoint_sha256': digest,
                'checkpoint_step': ck['step'], 'script_hashes': source_hashes,
                'fixed_context': 'one forward-corrupted X_t per input/t, held fixed across every draw/condition',
                'prediction': 'clean image, before reverse transition noise' if diffusion else 'direct decoded image',
                'noise_seeds': [cfg['seed'] + i for i in (30000, 30001, 30002)],
                'latent_noise': 'unclipped z_X + a*sigma*epsilon; injected noise, not learned posterior',
                'feature_metric': 'cosine distance of normalized Inception pool2048 from uint8 RGB',
                'retention': 'nearest of512 deterministic reconstructions; not semantic identity accuracy',
                'weights': 'EMA G/E/prior; all frozen', 'sigma': float(prior.sigma)}
    write_json(out / 'protocol.json', protocol)
    rows, arrays, references, context_hashes = [], {}, {}, {}
    for ti in (range(1, g.schedule.steps + 1) if diffusion else [0]):
        if diffusion:
            t = torch.full((n,), ti, device='cuda', dtype=torch.long)
            _, xt = g.schedule.forward_pair(real, t, context_rng)
            context_hashes[str(ti)] = hashlib.sha256(xt.cpu().numpy().tobytes()).hexdigest()
        base_features = None
        for kind, alpha in conditions:
            key = f't{ti}_{kind}_{alpha:g}'
            latents = (codes[:, None] + alpha * prior.sigma * noise).flatten(0, 1) if kind == 'encoded' else prior_z
            pixels, features, errors = [], [], []
            for lo in range(0, n * draws, batch):
                z = latents[lo:lo + batch]
                ids = torch.arange(lo, lo + len(z), device='cuda') // draws
                y = g(z, xt[ids], t[ids]) if diffusion else g(z)
                errors.append((y-real[ids]).square().flatten(1).mean(1).cpu())
                q = uint8_images(y)
                pixels.append(q.cpu()); features.append(extractor(q)[0].cpu())
            pixels = torch.cat(pixels).reshape(n, draws, 3, 32, 32)
            features = torch.cat(features).reshape(n, draws, -1)
            errors = torch.cat(errors).reshape(n, draws)
            metrics = pair_metrics(pixels, features)
            metrics['mse'] = errors.mean(1)
            metrics['feature_to_input'] = cosine(features, original_features[:, None]).mean(1)
            if kind == 'encoded' and alpha == 0:
                assert (metrics['unique_draws'] == 1).all()
                base_features = features[:, 0].clone()
                base_mse = float(errors.mean())
                reference = float(cosine(base_features, base_features.roll(257, 0)).mean())
                references[str(ti)] = {'unrelated_reconstruction_cosine': reference, 'base_mse': base_mse}
            if base_features is not None:
                sim = F.normalize(features.flatten(0, 1), dim=-1) @ F.normalize(base_features, dim=-1).T
                nearest = sim.argmax(1).reshape(n, draws)
                metrics['own_anchor_nearest'] = (nearest == torch.arange(n)[:, None]).float().mean(1)
            row = {'t': ti, 'kind': kind, 'noise_scale_sigma': alpha,
                   **{k: float(v.mean()) for k,v in metrics.items()}}
            row['pair_pixel_rmse'] = row['pair_pixel_mse'] ** .5
            if base_features is not None:
                row['feature_variation_percent_unrelated'] = 100 * row['pair_feature_cosine'] / reference
                row['mse_increase_percent'] = 100 * (row['mse'] / base_mse - 1)
            rows.append(row)
            arrays.update({f'{key}__{k}': v.numpy() for k,v in metrics.items()})
            print(row, flush=True)
    assert state_hash([g, e, prior]) == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    np.savez_compressed(out / 'per_input_metrics.npz', **arrays)
    result = {'protocol': protocol, 'rows': rows, 'references': references,
              'checkpoint_unchanged': True, 'model_state_unchanged': True, 'context_hashes': context_hashes,
              'seconds': time.perf_counter() - started,
              'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30}
    write_json(out / 'summary.json', result)
    print(f"COMPLETE seconds={result['seconds']:.1f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    audit(args.run, args.out)

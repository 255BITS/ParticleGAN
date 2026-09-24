#!/usr/bin/env python
"""Numerical conditional variation audit; no training or saved image grids."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_cifar_particle_ae import rng, state_hash, write_json
from lib.image_particle_autoencoder import DirectGenerator, ImageRoutingEncoder
from particlegan import MoGParticlePrior


def pair_metrics(pixels, features):
    """Per-input metrics over every unordered pair of repeated draws.

    pixels: [inputs, draws, ...], uint8; features: [inputs, draws, features].
    Return mean squared pixel distance (uint8 units) and normalized feature
    cosine distance. Taking sqrt after averaging MSE gives pooled pixel RMSE.
    """
    b, draws = pixels.shape[:2]
    if draws < 2:
        raise ValueError('at least two draws required')
    pairs = torch.triu_indices(draws, draws, offset=1, device=pixels.device)
    flat = pixels.flatten(2)
    left, right = flat[:, pairs[0]], flat[:, pairs[1]]
    same = (left == right).all(-1)
    normalized = F.normalize(features.float(), dim=-1)
    distances = (1 - (normalized[:, pairs[0]] * normalized[:, pairs[1]]).sum(-1)).clamp_min(0)
    # A draw is unique if it differs from every earlier draw.
    unique = torch.ones(b, device=pixels.device)
    for j in range(1, draws):
        unique += (~(flat[:, :j] == flat[:, j:j + 1]).all(-1).any(1)).float()
    return {'pair_pixel_mse': (left.float() - right.float()).square().mean((1, 2)),
            'pair_feature_cosine': distances.mean(1),
            'identical_pair_fraction': same.float().mean(1), 'unique_draws': unique}


def cosine(a, b):
    return (1 - (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1)).clamp_min(0)


@torch.no_grad()
def audit(run, out):
    from torchvision.datasets import CIFAR10
    from lib.cifar_metrics import uint8_images, PROTOCOL
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
    assert cfg['arm'] == 'bounded'
    for name, expected in ck['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    g = DirectGenerator(cfg['z_dim'], cfg['width'])
    e = ImageRoutingEncoder(cfg['z_dim'], cfg['width'])
    prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=cfg['z_dim'],
                             sigma=0, generator=rng(cfg['seed'] + 1, 'cpu'))
    for module, key in ((g, 'ema_G'), (e, 'ema_E'), (prior, 'ema_prior')):
        module.load_state_dict(ck[key])
        module.cuda().eval().requires_grad_(False)
    before = state_hash([g, e, prior])
    extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).cuda().eval().requires_grad_(False)
    dataset = CIFAR10(cfg['data_dir'], train=False)
    n, draws, batch = 512, 8, 128
    real_uint8 = torch.from_numpy(dataset.data[:n]).permute(0, 3, 1, 2).contiguous().cuda()
    real = real_uint8.float() / 127.5 - 1
    means = prior.means()
    codes, ids = [], []
    for x in real.split(batch):
        z, k, _, _ = e(x, means, prior.sigma, cfg['temperature'])
        codes.append(z); ids.append(k)
    codes, ids = torch.cat(codes), torch.cat(ids)
    original_features = torch.cat([extractor(x)[0] for x in real_uint8.split(batch)]).cpu()
    # One common tensor of draws for every noise-strength condition. This is
    # Monte Carlo evaluation of a single checkpoint, not a training seed sweep.
    noise = torch.randn((n, draws, cfg['z_dim']), device='cuda', generator=rng(cfg['seed'] + 30000))
    protocol = {'inputs': n, 'draws_per_input': draws, 'pairs_per_input': draws*(draws-1)//2,
                'test_indices': 'first 512 CIFAR-10 test images',
                'test_class_counts': np.bincount(dataset.targets[:n], minlength=10).tolist(),
                'conditions': ['encoded+0*sigma*noise', 'encoded+.25*sigma*noise',
                               'encoded+.5*sigma*noise', 'encoded+1*sigma*noise',
                               'encoded+2*sigma*noise', 'selected_center+sigma*noise'],
                'noise_seed': cfg['seed'] + 30000, 'common_noise_across_conditions': True,
                'clipping_latent': False, 'weights': 'EMA G/E/prior from final 10k checkpoint',
                'sigma': float(prior.sigma), 'feature_extractor': PROTOCOL,
                'precision': 'float32, TF32 disabled; uint8 feature input',
                'feature_metric': '1 - cosine_similarity in normalized Inception pool2048',
                'pixel_diversity_units': 'RMSE in uint8 levels [0,255]',
                'reconstruction_mse_units': 'float image pixels [-1,1]',
                'unrelated_reference': 'pair each deterministic reconstruction with index (i+257)%512',
                'checkpoint_sha256': digest,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out / 'protocol.json', protocol)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    all_metrics, rows = {}, []
    base_features = base_pixels = None
    conditions = [('encoded', a) for a in (0., .25, .5, 1., 2.)] + [('selected_center', 1.)]
    print(f'START inputs={n} draws={draws} conditions={len(conditions)} checkpoint={digest}', flush=True)
    for kind, alpha in conditions:
        label = f'{kind}_{alpha:g}'
        anchor = codes if kind == 'encoded' else means[ids]
        latents = (anchor[:, None] + alpha * prior.sigma * noise).flatten(0, 1)
        pixels, features, errors = [], [], []
        for lo in range(0, len(latents), batch):
            decoded = g(latents[lo:lo + batch])
            target_ids = torch.arange(lo, lo + len(decoded), device='cuda') // draws
            errors.append((decoded - real[target_ids]).square().flatten(1).mean(1).cpu())
            quantized = uint8_images(decoded)
            pixels.append(quantized.cpu())
            features.append(extractor(quantized)[0].cpu())
        pixels = torch.cat(pixels).reshape(n, draws, 3, 32, 32)
        features = torch.cat(features).reshape(n, draws, -1)
        errors = torch.cat(errors).reshape(n, draws)
        if base_features is None:
            base_features, base_pixels = features[:, 0].clone(), pixels[:, 0].clone()
            shifted = torch.roll(torch.arange(n), 257)
            unrelated_pixel_mse = float((base_pixels.float() - base_pixels[shifted].float()).square().mean())
            unrelated_features = float(cosine(base_features, base_features[shifted]).mean())
            unrelated_real_features = float(cosine(original_features, original_features[shifted]).mean())
            with np.load(run / f"recon_{ck['step']:06d}.npz") as saved:
                assert abs(errors.mean().item() - saved['recon'][:n].mean()) < 1e-5
            base_mse = float(errors.mean())
        metrics = pair_metrics(pixels, features)
        metrics['recon_mse'] = errors.mean(1)
        metrics['feature_to_input'] = cosine(features, original_features[:, None]).mean(1)
        metrics['feature_to_deterministic_reconstruction'] = cosine(features, base_features[:, None]).mean(1)
        # Descriptive retention of the deterministic decoded anchor, not a
        # semantic classifier or a guarantee of preserving the original image.
        similarities = F.normalize(features.flatten(0, 1), dim=-1) @ F.normalize(base_features, dim=-1).T
        nearest = similarities.argmax(1).reshape(n, draws)
        metrics['own_anchor_nearest_fraction'] = (nearest == torch.arange(n)[:, None]).float().mean(1)
        row = {'condition': label, 'noise_scale_sigma': alpha,
               **{key: float(value.mean()) for key, value in metrics.items()}}
        row['pair_pixel_rmse'] = row['pair_pixel_mse'] ** .5
        row['feature_diversity_percent_unrelated_reconstruction'] = 100 * row['pair_feature_cosine'] / unrelated_features
        row['recon_mse_change_percent'] = 100 * (row['recon_mse'] / base_mse - 1)
        if alpha == 0:
            assert row['identical_pair_fraction'] == 1 and row['unique_draws'] == 1
        rows.append(row)
        for key, value in metrics.items():
            all_metrics[f'{label}__{key}'] = value.numpy()
        print(json.dumps(row, allow_nan=False), flush=True)
    assert state_hash([g, e, prior]) == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    np.savez_compressed(out / 'per_input_metrics.npz', **all_metrics, selected_ids=ids.cpu().numpy())
    result = {'protocol': protocol, 'rows': rows,
              'reference': {'unrelated_reconstruction_pixel_rmse': unrelated_pixel_mse ** .5,
                            'unrelated_reconstruction_feature_cosine': unrelated_features,
                            'unrelated_real_feature_cosine': unrelated_real_features},
              'checkpoint_unchanged': True, 'model_state_unchanged': True,
              'total_seconds': time.perf_counter() - started,
              'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30}
    write_json(out / 'summary.json', result)
    print(f"COMPLETE seconds={result['total_seconds']:.1f} peak_GiB={result['peak_memory_gb']:.2f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/cifar_particle_ae/scout/bounded')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/cifar_particle_ae/variation')
    args = parser.parse_args()
    audit(args.run, args.out)

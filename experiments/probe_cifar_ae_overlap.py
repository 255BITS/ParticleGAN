#!/usr/bin/env python
"""Frozen-checkpoint noise sweep: historical FID, matched coverage, latent overlap."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
import yaml
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch_fidelity.metric_fid import fid_statistics_to_metric

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPENDENCIES = {
    'experiments/train_cifar_ae_scaling.py': '811441edc7795be251c57595a9cda19fe59f38526be5bcb95c722bd0ba6c13b6',
    'experiments/probe_cifar_ae_information.py': '564623d386035348a0ad0325f73a97b028e1d32eb1c895f0743bb5486f8c442f',
}
for name, sha in DEPENDENCIES.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
from experiments import train_cifar_ae_scaling as base
from experiments.probe_cifar_ae_information import density_coverage, tensor_hash
from experiments.run_grid import code_provenance
from lib.cifar_metrics import FIDEvaluator, PROTOCOL, uint8_images
from particlegan import MoGParticlePrior

DEFAULTS = {'checkpoint': '',
 'checkpoint_sha256': '',
 'out_dir': '',
 'samples': 50000,
 'quality_samples': 10000,
 'latent_samples': 32768,
 'batch_size': 128,
 'noise_scales': [1.0, 0.75, 0.5],
 'expected_fid': None,
 'real_feature_cache': 'results/cifar_ddgan/information_cache/real-4eef7a439a192b550544e798f6002218604fbf2e7ab903d094f2c77448fbf07d.pt',
 'real_features_sha256': 'fe089b56afca1897c928bcb682e5e0cc380ab23c4fce0492521122d87a62da46'}


def precision(tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


@torch.no_grad()
def latent_geometry(means, sigma, factor, samples):
    """Exact equal-weight isotropic posterior; all components, matched draws."""
    precision(False)
    count, dim = means.shape
    rows = torch.arange(count, device='cuda')
    nearest, siblings = [], []
    for lo in range(0, count, 512):
        dist = torch.cdist(means[lo:lo+512], means)
        dist[torch.arange(len(dist), device='cuda'), rows[lo:lo+len(dist)]] = float('inf')
        values, ids = dist.min(1)
        nearest.append(values)
        siblings.append(ids//factor == rows[lo:lo+len(dist)]//factor)
    nearest = torch.cat(nearest)
    stream = base.rng(79301)
    wrong = sibling_wrong = 0
    entropy = parent_entropy = 0.
    for lo in range(0, samples, 512):
        size = min(512, samples-lo)
        ids = torch.randint(count, (size,), device='cuda', generator=stream)
        z = means[ids] + sigma*torch.randn(size, dim, device='cuda', generator=stream)
        lp = (-torch.cdist(z, means).square()/(2*sigma*sigma)).log_softmax(1)
        predictions = lp.argmax(1)
        mistakes = predictions != ids
        wrong += int(mistakes.sum())
        sibling_wrong += int((mistakes & (predictions//factor == ids//factor)).sum())
        entropy += float(-(lp.exp()*lp).sum())/math.log(2)
        parents = lp.double().exp().reshape(size, count//factor, factor).sum(2)
        parents /= parents.sum(1, keepdim=True)
        parent_entropy += float(-(parents*parents.clamp_min(1e-38).log2()).sum())
    return dict(samples=samples, wrong_nearest_count=wrong, wrong_nearest_fraction=wrong/samples,
                same_parent_wrong_count=sibling_wrong, different_parent_wrong_count=wrong-sibling_wrong,
                posterior_entropy_bits=entropy/samples, parent_posterior_entropy_bits=parent_entropy/samples,
                median_nearest_distance=float(nearest.median()),
                nearest_distance_quantiles=torch.quantile(nearest, torch.tensor([0., .1, .5, .9, 1.], device='cuda')).tolist(),
                nearest_is_sibling_fraction=float(torch.cat(siblings).float().mean()),
                sigma=float(sigma), rms_noise_radius=float(sigma)*math.sqrt(dim))


@torch.no_grad()
def train(cfg):
    assert set(cfg) == set(DEFAULTS)
    assert cfg['samples'] > 1 and 5 < cfg['quality_samples'] <= 10000
    assert cfg['latent_samples'] > 0 and cfg['batch_size'] > 0
    assert 1. in cfg['noise_scales'] and all(math.isfinite(s) and s > 0 for s in cfg['noise_scales'])
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    path = ROOT/cfg['checkpoint']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(path) == cfg['checkpoint_sha256']
    ck = torch.load(path, map_location='cpu', weights_only=False)
    for name, sha in ck['sources'].items():
        assert digest(ROOT/name) == sha, name
    parent = {**base.DEFAULTS, **ck['config']}
    torch.manual_seed(parent['seed'])
    g, d, e = base.build_models(parent)
    del d, e
    g = g.cuda().eval().requires_grad_(False)
    g.load_state_dict(ck['ema_G'])
    prior = MoGParticlePrior(parent['num_particles'], parent['z_dim'], sigma_rel=parent['sigma_rel']).cuda()
    factor = parent['expansion_factor']
    assert factor == 16
    base.activate_expansion(prior, factor, parent['seed']+90000, False)
    prior.load_state_dict(ck['ema_prior'])
    prior.eval().requires_grad_(False)
    frozen = base.state_hash([g, prior])
    clone_state = prior.clone_rng.get_state()
    means, sigma = prior.means(), prior.sigma
    assert cfg['batch_size'] == parent['eval_batch_size'], 'preserve historical RNG batching'
    out = ROOT/cfg['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'summary.json').exists()
    provenance = code_provenance(__file__, sys.executable)
    provenance['sources'].update(DEPENDENCIES)
    base.write_json(out/'provenance.json', provenance)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, sha in provenance['sources'].items():
            assert digest(ROOT/name) == sha
            archive.write(ROOT/name, name)
    real = torch.from_numpy(CIFAR10(parent['data_dir'], train=True, download=False).data).permute(0, 3, 1, 2).contiguous()
    evaluator = FIDEvaluator(real, parent['fid_cache'], cfg['batch_size'])
    cache = torch.load(ROOT/cfg['real_feature_cache'], map_location='cpu', weights_only=False)
    assert tensor_hash(cache['features']) == cache['sha256'] == cfg['real_features_sha256']
    assert cache['metadata']['extractor_state_sha256'] == base.state_hash([evaluator.model])
    real_features = cache['features'][:cfg['quality_samples']].cuda()
    results = []
    for scale in cfg['noise_scales']:
        print(f'START_SCALE step={ck["step"]} scale={scale}', flush=True)
        precision(True)
        stream = base.rng(parent['seed']+10000)
        prior.clone_rng.manual_seed(parent['seed']+91000)
        chunks = []
        for lo in range(0, cfg['samples'], cfg['batch_size']):
            z, ids = prior.sample(min(cfg['batch_size'], cfg['samples']-lo), stream)
            if scale != 1.:
                z = means[ids] + scale*(z-means[ids])
            chunks.append(uint8_images(g(z)).cpu())
        prior.clone_rng.set_state(clone_state)
        images = torch.cat(chunks)
        save_image(images[:100].float()/255, out/f'samples_noise_{scale:g}.png', nrow=10)
        stats = evaluator.statistics(images)
        fid = float(fid_statistics_to_metric(stats, evaluator.real, verbose=False)['frechet_inception_distance'])
        print('FID', json.dumps(dict(step=ck['step'], scale=scale, fid=fid)), flush=True)
        if scale == 1. and cfg['samples'] == 50000 and cfg['expected_fid'] is not None:
            assert abs(fid-cfg['expected_fid']) < .01, (fid, cfg['expected_fid'])
        # Match the prior information probe's independent 10k quality protocol.
        precision(False)
        stream = base.rng(72337)
        features = []
        for lo in range(0, cfg['quality_samples'], cfg['batch_size']):
            size = min(cfg['batch_size'], cfg['quality_samples']-lo)
            ids = torch.randint(len(means), (size,), device='cuda', generator=stream)
            noise = torch.randn(size, parent['z_dim'], device='cuda', generator=stream)
            features.append(evaluator.model(uint8_images(g(means[ids]+sigma*scale*noise)))[0].cpu())
        features = torch.cat(features)
        quality = density_coverage(real_features, features.cuda())
        geometry = latent_geometry(means, sigma*scale, factor, cfg['latent_samples'])
        row = dict(step=ck['step'], noise_scale=scale, fid=fid, quality=quality, geometry=geometry)
        results.append(row)
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print('RESULT', json.dumps(row, allow_nan=False), flush=True)
        torch.save(features, out/f'features_noise_{scale:g}.pt')
    assert frozen == base.state_hash([g, prior]) and digest(path) == cfg['checkpoint_sha256']
    summary = dict(config=cfg, final=results[-1], results=results, parent_unchanged=True,
                   frozen_state_unchanged=True, fid_protocol=PROTOCOL,
                   quality_protocol=f"Historical information probe: {cfg['quality_samples']} real/fake, k5, FP32 TF32 disabled; identical draws across scales.",
                   real_features_sha256=cfg['real_features_sha256'], total_seconds=time.perf_counter()-started,
                   note='Inference-only distribution changes. Component confusion is not semantic collapse. Same RNG draws across noise scales; no training seed repeats.')
    base.write_json(out/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = base.read_config(args.config)
    assert not set(cfg)-set(DEFAULTS)
    train({**DEFAULTS, **cfg})

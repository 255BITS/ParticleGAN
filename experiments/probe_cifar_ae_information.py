#!/usr/bin/env python
"""Read-only particle information and real-feature density/coverage diagnostic.

Conditional bits use a deliberately small diagonal nearest-centroid softmax
decoder, fitted on train noise, selected on validation noise, measured on test
noise. These are estimates of a variational lower bound, not exact image entropy.
Density/coverage follow Naeem et al. (2020), including strict radius comparisons.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'experiments/train_cifar_ae_scaling.py'
BASE_SHA = '811441edc7795be251c57595a9cda19fe59f38526be5bcb95c722bd0ba6c13b6'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
from experiments import train_cifar_ae_scaling as base
from experiments.run_grid import code_provenance
from lib.cifar_metrics import uint8_images, PROTOCOL
from particlegan import MoGParticlePrior

DEFAULTS = {'checkpoint': '', 'checkpoint_sha256': '', 'out_dir': '',
            'samples': 10000, 'batch_size': 128, 'parents': 32,
            'train_draws': 64, 'val_draws': 32, 'test_draws': 64,
            'variance_draws': 16, 'k': 5, 'distance_chunk': 512,
            'feature_cache': 'results/cifar_ddgan/information_cache'}
SOURCES = {'information': 'https://arxiv.org/abs/1606.03657',
           'density_coverage': 'https://proceedings.mlr.press/v119/naeem20a.html',
           'radius_convention': 'https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py'}
FEATURE_EXTRACTION_PROTOCOL = {key: value for key, value in PROTOCOL.items()
                               if key not in ('dataset', 'split', 'real_samples', 'precision')}
FEATURE_EXTRACTION_PROTOCOL['precision'] = 'float32 extractor with TF32 disabled'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(x):
    return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()


def variance_decomposition(values):
    """Balanced [parent, child, draw, feature] nested ANOVA sum of squares."""
    x = values.double()
    assert x.ndim == 4 and x.shape[2] > 1
    p, k, n, _ = x.shape
    child = x.mean(2, keepdim=True)
    parent = child.mean(1, keepdim=True)
    mean = parent.mean(0, keepdim=True)
    within = float((x-child).square().sum())
    sibling = float((child-parent).square().sum()) * n
    between = float((parent-mean).square().sum()) * k*n
    total = float((x-mean).square().sum())
    assert abs(total-within-sibling-between) <= max(1., total)*1e-10
    return {'within_child_fraction': within/total if total else 0.,
            'between_siblings_fraction': sibling/total if total else 0.,
            'between_parents_fraction': between/total if total else 0.,
            'between_children_fraction': (sibling+between)/total if total else 0.,
            'total_variance_trace': total/(p*k*n-1),
            'total_sum_squares': total, 'within_child_sum_squares': within,
            'between_siblings_sum_squares': sibling, 'between_parents_sum_squares': between}


def squared_distances(x, y):
    return (x.square().sum(1)[:, None] + y.square().sum(1)[None, :] - 2*x@y.T).clamp_min_(0)


@torch.no_grad()
def density_coverage(real, fake, k=5, chunk=512):
    """Exact all-pairs neighborhoods with O(chunk*N) distance storage; no ANN."""
    assert real.ndim == fake.ndim == 2 and real.shape[1] == fake.shape[1]
    assert 0 < k < len(real) and len(fake) and chunk > 0
    assert torch.isfinite(real).all() and torch.isfinite(fake).all()
    radii = []
    for lo in range(0, len(real), chunk):
        x = real[lo:lo+chunk]
        distances = squared_distances(x, real)
        distances[torch.arange(len(x), device=x.device), torch.arange(lo, lo+len(x), device=x.device)] = float('inf')
        radii.append(distances.kthvalue(k, dim=1).values)
    radii = torch.cat(radii)
    contained = 0
    covered = 0
    for lo in range(0, len(real), chunk):
        distances = squared_distances(real[lo:lo+chunk], fake)
        mask = distances < radii[lo:lo+chunk, None]
        contained += int(mask.sum())
        covered += int(mask.any(1).sum())
    return {'density': contained/(k*len(fake)), 'coverage': covered/len(real),
            'real_samples': len(real), 'fake_samples': len(fake), 'k': k,
            'radius_squared_mean': float(radii.mean()), 'distance_dtype': str(real.dtype),
            'radius_boundary': 'strict <, kth other real neighbor'}


def fit_decoder(train, validation):
    """Train-only centroids and diagonal variance; validation selects calibration.

    Arrays [child, draw, feature]. No test inputs are accepted by this function.
    A uniform predictor is a validation candidate, avoiding forced overconfidence.
    """
    train, validation = train.double(), validation.double()
    k, _, d = train.shape
    centroids = train.mean(1)
    if k == 1:
        return {'validation_ce_nats': 0., 'shrinkage': None, 'temperature': None,
                'centroids': centroids, 'variance': None}
    variance = (train-centroids[:, None]).square().mean((0, 1))
    scale = max(float(variance.mean()), 1e-12)
    labels = torch.arange(k).repeat_interleave(validation.shape[1])
    best = {'validation_ce_nats': math.log(k), 'shrinkage': None, 'temperature': None,
            'centroids': centroids, 'variance': None}
    for shrinkage in (0.1, 0.5, 1.0):
        diagonal = ((1-shrinkage)*variance + shrinkage*scale).clamp_min(1e-12)
        # The x^2 term cancels in the softmax; using mean distance removes D scale.
        logits = (2*(validation.flatten(0, 1)/diagonal)@centroids.T - (centroids.square()/diagonal).sum(1))/d
        for temperature in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.):
            ce = float(torch.nn.functional.cross_entropy(logits/temperature, labels))
            if ce < best['validation_ce_nats']:
                best = {'validation_ce_nats': ce, 'shrinkage': shrinkage, 'temperature': temperature,
                        'centroids': centroids, 'variance': diagonal}
    return best


def score_decoder(model, test):
    test = test.double()
    k, n, d = test.shape
    labels = torch.arange(k).repeat_interleave(n)
    if model['temperature'] is None:
        losses = torch.full((k*n,), math.log(k), dtype=torch.float64)
        accuracy = 1/k
    else:
        c, v = model['centroids'], model['variance']
        logits = (2*(test.flatten(0, 1)/v)@c.T - (c.square()/v).sum(1))/d/model['temperature']
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        accuracy = float((logits.argmax(1) == labels).double().mean())
    ce_bits = float(losses.mean())/math.log(2)
    return {'decodable_bits': math.log2(k)-ce_bits, 'test_ce_bits': ce_bits,
            'test_accuracy': accuracy, 'validation_ce_bits': model['validation_ce_nats']/math.log(2),
            'shrinkage': model['shrinkage'], 'temperature': model['temperature']}


def shuffled_features(x, generator):
    """Reassign independently balanced labels by permuting feature rows."""
    flat = x.flatten(0, 1)
    return flat[torch.randperm(len(flat), generator=generator)].reshape_as(x)


def conditional_information(train, validation, test, seed=91337):
    assert train.ndim == validation.ndim == test.ndim == 4
    p, k, _, _ = train.shape
    assert validation.shape[:2] == test.shape[:2] == (p, k)
    rows = {'observed': [], 'shuffled_labels': [], 'identical_clones': []}
    rng = torch.Generator().manual_seed(seed)
    for parent in range(p):
        splits = [x[parent] for x in (train, validation, test)]
        variants = {'observed': splits,
                    'shuffled_labels': [shuffled_features(x, rng) for x in splits],
                    # Equal inputs for all labels, with independent splits retained.
                    'identical_clones': [x[:1].expand(k, -1, -1).clone() for x in splits]}
        for name, (a, b, c) in variants.items():
            rows[name].append(score_decoder(fit_decoder(a, b), c))
    result = {'parents': p, 'children_per_parent': k, 'available_sibling_bits': math.log2(k),
              'decoder': 'train-only diagonal nearest-centroid softmax; validation shrinkage/temperature including uniform',
              'train_draws_per_child': train.shape[2], 'validation_draws_per_child': validation.shape[2],
              'test_draws_per_child': test.shape[2], 'per_parent': rows}
    result['total_panel_images'] = p*k*(train.shape[2]+validation.shape[2]+test.shape[2])
    result['controls'] = {'shuffled_labels': 'Independently permuted balanced feature-to-label assignments in train, validation and test.',
                          'identical_clones': 'Feature-level exact null: repeat child0 features under every label, retaining independent noise splits. This is a decoder check, not another generator evaluation.'}
    for name, values in rows.items():
        bits = np.array([x['decodable_bits'] for x in values])
        result[name] = {'decodable_bits': float(bits.mean()),
                        'parent_standard_error': float(bits.std(ddof=1)/math.sqrt(p)) if p > 1 else None,
                        'test_ce_bits': float(np.mean([x['test_ce_bits'] for x in values])),
                        'test_accuracy': float(np.mean([x['test_accuracy'] for x in values]))}
    result['interpretation'] = 'Held-out estimate of a decoder-dependent conditional MI lower bound, not exact entropy. Negative estimates retained. Parent SE is descriptive, not uncertainty across training runs.'
    return result


@torch.no_grad()
def extract(model, images, batch_size):
    return torch.cat([model(x.cuda())[0].cpu() for x in images.split(batch_size)])


@torch.no_grad()
def generate_features(g, model, means, sigma, n, seed, batch_size):
    """Independent noise by (split,parent,child), invariant to extraction batches."""
    rows = []
    for i, mean in enumerate(means):
        stream = base.rng(seed+i)
        z = mean[None] + sigma*torch.randn(n, len(mean), device='cuda', generator=stream)
        pixels = torch.cat([uint8_images(g(x)).cpu() for x in z.split(batch_size)])
        rows.append(extract(model, pixels, batch_size))
    return torch.stack(rows)


@torch.no_grad()
def train(cfg):
    from torchvision.datasets import CIFAR10
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    assert set(cfg) == set(DEFAULTS)
    for key in ('samples', 'batch_size', 'parents', 'train_draws', 'val_draws', 'test_draws', 'variance_draws', 'k', 'distance_chunk'):
        assert type(cfg[key]) is int and cfg[key] > 0, key
    assert cfg['samples'] > cfg['k'] and cfg['variance_draws'] > 1
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    path = ROOT/cfg['checkpoint']
    assert digest(path) == cfg['checkpoint_sha256']
    ck = torch.load(path, map_location='cpu', weights_only=False)
    for name, expected in ck['sources'].items():
        assert digest(ROOT/name) == expected, name
    parent = {**base.DEFAULTS, **ck['config']}
    factor = parent['expansion_factor']
    assert factor in (1, 4, 8, 16) and cfg['parents'] <= parent['num_particles']
    torch.manual_seed(parent['seed'])
    g, d, e = base.build_models(parent)
    del d, e
    g = g.cuda().eval().requires_grad_(False)
    g.load_state_dict(ck['ema_G'])
    prior = MoGParticlePrior(parent['num_particles'], parent['z_dim'], sigma_rel=parent['sigma_rel']).cuda()
    if factor > 1:
        base.activate_expansion(prior, factor, parent['seed']+90000, False)
    prior.load_state_dict(ck['ema_prior'])
    prior.eval().requires_grad_(False)
    frozen = base.state_hash([g, prior])
    # All draws below use external RNG and direct means: clone RNG/exposure untouched.
    means, sigma = prior.means(), prior.sigma
    out = ROOT/cfg['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'summary.json').exists(), 'fresh output directory required'
    provenance = code_provenance(__file__, sys.executable)
    provenance['sources'][str(BASE.relative_to(ROOT))] = BASE_SHA
    base.write_json(out/'provenance.json', provenance)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in provenance['sources'].items():
            assert digest(ROOT/name) == expected
            archive.write(ROOT/name, name)
    model = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).cuda().eval().requires_grad_(False)
    real = torch.from_numpy(CIFAR10(parent['data_dir'], train=True, download=False).data).permute(0, 3, 1, 2).contiguous()
    assert cfg['samples'] <= len(real)
    real_ids = torch.randperm(len(real), generator=torch.Generator().manual_seed(71337))[:cfg['samples']]
    diagnostic_protocol = {'dataset': 'CIFAR-10', 'split': 'train', 'real_samples': cfg['samples'],
                           'fake_samples': cfg['samples'], 'real_subset_seed': 71337,
                           'feature_extraction': FEATURE_EXTRACTION_PROTOCOL,
                           'generation_precision': 'FP32 with TF32 disabled',
                           'distance_precision': 'FP32 exact chunked squared Euclidean distances',
                           'statistics_precision': 'float64 classifier and variance',
                           'density_coverage_k': cfg['k'], 'fid_recomputed': False,
                           'density_definition': 'Mean real-neighborhood memberships per fake, divided by k',
                           'coverage_definition': 'Fraction of real kNN balls containing at least one fake; strict radius comparisons'}
    metadata = {'feature_extraction_protocol': FEATURE_EXTRACTION_PROTOCOL, 'dataset': 'CIFAR-10', 'split': 'train',
                'indices_sha256': tensor_hash(real_ids), 'images_sha256': tensor_hash(real[real_ids]),
                'extractor_state_sha256': base.state_hash([model]), 'samples': cfg['samples']}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache = ROOT/cfg['feature_cache']/f'real-{key}.pt'
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            payload = torch.load(cache, map_location='cpu', weights_only=False)
            assert payload['metadata'] == metadata and tensor_hash(payload['features']) == payload['sha256']
            real_features = payload['features']
        else:
            real_features = extract(model, real[real_ids], cfg['batch_size'])
            temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
            torch.save({'metadata': metadata, 'features': real_features, 'sha256': tensor_hash(real_features)}, temporary)
            temporary.replace(cache)
    assert real_features.shape == (cfg['samples'], 2048) and torch.isfinite(real_features).all()
    print('REAL_FEATURES', json.dumps({'cache': str(cache), 'samples': cfg['samples']}), flush=True)
    stream = base.rng(72337)
    fake = []
    for lo in range(0, cfg['samples'], cfg['batch_size']):
        n = min(cfg['batch_size'], cfg['samples']-lo)
        ids = torch.randint(len(means), (n,), device='cuda', generator=stream)
        noise = torch.randn(n, parent['z_dim'], device='cuda', generator=stream)
        fake.append(model(uint8_images(g(means[ids]+sigma*noise)))[0].cpu())
    fake_features = torch.cat(fake)
    quality = density_coverage(real_features.cuda(), fake_features.cuda(), cfg['k'], cfg['distance_chunk'])
    real64, fake64 = real_features.double(), fake_features.double()
    quality['feature_variance_trace_ratio_to_real'] = float(fake64.var(0).sum()/real64.var(0).sum())
    quality['feature_mean_distance_squared'] = float((fake64.mean(0)-real64.mean(0)).square().sum())
    print('DENSITY_COVERAGE', json.dumps(quality), flush=True)
    parent_ids = torch.randperm(parent['num_particles'], generator=torch.Generator().manual_seed(73337))[:cfg['parents']]
    selected = means.reshape(parent['num_particles'], factor, -1)[parent_ids.cuda()]
    panels = {}
    for split, count, seed in [('train', cfg['train_draws'], 1000000), ('validation', cfg['val_draws'], 2000000),
                               ('test', cfg['test_draws'], 3000000), ('variance', cfg['variance_draws'], 4000000)]:
        # Parent-index stride16 keeps matching children/noise paired across counts.
        chunks = [generate_features(g, model, selected[i], sigma, count, seed+int(parent_ids[i])*16, cfg['batch_size'])
                  for i in range(cfg['parents'])]
        panels[split] = torch.stack(chunks)
        print('PANEL', json.dumps({'split': split, 'shape': list(panels[split].shape)}), flush=True)
    information = conditional_information(panels['train'], panels['validation'], panels['test'])
    variance = variance_decomposition(panels['variance'])
    variance['note'] = 'Finite-sample balanced ANOVA fractions; between-mean terms contain estimation noise. Descriptive feature dispersion, not semantic mode count.'
    torch.save({'metadata': metadata, 'parent_ids': parent_ids, 'panels': panels,
                'fake_features': fake_features, 'checkpoint_sha256': cfg['checkpoint_sha256']}, out/'features.pt')
    assert frozen == base.state_hash([g, prior]) and digest(path) == cfg['checkpoint_sha256']
    summary = {'final': {'information': information, 'variance': variance, 'density_coverage': quality},
               'config': cfg, 'parent_step': ck['step'], 'num_particles': len(means), 'expansion_factor': factor,
               'parent_unchanged': True, 'frozen_state_unchanged': True, 'protocol': diagnostic_protocol,
               'feature_extraction_protocol': FEATURE_EXTRACTION_PROTOCOL,
               'information': information, 'variance': variance, 'density_coverage': quality,
               'real_feature_cache': str(cache), 'real_reference': metadata, 'sources': SOURCES,
               'real_features_sha256': tensor_hash(real_features),
               'diagnostic_precision': 'FP32 generator and extractor with TF32 disabled; float64 classifier/variance. Matches extractor protocol, but generator TF32 differs from historical training/FID generation.',
               'total_seconds': time.perf_counter()-started,
               'note': 'Read-only EMA diagnostics. Information/variance are conditional on sampled original parents; feature distinguishability can reflect artifacts. Density and coverage depend on extractor, k and sample count, and do not establish semantic quality or recall. No FID50k is recomputed.'}
    base.write_json(out/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = base.read_config(args.config)
    assert not set(config)-set(DEFAULTS)
    train({**DEFAULTS, **config})

#!/usr/bin/env python
"""Read-only final-checkpoint FID5k audit to compare with earlier diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_cifar_particle_ae import generation, rng, write_json
from lib.cifar_metrics import FIDEvaluator
from lib.image_particle_autoencoder import DirectGenerator
from particlegan import MoGParticlePrior


def main(run):
    from torchvision.datasets import CIFAR10
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    start = time.perf_counter()
    path = run / 'checkpoint.pt'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    for name, expected in ck['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    g = DirectGenerator(cfg['z_dim'], cfg['width'])
    prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=cfg['z_dim'],
                             sigma=0, generator=rng(cfg['seed'] + 1, 'cpu'))
    g.load_state_dict(ck['ema_G'])
    prior.load_state_dict(ck['ema_prior'])
    g.cuda().eval().requires_grad_(False)
    prior.cuda().eval().requires_grad_(False)
    images = torch.from_numpy(CIFAR10(cfg['data_dir'], train=True).data).permute(0, 3, 1, 2).contiguous()
    evaluator = FIDEvaluator(images, cfg['fid_cache'], cfg['eval_batch_size'])
    out = run / 'audit5k'
    out.mkdir(exist_ok=False)
    print(f"AUDIT arm={cfg['arm']} step={ck['step']} final checkpoint, FID5k, no training", flush=True)
    final = generation(g, prior, evaluator, cfg, 5000, out, ck['step'])
    old = np.asarray(Image.open(run / f"samples_{ck['step']:06d}.png")).astype(float)
    new = np.asarray(Image.open(out / f"samples_{ck['step']:06d}.png")).astype(float)
    # Same first 100 draws. Permit at most one quantization level for CUDA numerics.
    assert np.max(np.abs(old - new)) <= 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    result = {'arm': cfg['arm'], 'step': ck['step'], 'final': final,
              'checkpoint_sha256': digest, 'checkpoint_unchanged': True,
              'grid_max_uint8_difference': float(np.max(np.abs(old - new))),
              'total_seconds': time.perf_counter() - start}
    write_json(out / 'summary.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)

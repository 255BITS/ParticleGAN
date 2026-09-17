#!/usr/bin/env python
"""Config runner for the Hopfield read study, on CPU or CUDA."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config, recipe_defaults
from experiments.run_grid import code_provenance

DEFAULTS = {
    **recipe_defaults('100gaussians'),
    'fourier': 2, 'dataset': 'imbalanced', 'read': 'uniform',
    'beta': 16.0, 'learn_beta': False, 'study_metrics': True,
    'n_eval': 100000, 'log_interval': 100, 'snapshot_interval': 500,
    'seed': 1234, 'prior_kind': 'particles', 'reg_fd_eps': 0.05,
    'reg_sync_stats': True, 'fused_adam': False,
    'device': 'cuda', 'threads': 1, 'save_checkpoint': True,
    'out_dir': 'results/hopfield/default',
}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')


def train(cfg):
    for key in ('epochs', 'steps_per_epoch', 'batch_size', 'num_particles',
                'log_interval', 'snapshot_interval', 'n_eval', 'threads'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not cfg['study_metrics']:
        raise ValueError('the study runner requires study_metrics=True')
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError(out)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    provenance = code_provenance(__file__, sys.executable)
    write_json(out / 'provenance.json', provenance)
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in provenance['sources']:
            archive.write(ROOT / name, name)
    device = torch.device(cfg['device'])
    environment = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'device': str(device), 'threads': cfg['threads'], 'tf32': False}
    if device.type == 'cuda':
        environment['gpu'] = torch.cuda.get_device_name(device)
    write_json(out / 'environment.json', environment)
    spec = importlib.util.spec_from_file_location('hopfield_example', ROOT / 'examples/100gaussians.py')
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    kwargs = {k: v for k, v in cfg.items() if k not in ('device', 'threads', 'save_checkpoint')}
    start = time.perf_counter()
    result = example.train(**kwargs, device_str=str(device), return_details=True)
    final = result['final']
    np.save(out / 'final_samples.npy', result['final_samples'].detach().cpu().numpy())
    if cfg['save_checkpoint']:
        checkpoint = {'config': cfg, 'G': result['ema_G'].state_dict(),
                      'prior': result['ema_prior'].state_dict()}
        if result.get('ema_read') is not None:
            checkpoint['read'] = result['ema_read'].state_dict()
        torch.save(checkpoint, out / 'final.pt')
    summary = {'config': cfg, 'final': final, 'noise_floor': result['noise_floor'],
               'train_seconds': result['train_seconds'],
               'total_seconds': time.perf_counter() - start,
               'environment': environment, 'provenance': provenance}
    write_json(out / 'summary.json', summary)
    print(f"COMPLETE TV={final['tv']:.5f} modes={final['modes']} "
          f"hq={final['hq']:.4f} sigma_ratio={final['sigma_ratio']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    user = read_config(args.config)
    if set(user) - set(DEFAULTS):
        raise ValueError(f'unknown config keys: {sorted(set(user) - set(DEFAULTS))}')
    train({**DEFAULTS, **user})


if __name__ == '__main__':
    main()

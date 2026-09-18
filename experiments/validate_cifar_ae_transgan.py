#!/usr/bin/env python
"""Audit completed scratch preflights and estimate scout cost without training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import train_cifar_ae_transgan as trainer
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults
from particlegan import MoGParticlePrior


@torch.no_grad()
def audit(manifest, output):
    provenance = code_provenance(trainer.__file__, sys.executable)
    defaults = trainer_defaults(trainer.__file__)
    rows = []
    for path in json.loads(manifest.read_text()):
        cfg = load_config(path, defaults)
        run = ROOT / cfg['out_dir']
        assert has_valid_summary(str(run), cfg, provenance), f'Uncertified: {run}'
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['start_step'] == 0 and summary['final']['step'] == cfg['steps']
        assert summary['frozen_features_unchanged'] and summary['sigma_unchanged']
        ckpath = run / 'checkpoint.pt'
        digest = hashlib.sha256(ckpath.read_bytes()).hexdigest()
        ck = torch.load(ckpath, map_location='cpu', weights_only=False)
        if cfg['generator_arch'] == 'transgan':
            g = trainer.TransGANGenerator(cfg['z_dim'], cfg['transgan_dim'], cfg['transgan_depths'],
                                          cfg['transgan_heads'], cfg['transgan_mlp_ratio'], cfg['transgan_rgb_gain'])
        else:
            assert cfg['g_depth'] == 1
            g = trainer.DirectGenerator(cfg['z_dim'], cfg['g_width'] or cfg['width'])
        prior = MoGParticlePrior(cfg['num_particles'], cfg['z_dim'], sigma_rel=cfg['sigma_rel'])
        g, prior = g.cuda().eval().requires_grad_(False), prior.cuda().eval().requires_grad_(False)
        probes = {}
        for tag, gkey, pkey in [('live', 'G', 'prior'), ('ema', 'ema_G', 'ema_prior')]:
            g.load_state_dict(ck[gkey]); prior.load_state_dict(ck[pkey])
            stream = torch.Generator(device='cuda').manual_seed(cfg['seed'] + 40000)
            images = torch.cat([g(prior.sample(64, stream)[0]).cpu() for _ in range(4)])
            assert torch.isfinite(images).all()
            probes[tag] = {'saturated_abs_gt_099': float((images.abs() > .99).float().mean()),
                           'pixel_mean': float(images.mean()), 'pixel_std': float(images.std()),
                           'mean_std_across_samples': float(images.std(dim=0).mean())}
        assert hashlib.sha256(ckpath.read_bytes()).hexdigest() == digest
        for state in ck['optimizer_d']['state'].values():
            assert int(state['step']) == cfg['steps'] * cfg['d_updates']
        assert all(int(s['step']) == cfg['steps'] for s in ck['optimizer_g']['state'].values())
        metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        first, last = metrics[1], metrics[-1]
        speed = (last['step'] - first['step']) / (last['train_seconds'] - first['train_seconds'])
        rows.append({'name': run.name, 'config': cfg, 'checkpoint_sha256': digest,
                     'initialization_sha256': summary['metadata']['initialization_sha256'],
                     'D_E_initialization_sha256': summary['metadata']['D_E_initialization_sha256'],
                     'rng_sha256': summary['rng_sha256'], 'parameters': summary['metadata']['parameters'],
                     'regularizer': summary['metadata']['regularizer'],
                     'train_seconds': summary['train_seconds'], 'total_seconds': summary['total_seconds'],
                     'peak_memory_gb': summary['peak_memory_gb'], 'steady_updates_per_second': speed,
                     'estimated_50k_training_hours': 50000 / speed / 3600,
                     'final_losses': {key: last[key] for key in ('d_loss', 'g_loss', 'recon_train_mse')},
                     'output_probe': probes, 'certified': True})
        del g, prior, ck
        torch.cuda.empty_cache()
    assert len({row['D_E_initialization_sha256'] for row in rows}) == 1
    assert len({json.dumps(row['rng_sha256'], sort_keys=True) for row in rows}) == 1
    by_name = {row['name']: row for row in rows}
    assert by_name['transgan_all']['initialization_sha256'] == by_name['transgan_e_only']['initialization_sha256']
    assert by_name['cnn_e_only']['initialization_sha256'] == '76e3fe984275380a6a29f9e8c424665c35d5bcfae870a0c5c9bcf7dc18bee722'
    for row in rows:
        assert row['regularizer']['every'] == 8 and row['regularizer']['applied_coeff'] == 8
    result = {'sources': provenance, 'all_certified': True, 'shared_D_E_initialization_equal': True,
              'training_rng_equal': True, 'historical_CNN_initialization_preserved': True,
              'checkpoint_hashes_unchanged': True, 'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for row in rows:
        print(row['name'], json.dumps({key: row[key] for key in
              ('steady_updates_per_second', 'estimated_50k_training_hours', 'peak_memory_gb', 'final_losses', 'output_probe')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.config_manifest, args.output)

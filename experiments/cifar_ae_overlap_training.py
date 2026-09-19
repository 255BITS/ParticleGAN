#!/usr/bin/env python
"""Checkpoint forks on GPU 1, followed by frozen endpoint quality diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline
from experiments.run_grid import code_provenance, has_valid_summary

PARENT = 'runs/cifar_particle_ae/particle_16k_80k/16k_80k/checkpoint_080000.pt'
SHA = '9ddcef1fb0bf82c47a581fa1b2d58872315d5d67b302eadc533c0abc41b7d5c7'


def main(scale, smoke):
    assert 0 < scale < 1
    track = 'particle_overlap_training' + ('_smoke' if smoke else '')
    pipeline.COMMON = ROOT/f'runs/cifar_particle_ae/{track}/PIPELINE.log'
    parent = json.loads((ROOT/'reports/cifar-particle-ae/particle_16k_80k/results.json').read_text())[0]
    assert hashlib.sha256((ROOT/PARENT).read_bytes()).hexdigest() == SHA
    assert has_valid_summary(str(ROOT/parent['config']['out_dir']), parent['config'],
                             code_provenance(str(ROOT/'experiments/train_cifar_ae_scaling.py'), sys.executable))
    original = torch.load(ROOT/PARENT, map_location='cpu', weights_only=False) if smoke else None
    report = ROOT/f'reports/cifar-particle-ae/{track}'
    results, checkpoints = [], {}
    for name, intervention in [('freeze_centers', {'freeze_prior': True, 'noise_scale': 1.}),
                               ('reduced_noise', {'freeze_prior': False, 'noise_scale': scale})]:
        cfg = {**parent['config'], **intervention, 'steps': 80016 if smoke else 100000,
               'eval_interval': 80016 if smoke else 5000, 'initial_eval_samples': 0,
               'eval_samples': 128 if smoke else 50000, 'final_samples': 128 if smoke else 50000,
               'recon_samples': 64 if smoke else 10000, 'keep_checkpoints': True,
               'max_train_seconds': 7200., 'resume_checkpoint': PARENT, 'resume_sha256': SHA,
               'out_dir': f'runs/cifar_particle_ae/{track}_{name}/{name}'}
        r = pipeline.grid(track+'_'+name, 'experiments/train_cifar_ae_overlap.py', '1', {name: cfg})[0]
        out = ROOT/cfg['out_dir']
        assert r['start_step'] == 80000 and r['final']['step'] == cfg['steps']
        assert r['frozen_features_unchanged'] and r['sigma_unchanged']
        if name == 'freeze_centers':
            assert r['prior_frozen_unchanged']
        rows = [json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
        assert all(m['learning_rates'] == {'G': .0003, 'E': .0003, 'prior': .003, 'D': .00045} for m in rows)
        curve = [m for m in rows if 'generation' in m]
        expected = [80016] if smoke else [85000, 90000, 95000, 100000]
        assert [m['step'] for m in curve] == expected
        assert all('latent_geometry' in m for m in curve)
        checkpoint = out/f"checkpoint_{cfg['steps']:06d}.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoints[name] = {'path': str(checkpoint.relative_to(ROOT)), 'sha256': digest, 'fid50k': r['final']['fid']}
        if smoke:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            for key in ('G', 'D', 'E'):
                assert any(not torch.equal(v, original[key][k]) for k, v in saved[key].items() if torch.is_tensor(v)), key
            for key in ('prior', 'ema_prior'):
                assert torch.equal(saved[key]['sigma'], original[key]['sigma']*intervention['noise_scale'])
                assert torch.equal(saved[key]['z'], original[key]['z']) == intervention['freeze_prior']
            if intervention['freeze_prior']:
                prior_id = original['optimizer_g']['param_groups'][2]['params'][0]
                assert all(torch.equal(v, saved['optimizer_g']['state'][prior_id][k])
                           for k, v in original['optimizer_g']['state'][prior_id].items())
                assert all(m['prior_update_rms'] == 0 for m in rows)
            print('SMOKE_VERIFIED', name, 'G/D/E update; live/EMA prior and optimizer invariants pass', flush=True)
        result = {'name': name, 'curve': curve, **r}
        results.append(result)
        pipeline.write(report/'results.json', results)
        pipeline.write(report/'CHECKPOINTS.json', checkpoints)
        if not smoke:
            probe_cfg = dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest,
                             expected_fid=r['final']['fid'], noise_scales=[1.],
                             out_dir=f'runs/cifar_particle_ae/{track}_quality_{name}/{name}')
            q = pipeline.grid(track+'_quality_'+name, 'experiments/probe_cifar_ae_overlap.py', '1', {name: probe_cfg})[0]
            result['endpoint_quality'] = q
            pipeline.write(report/'results.json', results)
        table = ['# Particle overlap checkpoint interventions', '',
                 '| Arm | Step | FID | Median nearest distance | Latent confusion |', '|---|---:|---:|---:|---:|']
        for result in results:
            for row in result['curve']:
                geometry = row['latent_geometry']
                table.append(f"| {result['name']} | {row['step']} | {row['generation']['fid']:.4f} | {geometry['median_nearest_distance']:.4f} | {geometry['wrong_nearest_fraction']:.4%} |")
        table += ['', 'Existing unchanged control: 80k 15.7527, 90k 15.7901, 100k 16.4609.',
                  'Same parent, learning rates, seed and objective. Freeze arm fixes live and EMA centers separately; prior optimizer state is retained but gets no updates.',
                  'Reduced-noise arm changes sigma for training and evaluation; initial sampling effects must be separated from learning.',
                  'No automatic continuation beyond 100k.']
        if smoke:
            table.insert(0, 'SMOKE ONLY: FID128 is not a benchmark.\n')
        (report/'LEADERBOARD.md').write_text('\n'.join(table)+'\n')
        print('\n'.join(table), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    main(args.noise_scale, args.smoke)

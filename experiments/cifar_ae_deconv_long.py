#!/usr/bin/env python
"""Continue the certified plain deconv checkpoint to 200k on GPU 1."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline
from experiments.run_grid import code_provenance, has_valid_summary

TRACK = 'deconv_16k_200k'
ARM = 'deconv_16k_no_norm'
TRAINER = 'experiments/train_cifar_ae_deconv.py'
pipeline.COMMON = ROOT / f'runs/cifar_particle_ae/{TRACK}/PIPELINE.log'


def prepare():
    parent_report = ROOT / 'reports/cifar-particle-ae/deconv_16k_scout'
    parent = json.loads((parent_report / 'results.json').read_text())[0]
    assert has_valid_summary(str(ROOT / parent['config']['out_dir']), parent['config'],
                             code_provenance(str(ROOT / TRAINER), sys.executable))
    checkpoint = json.loads((parent_report / 'CHECKPOINTS.json').read_text())['final']
    assert hashlib.sha256((ROOT / checkpoint['path']).read_bytes()).hexdigest() == checkpoint['sha256']
    cfg = {**parent['config'], 'steps': 200000, 'eval_interval': 10000,
           'eval_samples': 50000, 'final_samples': 50000, 'initial_eval_samples': 0,
           'max_train_seconds': 18000., 'keep_checkpoints': True,
           'resume_checkpoint': checkpoint['path'], 'resume_sha256': checkpoint['sha256'],
           'out_dir': f'runs/cifar_particle_ae/{TRACK}/{ARM}'}
    return parent, checkpoint, cfg


def run():
    parent, parent_checkpoint, cfg = prepare()
    r = pipeline.grid(TRACK, TRAINER, '1', {ARM: cfg})[0]
    assert r['start_step'] == 40000 and r['final']['step'] == 200000
    assert r['frozen_features_unchanged'] and r['sigma_unchanged']
    out = ROOT / cfg['out_dir']
    assert json.loads((out / 'resume.json').read_text())['interventions'] == {}
    metrics = [json.loads(x) for x in (out / 'metrics.jsonl').read_text().splitlines()]
    curve = [x for x in metrics if 'generation' in x]
    assert [x['step'] for x in curve] == list(range(50000, 200001, 10000))
    assert all(x['generation']['samples'] == 50000 for x in curve)
    assert all(x['learning_rates'] == dict(G=.0003, E=.0003, prior=.003, D=.00045) for x in metrics)
    report = ROOT / 'reports/cifar-particle-ae' / TRACK
    pipeline.write(report / 'results.json', [{'name': ARM, 'curve': curve, **r}])
    combined = [*parent['curve'], *curve]
    best = min(combined, key=lambda x: x['generation']['fid'])
    checkpoints = {'parent': parent_checkpoint}
    for name, step, fid in [('final', 200000, r['final']['fid']),
                            ('best_sampled', best['step'], best['generation']['fid'])]:
        folder = out if step > 40000 else ROOT / parent['config']['out_dir']
        path = folder / f'checkpoint_{step:06d}.pt'
        checkpoints[name] = dict(path=str(path.relative_to(ROOT)), step=step, fid50k=fid,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    pipeline.write(report / 'CHECKPOINTS.json', checkpoints)
    lines = ['# Plain deconv 16k, continuation to 200k', '',
             '| Step | FID50k |', '|---|---:|']
    lines += [f"| {x['step']} | {x['generation']['fid']:.4f} |" for x in combined]
    lines += ['', f"Best sampled: {best['generation']['fid']:.4f} at {best['step']}; "
              f"final: {r['final']['fid']:.4f}. Continuation train minutes: {r['train_seconds']/60:.2f}.",
              '', 'Full-state resume, unchanged recipe and constant learning rates. '
              'No further stage queued. Historical CNN best FID50k15.7527 at80k uses a different prior initialization history.']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    late = curve[-1]['generation']['fid'] - curve[-3]['generation']['fid']
    interpretation = ('FID improved over the last20k; a capacity ceiling is not established.' if late < 0
                      else 'FID did not improve over the last20k; retain the best checkpoint and investigate optimization/capacity before more training.')
    (report / 'FINDINGS.md').write_text('# Interpretation\n\n' + interpretation + '\n\n'
        'A flat or worsening FID curve alone cannot distinguish insufficient generator capacity, '
        'optimization dynamics, critic limitations, or partial mode loss. Inspect samples and coverage. '
        'No automatic promotion.\n')
    print('\n'.join(lines), flush=True)
    print(interpretation, flush=True)


if __name__ == '__main__':
    run()

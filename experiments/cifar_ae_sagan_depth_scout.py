#!/usr/bin/env python
"""Expand the certified SAGAN 200k checkpoint to two G and D attention blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline
from experiments.run_grid import code_provenance, has_valid_summary
from experiments import train_cifar_ae_sagan_depth as trainer

TRACK = 'sagan_gd_depth2_240k'
ARM = 'gd_depth2'
TRAINER = 'experiments/train_cifar_ae_sagan_depth.py'
REPORT = ROOT / 'reports/cifar-particle-ae' / TRACK
PARENT_REPORT = ROOT / 'reports/cifar-particle-ae/sagan_gd_16k_200k'
CONTROL = ROOT / 'runs/cifar_particle_ae/sagan_gd_16k_300k/sagan_gd_16k'


def prepare(mode):
    parent = json.loads((PARENT_REPORT / 'results.json').read_text())[0]
    assert has_valid_summary(str(ROOT / parent['config']['out_dir']), parent['config'],
                             code_provenance(str(ROOT / 'experiments/train_cifar_ae_sagan.py'), sys.executable))
    checkpoint = json.loads((PARENT_REPORT / 'CHECKPOINTS.json').read_text())['final']
    assert checkpoint['step'] == 200000
    track = TRACK + ('' if mode == 'run' else '_' + mode)
    cfg = {**parent['config'], 'g_attention_depth': 2, 'd_attention_depth': 2,
           'steps': 240000, 'max_train_seconds': 21600., 'keep_checkpoints': True,
           'resume_checkpoint': checkpoint['path'], 'resume_sha256': checkpoint['sha256'],
           'out_dir': f'runs/cifar_particle_ae/{track}/{ARM}', 'initial_eval_samples': 50000}
    if mode != 'run':
        cfg.update(steps=200016, eval_interval=200016, log_interval=8,
                   initial_eval_samples=128, eval_samples=128, final_samples=128, recon_samples=64)
    if mode == 'resume_smoke':
        smoke_report = ROOT / f'reports/cifar-particle-ae/{TRACK}_smoke/results.json'
        previous = json.loads(smoke_report.read_text())[0]
        assert has_valid_summary(str(ROOT / previous['config']['out_dir']), previous['config'],
                                 code_provenance(str(ROOT / TRAINER), sys.executable))
        path = ROOT / previous['config']['out_dir'] / 'checkpoint_200016.pt'
        cfg.update(resume_checkpoint=str(path.relative_to(ROOT)),
                   resume_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   steps=200024, eval_interval=200024, initial_eval_samples=0)
    trainer.validate(cfg)
    ck, audit = trainer.load_resume(cfg)
    expected = {} if mode == 'resume_smoke' else {
        key: {'before': 1, 'after': 2} for key in ('g_attention_depth', 'd_attention_depth')}
    assert audit['interventions'] == expected
    pipeline.write(ROOT / 'reports/cifar-particle-ae' / track / 'VALIDATION.json', {
        'parent_certified': True, 'parent_checkpoint': checkpoint, 'resume_audit': audit,
        'config_changes': {k: {'before': parent['config'].get(k), 'after': v}
                           for k, v in cfg.items() if v != parent['config'].get(k)}})
    return track, cfg, parent, checkpoint


def run(mode):
    track, cfg, parent, parent_checkpoint = prepare(mode)
    report = ROOT / 'reports/cifar-particle-ae' / track
    pipeline.COMMON = ROOT / f'runs/cifar_particle_ae/{track}/PIPELINE.log'
    if mode == 'run':
        validation = json.loads((REPORT / 'TESTS.json').read_text())
        assert validation['passed']
        for suffix in ('smoke', 'resume_smoke'):
            result = json.loads((ROOT / f'reports/cifar-particle-ae/{TRACK}_{suffix}/results.json').read_text())[0]
            assert has_valid_summary(str(ROOT / result['config']['out_dir']), result['config'],
                                     code_provenance(str(ROOT / TRAINER), sys.executable))
    pipeline.write(report / 'STATUS.json', {'status': 'running', 'gpu': 1,
                                           'start_step': 200016 if mode == 'resume_smoke' else 200000,
                                           'target_step': cfg['steps']})
    r = pipeline.grid(track, TRAINER, '1', {ARM: cfg})[0]
    assert r['start_step'] == (200016 if mode == 'resume_smoke' else 200000)
    assert r['final']['step'] == cfg['steps'] and r['frozen_features_unchanged'] and r['sigma_unchanged']
    out = ROOT / cfg['out_dir']
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    curve = [x for x in metrics if 'generation' in x]
    assert all(x['generation']['samples'] == cfg['final_samples'] for x in curve)
    assert all(x['learning_rates'] == dict(G=.0003, E=.0003, prior=.003, D=.00045) for x in metrics)
    for key in ('G_attention_extra', 'D_attention_extra'):
        assert any(row['gradient_norms'][key] > 0 for row in metrics)
    resume = json.loads((out / 'resume.json').read_text())
    if mode != 'resume_smoke':
        assert set(resume['attention_growth']) == {'G', 'D'}
        assert all(v['added_parameter_count'] == 5120 for v in resume['attention_growth'].values())
    else:
        assert resume['interventions'] == {} and 'attention_growth' not in resume
    pipeline.write(report / 'results.json', [{'name': ARM, 'curve': curve, **r}])
    if mode != 'run':
        pipeline.write(report / 'STATUS.json', {'status': 'complete', 'validation_only': True})
        print(f'{mode} certified: full-state growth/resume and finite learning in both added blocks', flush=True)
        return
    assert [x['step'] for x in curve] == list(range(205000, 240001, 5000))
    initial = json.loads((out / 'initial_evaluation.json').read_text())
    assert abs(initial['fid'] - parent_checkpoint['fid50k']) < .01, initial
    control_provenance = json.loads((CONTROL / 'provenance.json').read_text())
    for name, sha in control_provenance['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    control = {x['step']: x for x in
               (json.loads(line) for line in (CONTROL / 'metrics.jsonl').read_text().splitlines())
               if 'generation' in x}
    best = min(curve, key=lambda x: x['generation']['fid'])
    checkpoints = {'parent': parent_checkpoint}
    for label, row in [('final', curve[-1]), ('best_expanded', best)]:
        path = out / f"checkpoint_{row['step']:06d}.pt"
        checkpoints[label] = dict(path=str(path.relative_to(ROOT)), step=row['step'],
                                  fid50k=row['generation']['fid'], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    checkpoints['best_overall'] = min(checkpoints.values(), key=lambda x: x['fid50k'])
    pipeline.write(report / 'CHECKPOINTS.json', checkpoints)
    lines = ['# SAGAN G/D attention depth scout', '',
             '| Step | Two G/D blocks FID50k | Existing one-block control | Difference |',
             '|---|---:|---:|---:|']
    for row in curve:
        step, fid = row['step'], row['generation']['fid']
        ref = control[step]['generation']['fid']
        lines.append(f'| {step} | {fid:.4f} | {ref:.4f} | {fid-ref:+.4f} |')
    lines += ['', f"Parent 200k: {parent_checkpoint['fid50k']:.4f}. Best expanded: {best['generation']['fid']:.4f} at {best['step']}. "
              f"Training minutes {r['train_seconds']/60:.2f}; wall minutes {r['total_seconds']/60:.2f}."]
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    fid = r['final']['fid']
    delta = fid - control[240000]['generation']['fid']
    beats_parent = best['generation']['fid'] < parent_checkpoint['fid50k']
    recommendation = ('Consider extending the expanded model after reviewing the late curve and sample grids.'
                      if beats_parent and delta < 0 else
                      'Keep the best overall checkpoint; review whether gains justify added compute before extending.')
    (report / 'FINDINGS.md').write_text(
        '# Attention depth result\n\n'
        f'At 240k, expanded minus unchanged FID50k: {delta:+.4f}. '
        f'Best expanded beats the 200k parent: {beats_parent}.\n\n'
        'This is a joint G/D capacity intervention from the same full-state checkpoint. Existing weights, optimizer moments, '
        'EMA, prior and RNG were restored; new attention blocks began as identities with fresh Adam state. '
        'It does not separate G and D contributions or establish how the architecture would perform from scratch. '
        'The unchanged control was user-stopped after 260k; its matched 205k–240k evaluations and source hashes are preserved.\n\n'
        + recommendation + '\n\nNo further training queued.\n')
    pipeline.write(report / 'STATUS.json', {'status': 'complete', 'final_step': 240000,
                                           'final_fid': fid, 'checkpoints': checkpoints})
    print('\n'.join(lines), flush=True)
    print(recommendation, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['run', 'smoke', 'resume_smoke'], default='run')
    args = parser.parse_args()
    try:
        run(args.mode)
    except Exception as exc:
        track = TRACK + ('' if args.mode == 'run' else '_' + args.mode)
        pipeline.write(ROOT / 'reports/cifar-particle-ae' / track / 'STATUS.json',
                       {'status': 'failed', 'error': repr(exc)})
        raise

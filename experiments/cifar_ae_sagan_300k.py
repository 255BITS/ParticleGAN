#!/usr/bin/env python
"""Continue the certified SAGAN G/D 200k checkpoint to 300k on GPU 1."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline
from experiments.run_grid import code_provenance, has_valid_summary

TRACK = 'sagan_gd_16k_300k'
ARM = 'sagan_gd_16k'
TRAINER = 'experiments/train_cifar_ae_sagan.py'
REPORT = ROOT / 'reports/cifar-particle-ae' / TRACK
pipeline.COMMON = ROOT / f'runs/cifar_particle_ae/{TRACK}/PIPELINE.log'


def prepare():
    folder = ROOT / 'reports/cifar-particle-ae/sagan_gd_16k_200k'
    parent = json.loads((folder / 'results.json').read_text())[0]
    assert has_valid_summary(str(ROOT / parent['config']['out_dir']), parent['config'],
                             code_provenance(str(ROOT / TRAINER), sys.executable))
    checkpoint = json.loads((folder / 'CHECKPOINTS.json').read_text())['final']
    assert checkpoint['step'] == 200000
    assert hashlib.sha256((ROOT / checkpoint['path']).read_bytes()).hexdigest() == checkpoint['sha256']
    cfg = {**parent['config'], 'steps': 300000,
           'max_train_seconds': 21600., 'keep_checkpoints': True,
           'resume_checkpoint': checkpoint['path'], 'resume_sha256': checkpoint['sha256'],
           'out_dir': f'runs/cifar_particle_ae/{TRACK}/{ARM}'}
    from experiments.train_cifar_ae_sagan import load_resume, validate
    validate(cfg)
    ck, audit = load_resume(cfg)
    assert ck['step'] == 200000 and audit['interventions'] == {}
    assert cfg['eval_interval'] == 5000
    assert cfg['eval_samples'] == cfg['final_samples'] == 50000
    pipeline.write(REPORT / 'VALIDATION.json', {
        'parent_summary_certified': True, 'parent_checkpoint': checkpoint,
        'resume_audit': audit, 'config_changes': {
            k: {'before': parent['config'].get(k), 'after': v}
            for k, v in cfg.items() if v != parent['config'].get(k)},
        'note': 'Unmodified historical trainer; full-state resume preflight. No new seed or control run.'})
    return parent, checkpoint, cfg


def run():
    parent, parent_checkpoint, cfg = prepare()
    pipeline.write(REPORT / 'STATUS.json', {'status': 'running', 'gpu': 1,
                                           'start_step': 200000, 'target_step': 300000})
    r = pipeline.grid(TRACK, TRAINER, '1', {ARM: cfg})[0]
    assert r['start_step'] == 200000 and r['final']['step'] == 300000
    assert r['frozen_features_unchanged'] and r['sigma_unchanged']
    out = ROOT / cfg['out_dir']
    assert json.loads((out / 'resume.json').read_text())['interventions'] == {}
    metrics = [json.loads(x) for x in (out / 'metrics.jsonl').read_text().splitlines()]
    curve = [x for x in metrics if 'generation' in x]
    assert [x['step'] for x in curve] == list(range(205000, 300001, 5000))
    assert all(x['generation']['samples'] == 50000 for x in curve)
    assert all(x['learning_rates'] == dict(G=.0003, E=.0003, prior=.003, D=.00045) for x in metrics)
    pipeline.write(REPORT / 'results.json', [{'name': ARM, 'curve': curve, **r}])
    scout = json.loads((ROOT / 'reports/cifar-particle-ae/sagan_gd_16k_scout/results.json').read_text())[0]
    combined = [*scout['curve'], *parent['curve'], *curve]
    best = min(combined, key=lambda x: x['generation']['fid'])
    checkpoints = {'parent': parent_checkpoint}
    for name, step, fid in [('final', 300000, r['final']['fid']),
                            ('best_sampled', best['step'], best['generation']['fid'])]:
        source = parent if step > 40000 else scout
        folder = out if step > 200000 else ROOT / source['config']['out_dir']
        path = folder / f'checkpoint_{step:06d}.pt'
        checkpoints[name] = dict(path=str(path.relative_to(ROOT)), step=step, fid50k=fid,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    pipeline.write(REPORT / 'CHECKPOINTS.json', checkpoints)
    lines = ['# SAGAN G/D 16k, unchanged continuation to 300k', '',
             '| Step | FID50k |', '|---|---:|']
    lines += [f"| {x['step']} | {x['generation']['fid']:.4f} |" for x in combined]
    lines += ['', f"Best sampled: {best['generation']['fid']:.4f} at {best['step']}; "
              f"final: {r['final']['fid']:.4f}. Continuation train minutes: {r['train_seconds']/60:.2f}.",
              '', 'Existing wide GN deconv best45k: 17.4961; endpoint80k: 17.6565. '
              'Historical residual CNN best80k: 15.7527, with different prior initialization history. '
              'These are contextual references, not matched 300k architecture comparisons.']
    (REPORT / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    late = curve[-1]['generation']['fid'] - curve[-5]['generation']['fid']
    verdict = ('FID improved over the last 20k steps.' if late < 0
               else 'FID did not improve over the last 20k steps.')
    recommendation = ('Review the late trajectory and samples before choosing a further extension.' if late < 0
                      else 'Retain the best checkpoint; investigate the regression or plateau before further training.')
    (REPORT / 'FINDINGS.md').write_text('# Duration result\n\n' + verdict +
        f" Endpoint minus280k FID: {late:+.4f}. Endpoint minus200k: "
        f"{r['final']['fid']-parent['final']['fid']:+.4f}.\n\n" + recommendation +
        '\n\nOne continued trajectory tests duration with the same recipe. It does not isolate G versus D attention, '
        'establish statistical significance, or distinguish capacity from optimization and distribution coverage. '
        'BigGAN evaluation protocol parity remains unverified. No further training queued.\n')
    pipeline.write(REPORT / 'STATUS.json', {'status': 'complete', 'final_step': 300000,
                                           'final_fid': r['final']['fid'], 'checkpoints': checkpoints})
    print('\n'.join(lines), flush=True)
    print(verdict, recommendation, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        if args.preflight:
            prepare()
            print('PREFLIGHT certified: unchanged 200k full-state checkpoint -> 300k', flush=True)
        else:
            run()
    except Exception as exc:
        pipeline.write(REPORT / 'STATUS.json', {'status': 'failed', 'error': repr(exc)})
        raise

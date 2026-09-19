#!/usr/bin/env python
"""Continue the certified 16k/32k endpoints, then probe each new endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline
from experiments.run_grid import code_provenance, has_valid_summary

pipeline.COMMON = ROOT / 'runs/cifar_particle_ae/particle_duration/PIPELINE.log'
ARMS = {
    '16k_160k': dict(parent_track='particle_16k_80k', parent_arm='16k_80k',
                    stop=160000, gpu='0', trainer='experiments/train_cifar_ae_scaling.py',
                    probe='experiments/probe_cifar_ae_information.py'),
    '32k_80k': dict(parent_track='particle_32k_40k', parent_arm='32k_40k',
                   stop=80000, gpu='1', trainer='experiments/train_cifar_ae_scaling32.py',
                   probe='experiments/probe_cifar_ae_information32.py'),
}


def prepare(arm):
    spec = ARMS[arm]
    parent_report = ROOT / 'reports/cifar-particle-ae' / spec['parent_track']
    parent = json.loads((parent_report / 'results.json').read_text())[0]
    provenance = code_provenance(str(ROOT / spec['trainer']), sys.executable)
    assert has_valid_summary(str(ROOT / parent['config']['out_dir']), parent['config'], provenance)
    checkpoint = json.loads((parent_report / 'CHECKPOINTS.json').read_text())[spec['parent_arm']]
    assert hashlib.sha256((ROOT / checkpoint['path']).read_bytes()).hexdigest() == checkpoint['sha256']
    track = 'particle_' + arm
    cfg = {**parent['config'], 'steps': spec['stop'], 'eval_interval': 10000,
           'eval_samples': 50000, 'final_samples': 50000, 'initial_eval_samples': 0,
           'keep_checkpoints': True, 'max_train_seconds': 10800.,
           'resume_checkpoint': checkpoint['path'], 'resume_sha256': checkpoint['sha256'],
           'out_dir': f'runs/cifar_particle_ae/{track}/{arm}'}
    folder = ROOT / 'configs/cifar_particle_ae' / track
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{arm}.yaml'
    path.write_text(yaml.safe_dump(cfg))
    pipeline.write(folder / 'manifest.json', [str(path.relative_to(ROOT))])
    return spec, parent, track, cfg


def run(arm):
    spec, parent, track, cfg = prepare(arm)
    r = pipeline.grid(track, spec['trainer'], spec['gpu'], {arm: cfg})[0]
    out = ROOT / cfg['out_dir']
    assert r['start_step'] == parent['final']['step'] and r['final']['step'] == cfg['steps']
    assert r['frozen_features_unchanged'] and r['sigma_unchanged']
    assert json.loads((out / 'resume.json').read_text())['interventions'] == {}
    assert not json.loads((out / 'expansion_audit.json').read_text())['intervention']
    metrics = [json.loads(x) for x in (out / 'metrics.jsonl').read_text().splitlines()]
    curve = [x for x in metrics if 'generation' in x]
    assert [x['step'] for x in curve] == list(range(r['start_step'] + 10000, cfg['steps'] + 1, 10000))
    assert all(x['generation']['samples'] == 50000 for x in curve)
    assert all(x['learning_rates'] == dict(G=.0003, E=.0003, prior=.003, D=.00045) for x in metrics)
    report = ROOT / 'reports/cifar-particle-ae' / track
    pipeline.write(report / 'results.json', [{'name': arm, 'curve': curve, **r}])
    checkpoint = out / f"checkpoint_{cfg['steps']:06d}.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    pipeline.write(report / 'CHECKPOINTS.json', {arm: {
        'path': str(checkpoint.relative_to(ROOT)), 'sha256': digest, 'fid50k': r['final']['fid']}})
    best = min(curve, key=lambda x: x['generation']['fid'])
    lines = [f'# {arm} certified continuation', '', '| Step | FID50k |', '|---|---:|',
             f"| {parent['final']['step']} (starting checkpoint) | {parent['final']['fid']:.4f} |"]
    lines += [f"| {x['step']} | {x['generation']['fid']:.4f} |" for x in curve]
    lines += ['', f"Training minutes: {r['train_seconds']/60:.2f}. Best new sampled point "
              f"{best['generation']['fid']:.4f} at {best['step']}. Final {r['final']['fid']:.4f}."]
    if arm == '32k_80k':
        baseline = json.loads((ROOT / 'reports/cifar-particle-ae/particle_16k_80k/results.json').read_text())[0]
        lines += ['', f"Matched 16k at 80k: {baseline['final']['fid']:.4f}; "
                  f"32k minus 16k: {r['final']['fid']-baseline['final']['fid']:+.4f}."]
    lines += ['', 'No further training promotion queued. Keep endpoint and best sampled point distinct.']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    probe_track = track + '_information'
    probe_cfg = {'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': digest,
                 'out_dir': f'runs/cifar_particle_ae/{probe_track}/{arm}'}
    q = pipeline.grid(probe_track, spec['probe'], spec['gpu'], {arm: probe_cfg})[0]
    assert q['parent_unchanged'] and q['frozen_state_unchanged'] and q['parent_step'] == cfg['steps']
    assert abs(q['information']['identical_clones']['decodable_bits']) < 1e-8
    old = json.loads((parent_report_path(spec) / 'results.json').read_text())[0]
    assert q['real_features_sha256'] == old['real_features_sha256']
    probe_report = ROOT / 'reports/cifar-particle-ae' / probe_track
    pipeline.write(probe_report / 'results.json', [{'name': arm, 'checkpoint_fid50k': r['final']['fid'], **q}])
    info, quality = q['information'], q['density_coverage']
    text = (f"# {arm} endpoint diagnostics\n\nFID50k {r['final']['fid']:.4f}; "
            f"sibling information {info['observed']['decodable_bits']:.4f}/{info['available_sibling_bits']:.0f} bits; "
            f"density {quality['density']:.4f}; coverage {quality['coverage']:.2%}.\n\n"
            'Read-only certified probe; same real feature reference. Information is a restricted-decoder '
            'estimate, not exact entropy or semantic coverage. Review controls and per-parent results.\n')
    (probe_report / 'LEADERBOARD.md').write_text(text)
    print(text, flush=True)


def parent_report_path(spec):
    return ROOT / 'reports/cifar-particle-ae' / (spec['parent_track'] + '_information')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=list(ARMS), required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.arm)
    else:
        run(args.arm)

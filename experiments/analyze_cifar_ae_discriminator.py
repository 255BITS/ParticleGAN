#!/usr/bin/env python
"""Verify and summarize D-only diagnostic pipeline artifacts."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def analyze(track):
    trainer = str(ROOT / 'experiments/diagnose_cifar_ae_discriminator.py')
    defaults = trainer_defaults(trainer)
    provenance = code_provenance(trainer, sys.executable)
    paths = json.loads((ROOT / f'configs/cifar_particle_ae/{track}/manifest.json').read_text())
    rows = []
    for path in paths:
        cfg = load_config(path, defaults)
        run = ROOT / cfg['out_dir']
        assert has_valid_summary(str(run), cfg, provenance), run
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['frozen_state_unchanged'] and summary['parent_unchanged']
        assert summary['optimizer_d_steps_after'] - summary['optimizer_d_steps_before'] == cfg['steps']
        rows.append({'name': run.name, **summary,
                     'curve': [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]})
    matched = [r for r in rows if r['config']['steps']]
    assert len({r['initial_D_sha256'] for r in matched}) == 1
    assert len({r['config']['checkpoint_sha256'] for r in matched}) == 1
    report = ROOT / f'reports/cifar-particle-ae/{track}'
    report.mkdir(parents=True, exist_ok=True)
    (report / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['# Discriminator diagnosis', '', f'{len(rows)}/{len(paths)} runs certified. Higher AUC means better real/fake ranking; it is not FID.', '',
             '| Run | D-only updates | Test AUC | Train AUC | Fake image gradient | G adversarial gradient | Pixel/features cosine | Total / sum gradient norms | Train seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        for point in [r['baseline'], r['final']] if r['config']['steps'] else [r['final']]:
            t=point['test']; b=t['branches']['total']
            lines.append(f"| {r['name']} | {point['step']} | {b['auc']:.4f} | {point['train']['branches']['total']['auc']:.4f} | {b['fake_grad_mean']:.4f} | {t['adv_G_norm']:.4f} | {t['pixel_features_cosine']:.4f} | {t['total_over_sum_branch_grad_norm']:.4f} | {point['train_seconds']:.1f} |")
    lines += ['', 'Frozen G/E/prior/features, original checkpoints and D optimizer step counts verified. Fixed diagnostic draws are independent of training; CIFAR test split is never trained on. All D-only arms start with identical D state and the same parent training RNG. Production CUDA allows small floating-point differences. No seed replicates.', '',
              'AUC/gradient changes require a joint FID continuation to establish value. D-only warmup does not change the frozen generator FID. Detailed branch scores, cap activation and all evaluation points are in results.json.']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--track', default='discriminator_diagnosis');a=p.parse_args();analyze(a.track)

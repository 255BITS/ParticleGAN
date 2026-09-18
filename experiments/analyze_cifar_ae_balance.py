#!/usr/bin/env python
"""Certify matched G-only LR continuations, audit actual rates, and report FID."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def analyze(track):
    trainer = str(ROOT / 'experiments/train_cifar_ae_balance.py')
    defaults = trainer_defaults(trainer)
    provenance = code_provenance(trainer, sys.executable)
    paths = json.loads((ROOT / f'configs/cifar_particle_ae/{track}/manifest.json').read_text())
    rows, checkpoints = [], {}
    for path in paths:
        cfg = load_config(path, defaults)
        run = ROOT / cfg['out_dir']
        assert has_valid_summary(str(run), cfg, provenance), run
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['frozen_features_unchanged'] and summary['sigma_unchanged']
        assert summary['start_step'] == 10000 and summary['final']['step'] == cfg['steps']
        assert summary['final']['samples'] == cfg['final_samples']
        metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        expected = dict(G=cfg['lr'] * cfg['g_lr_scale'], E=cfg['lr'], prior=cfg['prior_lr'], D=cfg['d_lr'])
        assert all(row['learning_rates'] == expected for row in metrics)
        resume = json.loads((run / 'resume.json').read_text())
        assert resume['interventions'] == ({} if cfg['g_lr_scale'] == 1 else {'g_lr_scale': {'before': 1., 'after': cfg['g_lr_scale']}})
        rows.append({'name': run.name, 'curve': [m for m in metrics if 'generation' in m], **summary})
        ckpath = run / 'checkpoint.pt'
        checkpoints[run.name] = {'path': str(ckpath.relative_to(ROOT)), 'sha256': hashlib.sha256(ckpath.read_bytes()).hexdigest()}
    assert len({r['config']['resume_sha256'] for r in rows}) == 1
    assert len({json.dumps(r['rng_sha256'], sort_keys=True) for r in rows}) == 1
    configs = [{k:v for k,v in r['config'].items() if k not in ('out_dir', 'g_lr_scale')} for r in rows]
    assert all(c == configs[0] for c in configs)
    report = ROOT / f'reports/cifar-particle-ae/{track}'
    report.mkdir(parents=True, exist_ok=True)
    (report / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    (report / 'CHECKPOINTS.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
    if track.endswith('smoke'):
        import torch
        counters = {}
        for r in rows:
            ck = torch.load(ROOT / r['config']['out_dir'] / 'checkpoint.pt', map_location='cpu', weights_only=False)
            counters[r['name']] = {key: sorted({float(v['step']) for v in ck[key]['state'].values()}) for key in ('optimizer_g','optimizer_d')}
            assert all(values == [10008.] for values in counters[r['name']].values())
        (report/'VALIDATION.json').write_text(json.dumps({'certified':len(rows),'optimizer_counters':counters,'rates_correct':True,'matched_rng':True}, indent=2)+'\n')
        print(json.dumps(counters)); return
    rows.sort(key=lambda r:r['final']['fid'])
    lines = ['# Generator learning-rate scout', '', f'{len(rows)}/{len(paths)} certified. Same CNN E-only 10k parent: FID50k 19.4482.', '',
             '| Arm | FID at 15k | FID at 20k | Test MSE | Training minutes | Wall minutes |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['curve'][0]['generation']['fid']:.4f} | {r['final']['fid']:.4f} | {r['final']['reconstruction']['recon_mse']:.5f} | {r['train_seconds']/60:.2f} | {r['total_seconds']/60:.2f} |")
    control = next(r for r in rows if r['name'] == 'control')
    lines += ['', 'Final FID changes versus matched control: ' + ', '.join(f"{r['name']} {r['final']['fid'] - control['final']['fid']:+.4f}" for r in rows if r['name'] != 'control') + '.', '',
              'Only G learning rate changes (0.0003 to 0.00015). E 0.0003, prior 0.003, D 0.00045; one D step, coefficient 1, lazy bcap every 8. Full Adam/EMA/RNG restored. Actual optimizer rates and matched RNG consumption audited. FID uses 50k EMA samples and the unchanged CIFAR train reference. No seed replicates or automatic long promotion.']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for r in rows:
        ax.plot([10]+[p['step']/1000 for p in r['curve']], [19.4482]+[p['generation']['fid'] for p in r['curve']], '-o', label=r['name'])
    ax.axhline(13, color='black', linestyle=':', label='target 13')
    ax.set(xlabel='Joint updates (thousands)', ylabel='FID50k')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(report/'curves.png', dpi=170); plt.close(fig)
    print('\n'.join(lines))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',default='generator_balance');a=p.parse_args();analyze(a.track)

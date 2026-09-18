#!/usr/bin/env python
"""Certify plateau interventions and publish final and intermediate FID50k."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def analyze(manifest, report, trainer_name='experiments/train_cifar_ae_plateau.py'):
    trainer = str(ROOT / trainer_name)
    provenance = code_provenance(trainer, sys.executable)
    defaults = trainer_defaults(trainer)
    rows, missing = [], []
    for path in json.loads(manifest.read_text()):
        cfg = load_config(path, defaults)
        run = ROOT / cfg['out_dir']
        if not has_valid_summary(str(run), cfg, provenance):
            missing.append(run.name); continue
        s = json.loads((run / 'summary.json').read_text())
        assert s['final']['step'] == cfg['steps'] and s['final']['samples'] == 50000
        assert math.isfinite(s['final']['fid']) and s['frozen_features_unchanged'] and s['sigma_unchanged']
        curve = []
        for line in (run/'metrics.jsonl').read_text().splitlines():
            m=json.loads(line)
            if 'generation' in m:
                assert m['generation']['samples'] == 50000
                curve.append({'step':m['step'],'fid':m['generation']['fid'],
                              'recon_mse':m['reconstruction']['recon_mse']})
        best=min(curve,key=lambda c:c['fid'])
        rows.append({'name':run.name,'curve':curve,'best_observed':{**best,
                     'checkpoint':str(run / f"checkpoint_{best['step']:06d}.pt")},**s})
    rows.sort(key=lambda r:r['final']['fid'])
    lines=['# CIFAR AE-GAN plateau experiments','',f'{len(rows)}/{len(rows)+len(missing)} certified runs complete.', '',
           '| Rank | Run | Start → end | Final FID50k ↓ | Test MSE ↓ | Train min |',
           '|---:|---|---:|---:|---:|---:|']
    for i,r in enumerate(rows,1):
        lines.append(f"| {i} | {r['name']} | {r['start_step']} → {r['final']['step']} | {r['final']['fid']:.4f} | {r['final']['reconstruction']['recon_mse']:.5f} | {r['train_seconds']/60:.2f} |")
    lines+=['','All generation FIDs use 50,000 independent prior draws and EMA G/prior; CIFAR train50k reference, TF-compatible Inception. No fitted encoder sampling is used in training benchmarks. Reconstructions use the 10k test split. Each run restores its recorded parent checkpoint and preserves the model seed; these are configuration interventions, not seed experiments.', '',
            'Historical unchanged continuation: 50k **18.9012**, 60k **19.9770**, 100k **20.8058**. The historical 60k result is the matched endpoint control for the 60k scouts. It is not an independent replication.','',
            '| Run | Step | FID50k ↓ | Test MSE ↓ |','|---|---:|---:|---:|']
    for r in rows:
        for c in r['curve']:
            lines.append(f"| {r['name']} | {c['step']} | {c['fid']:.4f} | {c['recon_mse']:.5f} |")
    lines+=['','## Recommendation','']
    if missing: lines.append('Pending or uncertified: '+', '.join(missing)+'.')
    elif rows:
        r=rows[0]
        lines.append(f"Lowest final FID: **{r['name']} ({r['final']['fid']:.4f})**. "+
                     ('Target below 13 reached.' if r['final']['fid']<13 else 'Target below 13 remains unmet.'))
        best=min((x['best_observed'] for x in rows),key=lambda c:c['fid'])
        lines.append(f"Best observed intermediate/final measurement: **{best['fid']:.4f} at {best['step']:,}**. Checkpoint: `{best['checkpoint']}`. This is selected from the evaluated curve, separate from the final-endpoint ranking.")
        if r['final']['step']==60000:
            lines.append(f"Compared with unchanged continuation at 60k: {r['final']['fid']-19.97701107479669:+.4f} FID. Use the final endpoint ranking and the 55k→60k trend to select a continuation; short scouts do not establish its 200k outcome.")
    report.mkdir(parents=True,exist_ok=True)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        historical = ROOT / 'runs/cifar_particle_ae/duration_100k/n08/metrics.jsonl'
        if historical.exists():
            old = [json.loads(v) for v in historical.read_text().splitlines()]
            old = [v for v in old if 'generation' in v]
            axes[0].plot([v['step']/1000 for v in old], [v['generation']['fid'] for v in old],
                         '--o', color='gray', label='unchanged historical')
            axes[1].plot([v['step']/1000 for v in old], [v['reconstruction']['recon_mse'] for v in old],
                         '--o', color='gray', label='unchanged historical')
        for r in rows:
            c = r['curve']
            axes[0].plot([v['step']/1000 for v in c], [v['fid'] for v in c], '-o', label=r['name'])
            axes[1].plot([v['step']/1000 for v in c], [v['recon_mse'] for v in c], '-o', label=r['name'])
        axes[0].axhline(13, color='black', linestyle=':', label='target 13')
        for ax, title in zip(axes, ('Generation FID50k (lower is better)', 'Test reconstruction MSE')):
            ax.set_title(title); ax.set_xlabel('Global updates (thousands)'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(report/'curves.png', dpi=180); plt.close(fig)
        lines += ['', '![Learning curves](curves.png)']
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
    (report/'leaderboard.json').write_text(json.dumps({'complete':not missing,'missing':missing,'rows':rows},indent=2,allow_nan=False)+'\n')
    print('\n'.join(lines),flush=True)
    return int(bool(missing))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config_manifest',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--trainer',default='experiments/train_cifar_ae_plateau.py')
    a=p.parse_args();sys.exit(analyze(a.config_manifest,a.report,a.trainer))

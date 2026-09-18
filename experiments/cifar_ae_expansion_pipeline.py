#!/usr/bin/env python
"""Run a matched expansion scout through the grid, then certify and summarize."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults
TRAINER='experiments/train_cifar_ae_expansion.py'


def configs(track, smoke):
    folder=ROOT/f'configs/cifar_particle_ae/{track}';folder.mkdir(parents=True,exist_ok=True)
    parent=yaml.safe_load((ROOT/'configs/cifar_particle_ae/generator_balance/control.yaml').read_text())
    paths=[]
    for name,factor in [('control_1024',1),('split_4096',4)]:
        cfg={**parent,'expansion_factor':factor,'initial_eval_samples':128 if smoke else 50000,
             'out_dir':f'runs/cifar_particle_ae/{track}/{name}'}
        if smoke:cfg.update(steps=10008,eval_interval=10008,eval_samples=128,final_samples=128,recon_samples=64)
        path=folder/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg));paths.append(str(path.relative_to(ROOT)))
    manifest=folder/'manifest.json';manifest.write_text(json.dumps(paths,indent=2)+'\n')
    return manifest


def analyze(track):
    defaults=trainer_defaults(str(ROOT/TRAINER));provenance=code_provenance(str(ROOT/TRAINER),sys.executable)
    paths=json.loads((ROOT/f'configs/cifar_particle_ae/{track}/manifest.json').read_text());rows=[]
    for path in paths:
        cfg=load_config(path,defaults);run=ROOT/cfg['out_dir']
        assert has_valid_summary(str(run),cfg,provenance),run
        summary=json.loads((run/'summary.json').read_text())
        assert summary['frozen_features_unchanged'] and summary['sigma_unchanged']
        assert summary['start_step']==10000 and summary['final']['step']==cfg['steps']
        initial=json.loads((run/'initial_evaluation.json').read_text())
        if cfg['initial_eval_samples']==50000:assert abs(initial['fid']-19.4482)<.01,initial
        metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        rates=dict(G=.0003,E=.0003,prior=.003,D=.00045)
        assert all(m['learning_rates']==rates for m in metrics)
        ckpath=run/'checkpoint.pt'
        rows.append({'name':run.name,'initial':initial,'curve':[m for m in metrics if 'generation' in m],
                     'checkpoint_sha256':hashlib.sha256(ckpath.read_bytes()).hexdigest(),**summary})
    assert rows[0]['rng_sha256']==rows[1]['rng_sha256'],'original data/noise RNG pairing changed'
    report=ROOT/f'reports/cifar-particle-ae/{track}';report.mkdir(parents=True,exist_ok=True)
    (report/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    context = ('SMOKE ONLY: 128-image FID is not a benchmark.' if track.endswith('smoke') else 'Same E-only 10k parent; initial FID50k 19.4482.')
    lines=['# Particle expansion scout','','2/2 certified. ' + context,'',
           '| Arm | Initial FID | Midpoint FID | Final FID | Train minutes | Wall minutes | Sibling latent RMS |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['final']['fid']):
        lines.append(f"| {r['name']} | {r['initial']['fid']:.4f} | {r['curve'][0]['generation']['fid']:.4f} | {r['final']['fid']:.4f} | {r['train_seconds']/60:.2f} | {r['total_seconds']/60:.2f} | {r['curve'][-1]['descendants']['sibling_latent_rms']:.5f} |")
    lines+=['','Cloned centers preserve initial normalization and fixed sigma. Per-row Adam moments copied; rates unchanged. Reference-count variance/covariance correction preserves the initial particle regularizer. Additional centers change sampling exposure and optimizer dynamics. Paired original data/noise streams audited. Sibling distances and image variation are not semantic coverage metrics.']
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    track='particle_expansion_smoke' if a.smoke else 'particle_expansion_scout'
    if not a.analyze_only:
        manifest=configs(track,a.smoke);run=f'runs/cifar_particle_ae/{track}'
        subprocess.run([sys.executable,'-u','experiments/follow_grid.py','--root',run,'--log',f'{run}/PIPELINE.log','--',
                        '--config_manifest',str(manifest),'--gpus','0,1','--workers_per_gpu','1','--python',sys.executable,'--trainer',TRAINER],cwd=ROOT,check=True)
    analyze(track)

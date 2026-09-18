#!/usr/bin/env python
"""Verify matched joint D interventions and plot FID trajectories."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_grid import code_provenance,has_valid_summary,load_config,trainer_defaults


def analyze(track):
 trainer=str(ROOT/'experiments/train_cifar_ae_discriminator.py')
 defaults=trainer_defaults(trainer); provenance=code_provenance(trainer,sys.executable)
 paths=json.loads((ROOT/f'configs/cifar_particle_ae/{track}/manifest.json').read_text())
 rows=[]
 for path in paths:
  cfg=load_config(path,defaults);run=ROOT/cfg['out_dir']
  assert has_valid_summary(str(run),cfg,provenance),run
  s=json.loads((run/'summary.json').read_text())
  assert s['frozen_features_unchanged'] and s['sigma_unchanged'] and s['start_step']==10000
  assert s['final']['step']==cfg['steps'] and s['final']['samples']==cfg['final_samples']
  curve=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if '"generation"' in line]
  rows.append({'name':run.name,'curve':curve,**s})
 assert len({r['config']['resume_sha256'] for r in rows})==1
 report=ROOT/f'reports/cifar-particle-ae/{track}';report.mkdir(parents=True,exist_ok=True)
 (report/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 if track.endswith('smoke'):
  import torch
  counters={}
  for r in rows:
   ck=torch.load(ROOT/r['config']['out_dir']/'checkpoint.pt',map_location='cpu',weights_only=False)
   steps={name:sorted({float(v['step']) for v in ck[name]['state'].values()}) for name in ['optimizer_g','optimizer_d']}
   assert steps['optimizer_g']==[10008.]
   assert steps['optimizer_d']==[12056. if r['name']=='warmstart' else 10008.]
   counters[r['name']]=steps
  (report/'VALIDATION.json').write_text(json.dumps({'certified':3,'optimizer_counters':counters},indent=2)+'\n')
  print(json.dumps(counters));return
 rows.sort(key=lambda r:r['final']['fid'])
 lines=['# Joint discriminator interventions','',f'{len(rows)}/{len(paths)} certified. All start from the identical CNN E-only 10k checkpoint (FID50k 19.4482).', '',
        '| Run | Final FID50k | Test reconstruction MSE | Joint train minutes | Wall minutes |',
        '|---|---:|---:|---:|---:|']
 for r in rows:
  lines.append(f"| {r['name']} | {r['final']['fid']:.4f} | {r['final']['reconstruction']['recon_mse']:.5f} | {r['train_seconds']/60:.2f} | {r['total_seconds']/60:.2f} |")
 lines+=['','Warmstart has an additional 2048 D-only updates (48.6 training seconds), original regularization thereafter. Weaker changes only bcap coefficient from1 to0.1, retaining every8 schedule. Control preserves original recipe and all training state. One D update per G update throughout joint training. No seed experiments.','',
         '| Run | Step | FID50k |', '|---|---:|---:|']
 for r in rows:
  for p in r['curve']:lines.append(f"| {r['name']} | {p['step']} | {p['generation']['fid']:.4f} |")
 control=next(r for r in rows if r['name']=='control')
 lines+=['', 'Final changes relative to matched control: '+', '.join(f"{r['name']} {r['final']['fid']-control['final']['fid']:+.4f}" for r in rows if r['name']!='control')+'.',
         '', 'FID uses EMA G/prior, 50k generated images and the unchanged cached CIFAR train50k reference. Diagnostic D-only AUC uses live weights and held-out test images, so it is a different measurement. A single continuation per intervention does not quantify stochastic run-to-run uncertainty. No automatic long promotion.']
 (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4))
 for r in rows:ax.plot([10]+[p['step']/1000 for p in r['curve']],[19.4482]+[p['generation']['fid'] for p in r['curve']],'-o',label=r['name'])
 ax.axhline(13,color='black',linestyle=':',label='target13');ax.set(xlabel='Joint updates (thousands)',ylabel='FID50k');ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(report/'curves.png',dpi=170);plt.close(fig)
 print('\n'.join(lines))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--track',default='discriminator_joint');a=p.parse_args();analyze(a.track)

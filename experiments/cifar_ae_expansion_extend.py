#!/usr/bin/env python
"""Continue both completed expansion arms and report persistence of FID gains."""
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
TRACK='particle_expansion_40k'


def prepare():
    scout=ROOT/'reports/cifar-particle-ae/particle_expansion_scout/results.json'
    rows=json.loads(scout.read_text());by_name={r['name']:r for r in rows}
    control,split=by_name['control_1024'],by_name['split_4096']
    # A real change in both matched observations, useful for its small cost.
    assert all(a['generation']['fid']-b['generation']['fid']>.5 for a,b in zip(control['curve'],split['curve'])), 'no sustained scout gain'
    assert split['final']['fid']<19.4482, 'expanded endpoint does not beat parent'
    folder=ROOT/f'configs/cifar_particle_ae/{TRACK}';folder.mkdir(parents=True,exist_ok=True);paths=[]
    for row in rows:
        parent=ROOT/row['config']['out_dir']/'checkpoint_020000.pt'
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==row['checkpoint_sha256']
        cfg={**row['config'],'steps':40000,'eval_interval':5000,'eval_samples':50000,'final_samples':50000,
             'initial_eval_samples':0,'out_dir':f'runs/cifar_particle_ae/{TRACK}/{row["name"]}',
             'resume_checkpoint':str(parent.relative_to(ROOT)),'resume_sha256':row['checkpoint_sha256'],
             'max_train_seconds':3600.}
        path=folder/f'{row["name"]}.yaml';path.write_text(yaml.safe_dump(cfg));paths.append(str(path.relative_to(ROOT)))
    manifest=folder/'manifest.json';manifest.write_text(json.dumps(paths,indent=2)+'\n');return manifest


def analyze():
    paths=json.loads((ROOT/f'configs/cifar_particle_ae/{TRACK}/manifest.json').read_text())
    defaults=trainer_defaults(str(ROOT/TRAINER));provenance=code_provenance(str(ROOT/TRAINER),sys.executable);rows=[]
    for path in paths:
        cfg=load_config(path,defaults);run=ROOT/cfg['out_dir'];assert has_valid_summary(str(run),cfg,provenance),run
        summary=json.loads((run/'summary.json').read_text())
        assert summary['start_step']==20000 and summary['final']['step']==40000
        assert summary['frozen_features_unchanged'] and summary['sigma_unchanged']
        metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        curve=[m for m in metrics if 'generation' in m];assert [m['step'] for m in curve]==[25000,30000,35000,40000]
        assert all(m['generation']['samples']==50000 for m in curve)
        assert all(m['learning_rates']==dict(G=.0003,E=.0003,prior=.003,D=.00045) for m in metrics)
        rows.append({'name':run.name,'curve':curve,**summary})
    assert rows[0]['rng_sha256']==rows[1]['rng_sha256']
    report=ROOT/f'reports/cifar-particle-ae/{TRACK}';report.mkdir(parents=True,exist_ok=True)
    (report/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Particle expansion persistence check','','2/2 certified; both respective 20k scout endpoints continued with unchanged optimizer/EMA/RNG.','',
           '| Arm | FID25k | FID30k | FID35k | FID40k | Train minutes |',
           '|---|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['final']['fid']):
        values=' | '.join(f"{m['generation']['fid']:.4f}" for m in r['curve'])
        lines.append(f"| {r['name']} | {values} | {r['train_seconds']/60:.2f} |")
    by_name={r['name']:r for r in rows};c,s=by_name['control_1024'],by_name['split_4096']
    deltas=[a['generation']['fid']-b['generation']['fid'] for a,b in zip(c['curve'],s['curve'])]
    lines+=['','Control minus expanded FID (positive favors expansion): '+', '.join(f'{x:+.4f}' for x in deltas)+'.', '',
            'Expanded center count does not establish semantic coverage. Review sibling grids/variation alongside FID. No automatic promotion to 200k.']
    finding=('Expansion retains a useful advantage at every evaluation.' if all(x>.5 for x in deltas) else 'Expansion does not retain a >0.5 FID advantage at every evaluation; inspect the trajectory before promotion.')
    recommendation=('Consider the next longer matched continuation if the endpoint and recent trend support it; do not select only the minimum.' if all(x>.5 for x in deltas) else 'Do not automatically promote. Prioritize an independent discriminator feedback/robustness test if the gain faded.')
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
    (report/'FINDINGS.md').write_text(f'# Persistence result\n\n{finding}\n\n{recommendation}\n\nSee LEADERBOARD.md and results.json for the complete FID50k trajectory, diagnostics and costs. No seed repeats.\n')
    print('\n'.join(lines)+'\n'+finding+'\n'+recommendation,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if not a.analyze_only:
        manifest=prepare();run=f'runs/cifar_particle_ae/{TRACK}'
        subprocess.run([sys.executable,'-u','experiments/follow_grid.py','--root',run,'--log',f'{run}/PIPELINE.log','--',
                        '--config_manifest',str(manifest),'--gpus','0,1','--workers_per_gpu','1','--python',sys.executable,'--trainer',TRAINER],cwd=ROOT,check=True)
    analyze()

#!/usr/bin/env python
"""Launch and certify read-only support diagnostics after the matched scouts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_grid import code_provenance,has_valid_summary,load_config,trainer_defaults


def main(smoke,gpus):
    track='particle_support_smoke_v2' if smoke else 'particle_support'
    folder=ROOT/f'configs/cifar_particle_ae/{track}';folder.mkdir(parents=True,exist_ok=True)
    run=ROOT/f'runs/cifar_particle_ae/{track}';run.mkdir(parents=True,exist_ok=True)
    report=ROOT/f'reports/cifar-particle-ae/{track}';report.mkdir(parents=True,exist_ok=True)
    parents={'parent_10k':'transgan_scout/cnn_e_only/checkpoint_010000.pt'}
    if not smoke:parents.update(control_20k='generator_balance/control/checkpoint_020000.pt',half_g_20k='generator_balance/half_g/checkpoint_020000.pt')
    paths=[]
    for name,parent in parents.items():
        p=Path('runs/cifar_particle_ae')/parent;assert p.exists(),p
        cfg={'checkpoint':str(p),'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
             'out_dir':f'runs/cifar_particle_ae/{track}/{name}'}
        if smoke:cfg.update(samples=128,particles=4,draws_per_particle=2)
        path=folder/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg));paths.append(str(path.relative_to(ROOT)))
    manifest=folder/'manifest.json';manifest.write_text(json.dumps(paths,indent=2)+'\n')
    trainer=ROOT/'experiments/probe_cifar_ae_support.py'
    subprocess.run([sys.executable,'experiments/follow_grid.py','--root',str(run),'--log',str(run/'PIPELINE.log'),'--',
                    '--config_manifest',str(manifest),'--gpus',gpus,'--workers_per_gpu','1','--python',sys.executable,'--trainer',str(trainer)],cwd=ROOT,check=True)
    defaults=trainer_defaults(str(trainer));provenance=code_provenance(str(trainer),sys.executable);rows=[]
    for path in paths:
        cfg=load_config(path,defaults);out=ROOT/cfg['out_dir']
        assert has_valid_summary(str(out),cfg,provenance),out
        r=json.loads((out/'summary.json').read_text());assert r['parent_unchanged'] and r['frozen_state_unchanged']
        if not smoke:
            if out.name=='parent_10k':
                metrics=[json.loads(line) for line in (ROOT/'runs/cifar_particle_ae/transgan_scout/cnn_e_only/metrics.jsonl').read_text().splitlines()]
                reference=next(m['generation']['fid'] for m in metrics if m['step']==10000)
            else:
                arm='control' if out.name=='control_20k' else 'half_g'
                reference=json.loads((ROOT/f'runs/cifar_particle_ae/generator_balance/{arm}/summary.json').read_text())['final']['fid']
            baseline=next(m['fid'] for m in r['sampling'] if m['noise_scale']==1.)
            r['baseline_reproduction_error']=baseline-reference
            assert abs(baseline-reference)<.01,(out,baseline,reference)
        rows.append({'name':out.name,**r})
    (report/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    if smoke:
        print('SMOKE certified, frozen state and original sampler checks passed');return
    lines=['# Read-only particle-support diagnostics','',f'{len(rows)}/{len(paths)} certified; standard-sampler FID reproduction within 0.01 for every parent.','',
           '| Checkpoint | Noise x0 FID50k | Original x1 FID50k | Noise x2 FID50k | Within-particle feature variation |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        values={p['noise_scale']:p['fid'] for p in r['sampling']}
        lines.append(f"| {r['name']} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {100*r['diversity']['inception']['within_fraction']:.2f}% |")
    lines+=['','Noise x0 repeatedly samples the 1024 particle centers, so its FID describes a discrete empirical generator. Noise x2 is an inference-only distribution change. These are not training results. Within-particle variation is a balanced ANOVA over 128 particles and 16 draws per particle, not a class-coverage or recall score.']
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--gpus',default='0,1');a=p.parse_args();main(a.smoke,a.gpus)

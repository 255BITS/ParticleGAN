#!/usr/bin/env python
"""Run read-only joint-checkpoint diagnostics through the experiment pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main(stage,gpu):
    parents = ({'control_8':'discriminator_joint_smoke/control/checkpoint.pt',
                'warmstart_8':'discriminator_joint_smoke/warmstart/checkpoint.pt',
                'control_20k':'discriminator_joint/control/checkpoint_020000.pt',
                'weaker_20k':'discriminator_joint/weaker/checkpoint_020000.pt'} if stage=='early' else
               {'warmstart_20k':'discriminator_joint/warmstart/checkpoint_020000.pt'})
    track=f'discriminator_joint_probes_{stage}'
    run=Path(f'runs/cifar_particle_ae/{track}')
    config=Path(f'configs/cifar_particle_ae/{track}')
    config.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,parent in parents.items():
        parent=Path('runs/cifar_particle_ae')/parent
        assert parent.exists(),parent
        cfg={'checkpoint':str(parent),'checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
             'out_dir':str(run/name),'steps':0,'samples':2048}
        path=config/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg));paths.append(str(path))
    manifest=config/'manifest.json';manifest.write_text(json.dumps(paths,indent=2)+'\n')
    subprocess.run([sys.executable,'experiments/follow_grid.py','--root',str(run),'--log',str(run/'PIPELINE.log'),'--',
                    '--config_manifest',str(manifest),'--gpus',gpu,'--workers_per_gpu','1','--python',sys.executable,
                    '--trainer','experiments/diagnose_cifar_ae_discriminator.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/analyze_cifar_ae_discriminator.py','--track',track],cwd=ROOT,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['early','final'],required=True);p.add_argument('--gpu',default='0');a=p.parse_args();main(a.stage,a.gpu)

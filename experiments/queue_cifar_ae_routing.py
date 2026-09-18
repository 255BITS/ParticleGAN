#!/usr/bin/env python
"""Start each routing scout as its GPU finishes the preceding plateau scout."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
run=Path('runs/cifar_particle_ae/routing_scout');run.mkdir(parents=True,exist_ok=True)
lock=(run/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env={**os.environ,'PYTHONUNBUFFERED':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'}
jobs=[(0,'recon01_lr025','no_recon_prior'),(1,'d2','encoder_only_recon')]
processes={};offsets={};status=0
with (run/'PIPELINE.log').open('a',buffering=1) as log:
    def emit(name,line):
        log.write(f'{datetime.now().isoformat(timespec="seconds")} [{name}] {line}\n')
    emit('queue','Routing scouts will start independently as GPU 0/1 become available.')
    while True:
        for gpu,gate,name in jobs:
            certificate=Path('runs/cifar_particle_ae/plateau_scout')/gate/'run_grid_complete.json'
            if name not in processes and certificate.exists():
                out=(run/f'gpu{gpu}.runner.log').open('w')
                command=['.venv/bin/python','-u','experiments/run_grid.py','--configs',
                         f'configs/cifar_particle_ae/routing_scout/{name}.yaml',
                         '--gpus',str(gpu),'--workers_per_gpu','1','--python','.venv/bin/python',
                         '--trainer','experiments/train_cifar_ae_routing.py']
                processes[name]=subprocess.Popen(command,env=env,stdout=out,stderr=subprocess.STDOUT)
                emit('launch',f'{name} on GPU {gpu}, preceding scout {gate} completed; pid={processes[name].pid}')
        for path in sorted(run.glob('*/log.txt'))+sorted(run.glob('*.runner.log')):
            with path.open(errors='replace') as source:
                source.seek(offsets.get(path,0))
                for line in source:
                    if line.strip():emit(path.parent.name if path.name=='log.txt' else path.name,line.rstrip())
                offsets[path]=source.tell()
        if len(processes)==len(jobs) and all(p.poll() is not None for p in processes.values()):
            status=int(any(p.returncode for p in processes.values()));break
        previous=Path('runs/cifar_particle_ae/plateau_scout/PIPELINE.log').read_text()
        if 'PIPELINE COMPLETE status=' in previous and 'PIPELINE COMPLETE status=0' not in previous:
            raise RuntimeError('Preceding pipeline failed; inspect logs')
        time.sleep(1)
    if status==0:
        status=subprocess.call(['.venv/bin/python','-u','experiments/analyze_cifar_ae_plateau.py',
            '--config_manifest','configs/cifar_particle_ae/routing_scout/manifest.json',
            '--report','reports/cifar-particle-ae/routing_scout','--trainer','experiments/train_cifar_ae_routing.py'],
            env=env,stdout=log,stderr=subprocess.STDOUT)
    emit('complete',f'PIPELINE COMPLETE status={status} report=reports/cifar-particle-ae/routing_scout/LEADERBOARD.md')
raise SystemExit(status)

#!/usr/bin/env python
"""Wait for the current two-GPU continuation, preflight, then run count scouts."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cifar_particle_ae/particle_scaling_scout'
REPORT=ROOT/'reports/cifar-particle-ae/particle_scaling_scout'


def main():
    RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage):
            record={'time':datetime.now().isoformat(timespec='seconds'),'stage':stage,'pid':os.getpid()}
            (REPORT/'QUEUE_STATUS.json').write_text(json.dumps(record,indent=2)+'\n')
            with (RUN/'PIPELINE.log').open('a') as out:out.write(f"{record['time']} [queue] {stage}\n")
            print(record,flush=True)
        env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
        def execute(command,log,extra_env=None):
            with log.open('w') as out:
                subprocess.run(command,cwd=ROOT,env={**env,**(extra_env or {})},stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,check=True)
        try:
            status('waiting for particle_expansion_40k to finish on both GPUs')
            deadline=time.monotonic()+3600
            completion=ROOT/'reports/cifar-particle-ae/particle_expansion_40k/results.json'
            while not completion.exists():
                if time.monotonic()>deadline:raise RuntimeError('40k continuation did not finish within queue deadline')
                time.sleep(15)
            execute([sys.executable,'experiments/cifar_ae_expansion_extend.py','--analyze-only'],REPORT/'PREVIOUS_CERTIFICATION.log')
            status('running CUDA identity, optimizer mapping, resume and E-only gradient tests')
            execute([sys.executable,'-m','pytest','-q','tests/test_cifar_ae_scaling.py','-s'],REPORT/'TESTS.txt',
                    {'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','RUN_CUDA_IMAGE_TESTS':'1'})
            status('running eight-update 8192/16384 smoke configs on both GPUs')
            execute([sys.executable,'-u','experiments/cifar_ae_scaling_pipeline.py','--smoke'],REPORT/'SMOKE_LAUNCHER.log')
            status('running 8192/16384 scouts: original 10k parent to 20k, FID50k at initialization/15k/20k')
            execute([sys.executable,'-u','experiments/cifar_ae_scaling_pipeline.py'],RUN/'scout_launcher.log')
            status('complete: both scouts certified; leaderboard, scaling curve and findings written')
        except BaseException as error:
            status(f'failed; no further stages will launch: {type(error).__name__}: {error}')
            raise


if __name__=='__main__':main()

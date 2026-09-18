#!/usr/bin/env python
"""Run information diagnostics after the active particle-count scouts finish."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cifar_particle_ae/particle_information'
REPORT=ROOT/'reports/cifar-particle-ae/particle_information'


def main():
    RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    with (RUN/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                 'experiments/probe_cifar_ae_information.py','tests/test_cifar_ae_information.py','experiments/cifar_ae_information_pipeline.py')}
        def status(stage):
            record={'time':datetime.now().isoformat(timespec='seconds'),'stage':stage,'pid':os.getpid(),'sources':sources}
            (REPORT/'QUEUE_STATUS.json').write_text(json.dumps(record,indent=2)+'\n')
            with (RUN/'PIPELINE.log').open('a') as out:out.write(f"{record['time']} [queue] {stage}\n")
            print(record,flush=True)
        env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
        def execute(command,log):
            with log.open('w') as out:
                subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,check=True)
        try:
            status('waiting for8192/16384 scouts to finish; no diagnostic GPU work yet')
            deadline=time.monotonic()+5400
            dependency=ROOT/'reports/cifar-particle-ae/particle_scaling_scout/QUEUE_STATUS.json'
            while True:
                state=json.loads(dependency.read_text())
                if state['stage'].startswith('failed'):raise RuntimeError('scaling queue failed: '+state['stage'])
                if state['stage'].startswith('complete:'):break
                if time.monotonic()>deadline:raise RuntimeError('scaling scouts did not finish before queue deadline')
                time.sleep(15)
            for name,expected in sources.items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, 'queued source changed: '+name
            # Recheck current run certificates. This only reads existing checkpoints/results.
            execute([sys.executable,'experiments/cifar_ae_scaling_pipeline.py','--analyze-only'],REPORT/'PREVIOUS_CERTIFICATION.log')
            status('running diagnostic unit tests')
            execute([sys.executable,'-m','pytest','-q','tests/test_cifar_ae_information.py'],REPORT/'TESTS.txt')
            status('running read-only4096 checkpoint GPU smoke; reduced budgets')
            execute([sys.executable,'-u','experiments/cifar_ae_information_pipeline.py','--smoke','--gpus','0'],REPORT/'SMOKE_LAUNCHER.log')
            status('running seven read-only checkpoint probes across both GPUs')
            execute([sys.executable,'-u','experiments/cifar_ae_information_pipeline.py'],RUN/'probe_launcher.log')
            status('complete: seven probes certified; feature-information and quality leaderboards written')
        except BaseException as error:
            status(f'failed; no further stages launch: {type(error).__name__}: {error}')
            raise


if __name__=='__main__':main()

"""Disclosed execution-only recovery; same immutable scientific requests."""
from pathlib import Path
import json,os
from experiments.forge.queue import Queue,drain
from experiments.forge.contracts import atomic_json,read_json,utc_now
ROOT=Path(__file__).resolve().parents[5]
QROOT=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/queue')
PROGRESS=Path('/tmp/bcap-physics-round5-20261009/gaussian_regression/progress.json')
CAMPAIGN='gaussian_regression-round5-v1'

def main():
    queue=Queue(QROOT,report_root=ROOT/'reports/forge',on_completion=None);state=queue.inspect()
    assert not any(j['status']=='running' for j in state['jobs'].values())
    rid=read_json(PROGRESS)['requests']['finite']
    key=next(k for k,j in state['jobs'].items() if rid in j['subscribers'] and j['definition']['task_id']=='gaussian1d_smoke')
    job=state['jobs'][key]
    if job['result'] and job['result']['raw']['attempt_status']=='timeout':
        assert len(job['attempts'])==1
        assert 18360+120+420<=21600
        queue.retry(key,reason='Execution-only recovery from unchanged120s smoke timeout and transient host RAM contention. Retain INCOMPLETE receipt; replay same frozen source/runtime/seed/initializer/batches/task budget on physicalGPU1, logicalcuda:0. Poll5s and limit pre-import BLAS/OpenMP threads; actual scientific Torch thread budget remains1.')
    progress=read_json(PROGRESS);progress.update(phase='execution_recovery',full_reservations_with_retry=18480,execution_retries=1)
    atomic_json(PROGRESS,progress)
    print(json.dumps(dict(time=utc_now(),event='execution_recovery',gpu='1',poll_seconds=5,preimport_thread_limits={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})),flush=True)
    drain(queue,['1'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN,poll_seconds=5)
    print(json.dumps(dict(time=utc_now(),event='recovery_drained',campaign=queue.inspect()['campaigns'][CAMPAIGN])),flush=True)
if __name__=='__main__':main()

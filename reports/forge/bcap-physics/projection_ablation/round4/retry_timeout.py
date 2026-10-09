"""One disclosed execution-only retry; same frozen source/runtime/seed/budgets."""
from pathlib import Path
from datetime import datetime,timezone
import json
from experiments.forge.queue import Queue,drain
from experiments.forge.contracts import atomic_json,read_json
ROOT=Path(__file__).resolve().parents[5]
QROOT=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_ablation/queue')
PROGRESS=Path('/tmp/bcap-physics-round4-20261009/projection_ablation/progress.json')
CAMPAIGN='projection-ablation-round4-v1'


def main():
    queue=Queue(QROOT,report_root=ROOT/'reports/forge',on_completion=None)
    state=queue.inspect()
    assert all(j['status']=='terminal' for j in state['jobs'].values())
    rid=next(rid for rid,e in state['submissions'].items() if e['request']['candidate']['id']=='projection-ablation-round4-strict_progress-v1')
    key=next(k for k,j in state['jobs'].items() if rid in j['subscribers'] and j['definition']['task_id']=='two_pole')
    job=state['jobs'][key]
    assert len(job['attempts'])==1 and job['result']['raw']['attempt_status']=='timeout'
    assert 19260+job['definition']['budget_seconds']==19560<=21600
    reason='Execution-only repair: original 300s wall timeout followed all80 matching inactive observations but preceded certified result publication; retain INCOMPLETE evidence, rerun unchanged frozen fixture under identical source/runtime/seed and full300s reservation.'
    queue.retry(key,reason=reason)
    progress=read_json(PROGRESS);progress.update(phase='execution_retry',full_reserved_seconds_including_repair=19560);atomic_json(PROGRESS,progress)
    print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),phase='retry_draining',key=key,reason=reason)),flush=True)
    drain(queue,['0'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN)
    print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),phase='retry_drained',campaign=queue.inspect()['campaigns'][CAMPAIGN])),flush=True)

if __name__=='__main__':main()

"""Start only the frozen CPU checkpoint auditor in its private area."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (HERE/'WATCHER.json').exists()
frozen=read(HERE/'CHECKER-FROZEN.json')
for key in ('files','original_checker','reviewed_hashes'):
    for p,d in frozen[key].items():assert sha(p)==d,p
summary=read(HERE/'accepted-attempt1/summary.json')
assert summary['status']=='VALID' and summary['source_integrity']['status']=='VALID'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
    'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
log=HERE/'watcher.log'
with log.open('xb') as handle:
    process=subprocess.Popen(['/tmp/pr38-default-env/bin/python','-B',str(HERE/'audit_checkpoints.py'),'--watch'],
        cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
startticks=int(Path(f'/proc/{process.pid}/stat').read_text().split()[21])
record=dict(pid=process.pid,startticks=startticks,started_utc=datetime.now(timezone.utc).isoformat(),
    log=str(log),output=str(HERE/'accepted-attempt1'),cpu_only=True,cuda_visible_devices='',
    checker_sha256=sha(HERE/'audit_checkpoints.py'),checker_freeze_sha256=sha(HERE/'CHECKER-FROZEN.json'),
    source_ready_sha256=frozen['prospective_ready_sha256'],lane_freeze_sha256=frozen['lane_freeze_sha256'],
    initial_steps=summary['sealed_steps'],initial_status=summary['status'])
with (HERE/'WATCHER.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))

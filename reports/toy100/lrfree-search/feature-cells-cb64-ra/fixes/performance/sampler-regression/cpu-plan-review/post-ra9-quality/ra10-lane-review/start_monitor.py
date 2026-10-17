"""Start only the unchanged CPU metadata watcher after sealed lane PASS."""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
LANE=ROOT/'validation-cb64-ra10'
OUTPUT=ROOT/'integration/review/validation-cb64-ra10-monitor'
MONITOR=ROOT/'integration/review/monitor_validation.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU watcher only'
assert sha(MONITOR)=='4ee4cae810342b675a39b269112ca907be3ffe88ae9cadab157e4933c487328b'
assert not OUTPUT.exists() and not (HERE/'MONITOR-START.json').exists()
seal=json.loads((HERE/'FROZEN.json').read_text()); assert seal['status']=='PASS'
for mapping in (seal['source_sha256'],seal['source_and_input_sha256']):
    for name,h in mapping.items(): assert sha(name)==h,name
assert sha(LANE/'source-freeze.json')==seal['source_freeze_sha256']
command=[sys.executable,'-B',str(MONITOR),'--validation',str(LANE),'--output',str(OUTPUT),'--watch']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
with (HERE/'monitor-process.log').open('wb') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
stat=Path(f'/proc/{child.pid}/stat').read_text()
startticks=int(stat[stat.rfind(')')+2:].split()[19])
cmdline=Path(f'/proc/{child.pid}/cmdline').read_bytes().split(b'\0')
actual=[item.decode() for item in cmdline if item]
assert actual==command,(actual,command)
value=dict(status='CPU_METADATA_WATCHER_STARTED',utc=datetime.now(timezone.utc).isoformat(),
    pid=child.pid,startticks=startticks,command=command,validation=str(LANE),output=str(OUTPUT),
    monitor_sha256=sha(MONITOR),lane_review_seal_sha256=sha(HERE/'FROZEN.json'),
    ready_sha256=seal['ready_sha256'],source_freeze_sha256=seal['source_freeze_sha256'],
    launch_source_sha256=sha(Path(__file__)),cpu_only=True,numerical_jobs_started=0,
    collector_indexed_declared_before_lane_freeze=True,RA4_runtime_adapter_reused=False,
    original_quality_gates_unchanged=True,other_queues_or_watchers_signaled=False)
(HERE/'MONITOR-START.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)

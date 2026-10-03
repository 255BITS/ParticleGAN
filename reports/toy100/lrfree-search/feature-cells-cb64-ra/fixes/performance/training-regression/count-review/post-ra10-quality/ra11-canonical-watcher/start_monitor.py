"""Root launches one unchanged read-only RA11 canonical CPU watcher."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
LANE=ROOT/'validation-cb64-ra11'
OUTPUT=ROOT/'integration/review/validation-cb64-ra11-monitor'
MONITOR=ROOT/'integration/review/monitor_validation.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU watcher only'
ready=json.loads((HERE/'WATCHER-READY.json').read_text())
frozen=json.loads((HERE/'PREPARATION-FROZEN.json').read_text())
assert ready['status']=='READY_SOURCE_ONLY' and frozen['status']=='PASS'
assert sha(HERE/'WATCHER-READY.json')==frozen['ready_sha256']
for path,digest in frozen['source_and_input_sha256'].items():assert sha(path)==digest,path
for path,digest in frozen['private_closed_sha256'].items():assert sha(path)==digest,path
assert sha(MONITOR)=='4ee4cae810342b675a39b269112ca907be3ffe88ae9cadab157e4933c487328b'
assert not OUTPUT.exists() and not (HERE/'MONITOR-START.json').exists()
command=[sys.executable,'-B',str(MONITOR),'--validation',str(LANE),'--output',str(OUTPUT),'--watch']
environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
with (HERE/'monitor-process.log').open('xb') as stream:
    child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
assert child.poll() is None
stat=Path(f'/proc/{child.pid}/stat').read_text()
startticks=int(stat[stat.rfind(')')+2:].split()[19])
actual=[item.decode() for item in Path(f'/proc/{child.pid}/cmdline').read_bytes().split(b'\0') if item]
assert actual==command,(actual,command)
value=dict(status='CPU_METADATA_WATCHER_STARTED',utc=datetime.now(timezone.utc).isoformat(),
    pid=child.pid,startticks=startticks,command=command,validation=str(LANE),output=str(OUTPUT),
    monitor_sha256=sha(MONITOR),preparation_freeze_sha256=sha(HERE/'PREPARATION-FROZEN.json'),
    root_ready_sha256=ready['root_ready_sha256'],source_freeze_sha256=ready['source_freeze_sha256'],
    launch_source_sha256=sha(__file__),cpu_only=True,numerical_jobs_started=0,
    collector_indexed_declared_before_lane_freeze=True,RA4_runtime_adapter_reused=False,
    original16_screen_checks_unchanged=True,other_queues_or_watchers_signaled=False)
(HERE/'MONITOR-START.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps(value),flush=True)

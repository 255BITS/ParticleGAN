"""Exclusive CPU child/provenance wrapper for the one root-authorized API run."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=HERE/'API-INPUTS-FROZEN.json';helper=HERE/'check_affected_api.py';go=HERE/'ROOT-GO-API.json'
assert sha(go)=='7a7d861758707821aeeac1bdb9574ca3f342632e17dd39cb735f724049fa9773'
approval=json.loads(go.read_text())
assert approval['status']=='GO_ONE_AFFECTED_CPU_API_INVOCATION'
assert sha(inputs)==approval['input_seal_sha256'] and sha(helper)==approval['helper_sha256']
assert approval['API_sample_calls']==3 and approval['CPU_updates_total']==2
bindings=json.loads(inputs.read_text())
for p,h in bindings['protected_sha256'].items():assert sha(p)==h,p
log=HERE/'numerical-attempt1.log';launch=HERE/'LAUNCH-attempt1.json';exit_path=HERE/'EXIT-attempt1.json'
output=HERE/'accepted-attempt1'
assert not any(p.exists() for p in (log,launch,exit_path,output))
environment=dict(os.environ)
cpu_environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
environment.update(cpu_environment)
command=['/tmp/pr38-default-env/bin/python','-B',str(helper),'--inputs',str(inputs),'--output',str(output)]
start=time.monotonic();utc=datetime.now(timezone.utc).isoformat()
with log.open('xb') as stream:
    child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    value=dict(status='STARTED_ONE_ROOT_AUTHORIZED_CPU_API',utc=utc,pid=child.pid,startticks=ticks,
        command=command,cwd=str(ROOT),CPU_environment=cpu_environment,input_seal_sha256=sha(inputs),
        helper_sha256=sha(helper),root_GO_sha256=sha(go),launcher_sha256=sha(__file__),
        API_sample_calls=3,CPU_updates_total=2,new_quality_emissions=0)
    launch.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(event='started',pid=child.pid,startticks=ticks)),flush=True)
    code=child.wait()
result=dict(status='EXITED',utc=datetime.now(timezone.utc).isoformat(),pid=child.pid,startticks=ticks,
    exit_code=code,elapsed_seconds=time.monotonic()-start,log_sha256=sha(log),root_GO_sha256=sha(go),
    process_absent=not Path(f'/proc/{child.pid}').exists(),API_sample_calls_authorized=3,CPU_updates_total_authorized=2)
exit_path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result),flush=True)
if code:raise SystemExit(code)

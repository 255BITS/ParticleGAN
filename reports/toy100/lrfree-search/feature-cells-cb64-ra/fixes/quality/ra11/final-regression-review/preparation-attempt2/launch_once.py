"""Exclusive one-shot CPU launch; root GO and closed input hashes required."""
import argparse
import os
import subprocess
import sys
import time
from common import *

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-freeze',type=Path,required=True);parser.add_argument('--root-go',type=Path,required=True)
args=parser.parse_args()
seal=read(args.input_freeze);verify(seal['source_and_input_sha256'])
go=read(args.root_go)
assert go['input_freeze_sha256']==sha(args.input_freeze) and go['helper_sha256']==sha(HERE/'audit_final.py')
assert go['exactly_one_CPU_artifact_invocation'] is True
assert not (HERE/'LAUNCH-attempt1.json').exists() and not (HERE/'EXIT-attempt1.json').exists()
command=['/tmp/pr38-default-env/bin/python','-B',str(HERE/'audit_final.py'),'--input-freeze',str(args.input_freeze.resolve()),'--root-go',str(args.root_go.resolve()),'--output',str(HERE/'accepted-attempt1')]
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
started=now();clock=time.monotonic()
with (HERE/'attempt1.log').open('xb') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    identity=process_identity(child.pid)
    write_new(HERE/'LAUNCH-attempt1.json',dict(status='ONE_CPU_ARTIFACT_AUDIT_STARTED',utc=started,pid=child.pid,startticks=identity.get('startticks'),command=command,
        log=str(HERE/'attempt1.log'),input_freeze_sha256=sha(args.input_freeze),root_GO_sha256=sha(args.root_go),helper_sha256=sha(HERE/'audit_final.py'),CUDA_VISIBLE_DEVICES='',PT_loads_planned=14))
    print(json.dumps(dict(event='CPU_artifact_audit_started',pid=child.pid,startticks=identity.get('startticks'))),flush=True)
    code=child.wait()
write_new(HERE/'EXIT-attempt1.json',dict(status='CPU_ARTIFACT_AUDIT_EXITED',utc=now(),returncode=code,pid=child.pid,startticks=identity.get('startticks'),seconds=time.monotonic()-clock,
    log_closed=True,log_sha256=sha(HERE/'attempt1.log'),child_identity_after_wait=process_identity(child.pid),source_and_input_guard_after_exit=True))
verify(seal['source_and_input_sha256'])
print(json.dumps(dict(event='CPU_artifact_audit_exited',returncode=code,log_sha256=sha(HERE/'attempt1.log'))),flush=True)
sys.exit(code)

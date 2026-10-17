"""CPU-only inherited-flock fixture with private dummy processes; no signals."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

AREA = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
CHILD = """import json, os, sys, time
from pathlib import Path
area=Path(sys.argv[1]); fd=int(sys.argv[2])
info=os.fstat(fd)
(area/'child-ready.json').write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),fd=fd,ino=info.st_ino)))
deadline=time.monotonic()+10
while not (area/'release').exists():
    if time.monotonic()>deadline: raise SystemExit('fixture release timed out')
    time.sleep(.005)
(area/'child-done.json').write_text(json.dumps(dict(pid=os.getpid(),fd_still_open=os.fstat(fd).st_ino==info.st_ino)))
"""
PARENT = """import fcntl, json, os, subprocess, sys
from pathlib import Path
area=Path(sys.argv[1]); child_source=sys.argv[2]
lock=(area/'private.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (area/'dummy-child.log').open('w') as log:
    child=subprocess.Popen([sys.executable,'-B','-c',child_source,str(area),str(lock.fileno())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
(area/'parent.json').write_text(json.dumps(dict(parent=os.getpid(),child=child.pid,fd=lock.fileno())))
os._exit(0)
"""

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def wait_for(test):
    deadline=time.monotonic()+5
    while not test():
        if time.monotonic()>deadline: raise AssertionError('private fixture timed out')
        time.sleep(.005)
def child_state(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError: return 'gone'

phase=ROOT/'quality/group-profile-phase/READY.json'
v1=ROOT/'quality/run_group_profile.py'; v2=ROOT/'quality/run_group_profile_v2.py'
prep=ROOT/'quality/group-profile-phase-v2/PREPARATION.json'
assert sha(v2)=='5bc48310ee58859a4389dd547599934f3f46fdf803e7e548d85bdc4091168707'
assert sha(prep)=='1fda9828e929a7dab0e2a8e19d56774fe95b78d860b88c49c50514f63668dfbb'
restored=v2.read_text().replace("'group-profile-phase-v2'", "'group-profile-phase'")
restored=restored.replace('stderr=subprocess.STDOUT,\n                pass_fds=(lock.fileno(),))','stderr=subprocess.STDOUT)')
assert restored.encode()==v1.read_bytes()
frozen=json.loads(phase.read_text())
assert len(frozen['source_sha256'])==83
assert all(sha(Path(path))==value for path,value in frozen['source_sha256'].items())
assert not (AREA/'parent.json').exists()
started=datetime.now(timezone.utc).isoformat()
command=[sys.executable,'-B','-c',PARENT,str(AREA),CHILD]
result=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=5)
assert result.returncode==0, result.stderr
wait_for(lambda:(AREA/'child-ready.json').exists())
parent=json.loads((AREA/'parent.json').read_text()); child=json.loads((AREA/'child-ready.json').read_text())
assert parent['child']==child['pid'] and child_state(child['pid']) not in ('Z','X','gone')
assert not Path(f"/proc/{parent['parent']}").exists()
probe=(AREA/'private.lock').open('a')
try:
    try:
        fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        blocked_after_parent_exit=True
    else:
        fcntl.flock(probe,fcntl.LOCK_UN)
        raise AssertionError('lock released while orphan dummy-child alive')
finally:
    (AREA/'release').write_text('release private dummy-child\n')
wait_for(lambda:child_state(child['pid']) in ('Z','X','gone'))
terminal=child_state(child['pid'])
fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
fcntl.flock(probe,fcntl.LOCK_UN)
probe.close()
done=json.loads((AREA/'child-done.json').read_text())
assert done['fd_still_open']
receipt=dict(status='PASS',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),scope='one private CPU subprocess inherited-flock fixture and read-only wrapper delta/83-map proof',wrapper_source_sha256=sha(v2),preparation_sha256=sha(prep),base_phase_ready_sha256=sha(phase),base_source_sha256=sha(v1),verified_base_inputs=83,full_v1_bytes_restored_after_two_changes=True,private_parent=parent,private_child=child,parent_exited_before_lock_probe=True,lock_blocked_while_orphan_child_alive=blocked_after_parent_exit,lock_acquired_after_child_exit=True,child_terminal_state=terminal,child_held_fd_until_exit=done['fd_still_open'],real_job_operations=0,signals=0,cuda_imports=0,unchanged=['owner gpu_command','all numerical inputs/law','GPU settings','pre/post guards','output paths','normal-path receipts'],remaining_limit='An interrupted phase may still lack PHASE-RESULT; preserve partial evidence. The inherited serial lock now prevents another cooperative phase until the profiler exits.')
(AREA/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],parent_exited=True,orphan_lock_blocked=True,lock_released_after_child_exit=True,area=str(AREA))))

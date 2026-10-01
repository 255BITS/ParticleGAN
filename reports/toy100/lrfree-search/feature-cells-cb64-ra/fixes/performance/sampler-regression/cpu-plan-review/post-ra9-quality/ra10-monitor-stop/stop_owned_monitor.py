"""Stop only the exact RA10 owned metadata watcher after canonical grid failure."""
from datetime import datetime,timezone
import hashlib,json,os,signal,time
from pathlib import Path
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
preparation=read(HERE/'PREPARATION-FROZEN.json')
assert preparation['status']=='FROZEN_OWNED_CPU_WATCHER_STOP_SOURCE_GUARDS'
for p,h in preparation['files'].items():assert sha(p)==h,p
START=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/ra10-lane-review/MONITOR-START.json'
assert sha(START)=='dfcee3d1662f74d0057de38fa55ff3534fcba56040f9597f4180fe50acff2c38'
owned=read(START);pid=owned['pid'];assert pid==1030112 and owned['startticks']==167460340
assert owned['command']==['/tmp/pr38-default-env/bin/python','-B',str(ROOT/'integration/review/monitor_validation.py'),'--validation',str(ROOT/'validation-cb64-ra10'),'--output',str(ROOT/'integration/review/validation-cb64-ra10-monitor'),'--watch']
assert sha(ROOT/'integration/review/monitor_validation.py')==owned['monitor_sha256']
artifact=read(ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/ra10-grid-artifact-review/accepted-attempt2/receipt.json')
assert artifact['status']==artifact['evidence_status']=='VALID' and artifact['quality_verdict']=='FAIL'
canonical=Path(owned['output'])/'canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
accepted=read(canonical)
assert accepted['canonical_fixture_validity']=='VALID' and not accepted['validity_reasons']
assert accepted['primary_status']==accepted['acceptance_status']=='FAIL' and accepted['source_integrity']['status']=='VALID'
assert sha(canonical)=='1a354e6acbdc097233e9afc83c167383d917e0462bcb904f89958af56f02f147'
target=HERE/'STOPPED-AT-GRID-FAILURE.json';assert not target.exists()

def inspect_owned():
    proc=Path(f'/proc/{pid}')
    stat=(proc/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
    command=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    children={str(p.parent.name):p.read_text().split() for p in sorted((proc/'task').glob('*/children'))}
    assert int(parts[19])==owned['startticks'] and command==owned['command']
    assert parts[0]!='Z' and children and not any(children.values())
    env=dict(item.split('=',1) for item in (proc/'environ').read_bytes().decode().split('\0') if '=' in item)
    assert env.get('CUDA_VISIBLE_DEVICES')=='' and all(env.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    return dict(pid=pid,startticks=int(parts[19]),state=parts[0],command=command,all_thread_children=children,cpu_environment_checked=True)

pidfd=os.pidfd_open(pid,0)
try:
    before=inspect_owned();second=inspect_owned()
    (HERE/'identity-before-stop.json').write_text(json.dumps([before,second],indent=2)+'\n')
    signal.pidfd_send_signal(pidfd,signal.SIGTERM)
    terminal=None
    for _ in range(50):
        p=Path(f'/proc/{pid}/stat')
        if not p.exists():terminal='absent';break
        raw=p.read_text();parts=raw[raw.rfind(')')+2:].split()
        assert int(parts[19])==owned['startticks'],'PID changed after signal'
        if parts[0]=='Z':terminal='zombie';break
        time.sleep(.1)
    assert terminal,'Owned watcher did not terminate; no other signal sent'
finally:os.close(pidfd)
summary=Path(owned['output'])/'summary.json'
copy=HERE/'summary-after-stop.json';assert not copy.exists();copy.write_bytes(summary.read_bytes());s=read(copy)
assert s['status']=='PENDING' and s['completed']==1 and s['total']==16 and s['source_integrity']['status']=='VALID'
grid=[row for row in s['records'] if row['task']=='grid100'];pending=[row for row in s['records'] if row['acceptance_status']=='PENDING']
assert len(grid)==1 and grid[0]['canonical_fixture_validity']=='VALID' and grid[0]['acceptance_status']=='FAIL'
assert len(pending)==15 and all(row['canonical_fixture_validity']=='UNVERIFIED' and row['primary_status']=='PENDING' for row in pending)
assert sha(canonical)=='1a354e6acbdc097233e9afc83c167383d917e0462bcb904f89958af56f02f147'
for p,h in preparation['files'].items():assert sha(p)==h,p
value=dict(status='STOPPED_OWNED_CPU_WATCHER_AT_COMPLETED_GRID_FAILURE',utc=datetime.now(timezone.utc).isoformat(),pid=pid,startticks=owned['startticks'],
 two_exact_identity_nochild_checks=[before,second],signal_sent='SIGTERM',signals_sent=1,pidfd_identity_bound_signal=True,termination=terminal,
 monitor_start_sha256=sha(START),canonical_receipt_sha256=sha(canonical),stop_source_sha256=sha(Path(__file__)),summary_snapshot_sha256=sha(copy),
 completed_screens=1,total_screens=16,pending_tasks=[row['task'] for row in pending],summary_status='PENDING',remaining15_original_screens='Deliberately not run after original grid quality failure; PENDING with fixtures UNVERIFIED; no quality verdict.',
 artifact_validity='VALID',quality_verdict='FAIL',original_gates_preserved=True,cpu_only=True,other_queues_or_watchers_signaled=False,numerical_jobs_started=0,
 frozen_sources_or_prior_receipts_changed=False,source_guards_checked_before_signal=len(preparation['files']))
target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(status=value['status'],pid=pid,startticks=owned['startticks'],completed_screens=1,total_screens=16,pending_screens=15,termination=terminal)),flush=True)

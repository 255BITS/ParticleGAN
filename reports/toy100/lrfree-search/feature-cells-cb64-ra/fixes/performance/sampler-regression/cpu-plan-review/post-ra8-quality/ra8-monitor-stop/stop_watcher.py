"""Seal the completed grid and stop only the authorized owned CPU watcher."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import time

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
LANE = ROOT / 'validation-cb64-ra8'
OUT = ROOT / 'integration/review/validation-cb64-ra8-monitor'
RUN = LANE / 'screens/runs/grid100'
PID, START = 774187, 165604598
COMMAND = ['/tmp/pr38-default-env/bin/python', '-u', '-B',
           str(ROOT / 'integration/review/monitor_validation.py'), '--validation', str(LANE),
           '--output', str(OUT), '--watch']
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
write = lambda path, value: path.write_text(json.dumps(value, indent=2) + '\n')


def identity():
    proc = Path(f'/proc/{PID}')
    raw = (proc/'stat').read_text()
    fields = raw.rsplit(')',1)[1].split()
    return dict(pid=PID,startticks=int(fields[19]),process_state=fields[0],stat=raw,
                cmdline=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x],
                children=sorted({int(child) for task in (proc/'task').iterdir()
                                 for child in (task/'children').read_text().split()}))


assert not (HERE/'STOPPED-AT-GRID-FAILURE.json').exists()
acceptance_path=OUT/'canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
canonical=read(acceptance_path)
assert canonical['canonical_fixture_validity']=='VALID' and canonical['validity_reasons']==[]
assert canonical['primary_status']==canonical['acceptance_status']=='FAIL'
assert canonical['completed_steps']==7000 and canonical['observations']==34
assert canonical['native']['terminal_accuracy']==[False]*5
assert canonical['legacy_fixture_comparison']=='MATCH'
assert canonical['native_evidence']['initial_parameter_match']=={
    'G':{'weight':True,'bias':True},'prior':{'z':True}}
assert canonical['native_evidence']['prior_range_match'] is True
result=read(RUN/'result.json')
execution=read(RUN/'execution-receipt.json')
assert execution['process_exit_code']==0
assert sha(RUN/'result.json')==canonical['result_sha256']
assert sha(RUN/'execution-receipt.json')==canonical['execution_receipt_sha256']
summary=read(OUT/'summary.json')
assert summary['status']=='PENDING' and summary['completed']==1 and summary['total']==16
assert summary['counts']=={'portability':{'PENDING':13},'native':{'FAIL':1,'PENDING':2}}
pending=[r for r in summary['records'] if r['task']!='grid100']
assert len(pending)==15 and all(r['primary_status']==r['acceptance_status']=='PENDING'
    and r['canonical_fixture_validity']=='UNVERIFIED' for r in pending)
assert next(r for r in summary['records'] if r['task']=='grid100')==canonical
guards={}
freeze=LANE/'source-freeze.json'
guards[str(freeze)]=sha(freeze)
frozen=read(freeze)
guards.update({str(LANE/n):h for n,h in frozen['local_sources'].items()})
guards.update(frozen['external_sources'])
for item in read(LANE/'screens/source-freeze.json')['files'].values():
    guards[item['path']]=item['sha256']
for path,h in guards.items():assert sha(path)==h,path
first=identity()
assert first['startticks']==START and first['cmdline']==COMMAND and first['children']==[]
assert first['process_state'] in {'R','S'}
write(HERE/'identity-before-stop.json',first)
second=identity()
assert second['startticks']==START and second['cmdline']==COMMAND and second['children']==[]
os.kill(PID,signal.SIGTERM)
terminal=None
for _ in range(30):
    path=Path(f'/proc/{PID}/stat')
    if not path.exists():terminal='gone';break
    fields=path.read_text().rsplit(')',1)[1].split()
    assert int(fields[19])==START,'PID reused; no additional signal allowed'
    if fields[0]=='Z':terminal='zombie';break
    time.sleep(.1)
assert terminal is not None,'Owned watcher did not terminate; no additional signal sent'
for original,name in ((OUT/'summary.json','summary-at-stop.json'),
                      (acceptance_path,'canonical-grid-acceptance-receipt.json'),
                      (OUT/'CHECKER-IDENTITY.json','CHECKER-IDENTITY.json'),
                      (RUN/'result.json','grid-result.json'),
                      (RUN/'execution-receipt.json','grid-execution-receipt.json')):
    (HERE/name).write_bytes(original.read_bytes())
assert read(HERE/'summary-at-stop.json')==summary
artifacts={str(p):sha(p) for p in sorted(RUN.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
write(HERE/'GRID-ARTIFACT-MANIFEST.json',dict(status='SEALED_COMPLETED_SAVED_ARTIFACTS',files=artifacts))
failed={k:v for k,v in canonical['threshold_margins'].items() if v['passing_margin']<0}
receipt=dict(status='STOPPED_AT_GRID_FAILURE',utc=datetime.now(timezone.utc).isoformat(),
    authorization='Root requested stop of this owned CPU watcher after completed valid grid FAIL; no more RA8 queued.',
    owned_pid=PID,owned_startticks=START,exact_cmdline=COMMAND,no_children_before_signal=True,
    signal='SIGTERM',terminal=terminal,source_integrity=summary['source_integrity'],
    canonical_receipt_sha256=sha(acceptance_path),canonical_fixture_validity='VALID',
    grid_quality='FAIL',final_metrics=canonical['final'],failed_original_terminal_thresholds=failed,
    native=canonical['native'],canonical_screens=dict(completed=1,total=16,valid_failed=1,pending=15,
        unrun_fixture_validity='UNVERIFIED',unrun_quality_verdict=None),
    source_and_input_sha256=guards,grid_artifact_sha256=artifacts,
    evidence_sha256={str(p):sha(p) for p in HERE.iterdir() if p.is_file()},
    parent_original_ra4_processes_signaled=False,gpu_processes_signaled=False,
    numerical_jobs_started=0,torch_imported=False,numerical_inputs_sources_modified=False)
write(HERE/'STOPPED-AT-GRID-FAILURE.json',receipt)
print(json.dumps(dict(status=receipt['status'],pid=PID,terminal=terminal,
    source_files=len(guards),grid_artifacts=len(artifacts),canonical_completed=1,canonical_pending=15,
    failed_terminal_thresholds=list(failed))),flush=True)

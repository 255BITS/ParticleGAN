"""Stop only the completed negative lane's owned CPU metadata watcher."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
START=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra8-quality/ra9-lane-review/MONITOR-START.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only ownership action'
assert sha(START)=='4f60313c1f60005084d602810a54c46477b7f66e83209da0a48070b774e76589'
owned=json.loads(START.read_text())
pid=owned['pid']; assert pid==860452 and owned['startticks']==166179427
assert owned['command']==['/tmp/pr38-default-env/bin/python','-B',
    str(ROOT/'integration/review/monitor_validation.py'),'--validation',str(ROOT/'validation-cb64-ra9'),
    '--output',str(ROOT/'integration/review/validation-cb64-ra9-monitor'),'--watch']
audit=json.loads((HERE/'receipt.json').read_text())
assert audit['status']=='PASS' and audit['quality_verdict']=='FAIL' and audit['canonical_fixture_validity']=='VALID'
assert len(audit['pending_tasks'])==15
canonical=ROOT/'integration/review/validation-cb64-ra9-monitor/canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
assert sha(canonical)==audit['canonical_receipt_sha256']
target=HERE/'STOPPED-AT-GRID-FAILURE.json'; assert not target.exists()


def inspect_owned():
    path=Path(f'/proc/{pid}')
    stat=(path/'stat').read_text(); parts=stat[stat.rfind(')')+2:].split()
    actual=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x]
    children={str(p.parent.name):p.read_text().split() for p in sorted((path/'task').glob('*/children'))}
    assert int(parts[19])==owned['startticks'] and actual==owned['command']
    assert parts[0]!='Z' and children and not any(children.values())
    return dict(pid=pid,startticks=int(parts[19]),state=parts[0],command=actual,all_thread_children=children)


before=inspect_owned(); second=inspect_owned()
os.kill(pid,signal.SIGTERM)
gone=False
for _ in range(50):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists(): gone=True; break
    stat=path.read_text(); parts=stat[stat.rfind(')')+2:].split()
    assert int(parts[19])==owned['startticks'], 'PID identity changed after owned stop'
    if parts[0]=='Z': gone=True; break
    time.sleep(.1)
assert gone,'Owned watcher did not terminate; no additional signal sent'
summary=Path(owned['output'])/'summary.json'
copy=HERE/'summary-after-stop.json'; assert not copy.exists(); copy.write_bytes(summary.read_bytes())
s=json.loads(copy.read_text())
assert s['completed']==1 and s['total']==16 and s['source_integrity']['status']=='VALID'
assert all(row['canonical_fixture_validity']=='UNVERIFIED' for row in s['records'] if row['acceptance_status']=='PENDING')
assert sha(canonical)==audit['canonical_receipt_sha256']
value=dict(status='STOPPED_OWNED_CPU_WATCHER_AT_COMPLETED_GRID_FAILURE',utc=datetime.now(timezone.utc).isoformat(),
    pid=pid,startticks=owned['startticks'],two_exact_identity_nochild_checks=[before,second],
    signal_sent='SIGTERM',signals_sent=1,terminated_or_zombie=True,
    monitor_start_sha256=sha(START),canonical_receipt_sha256=sha(canonical),
    audit_receipt_sha256=sha(HERE/'receipt.json'),stop_source_sha256=sha(Path(__file__)),
    summary_snapshot_sha256=sha(copy),completed_screens=1,total_screens=16,
    remaining15_original_screens='PENDING; fixtures UNVERIFIED; no quality verdict',
    quality_verdict='FAIL',canonical_fixture_validity='VALID',original_gates_preserved=True,
    cpu_only=True,other_queues_or_watchers_signaled=False,numerical_jobs_started=0,
    frozen_sources_or_prior_receipts_changed=False)
target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)

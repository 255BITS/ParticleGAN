"""Post-exit seal for one completed private RA10 grid audit, no signals/reruns."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (HERE/'FROZEN.json').exists()
receipt=read(HERE/'receipt.json'); prepared=read(HERE/'HELPERS-FROZEN.json')
assert receipt['status']=='PASS' and receipt['canonical_fixture_validity']=='VALID'
assert receipt['quality_verdict'] in ('PASS','FAIL')
watch=read(HERE/'WATCH-START.json'); proc=Path(f'/proc/{watch["pid"]}/stat')
if proc.exists():
    stat=proc.read_text(); values=stat[stat.rfind(')')+2:].split()
    assert int(values[19])==watch['startticks'] and values[0]=='Z','audit watcher not exited'
for mapping in (prepared['source_and_input_sha256'],prepared['helper_sha256'],receipt['artifact_sha256']):
    for name,h in mapping.items(): assert sha(name)==h,name
assert (HERE/'watch-attempt1.log').is_file()
assert 'grid_audit_complete' in (HERE/'watch-attempt1.log').read_text()
files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
local={str(p):sha(p) for p in files}
value=dict(status='PASS',scope='AUTHORITATIVE_POST_EXIT_ORIGINAL_GRID_AUDIT_SEAL',
    utc=datetime.now(timezone.utc).isoformat(),receipt_sha256=sha(HERE/'receipt.json'),
    quality_verdict=receipt['quality_verdict'],canonical_fixture_validity='VALID',
    source_and_input_sha256=prepared['source_and_input_sha256'],artifact_sha256=receipt['artifact_sha256'],
    local_sha256=local,watcher_identity=watch,watcher_exited=True,log_closed=True,
    original_quality_verdict_unchanged=True,cpu_only=True,signals_sent=0,
    no_PT_objects_models_forwards_draws_training_or_rescoring=True)
(HERE/'FROZEN.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(status='PASS',quality_verdict=value['quality_verdict'],local_files=len(local),
    artifact_files=len(value['artifact_sha256']),frozen_sha256=sha(HERE/'FROZEN.json'))),flush=True)

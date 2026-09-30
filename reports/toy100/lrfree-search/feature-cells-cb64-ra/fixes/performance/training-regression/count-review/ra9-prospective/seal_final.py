"""Seal only closed RA9 toy auditor receipts and logs, after review exits."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (HERE/'FINAL-FROZEN.json').exists()
r=read(HERE/'FINAL-RECEIPT.json')
assert r['status']=='VALID'
for name,digest in r['verified_hashes'].items():assert sha(name)==digest,name
watcher=read(HERE/'WATCHER.json');stat=Path(f"/proc/{watcher['pid']}/stat")
if stat.exists():
    fields=stat.read_text().rsplit(')',1)[1].split()
    assert int(fields[19])!=watcher['startticks'] or fields[0]=='Z'
files={str(p):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
frozen=dict(status='VALID',post_exit=True,frozen_utc=datetime.now(timezone.utc).isoformat(),
    receipt_sha256=sha(HERE/'FINAL-RECEIPT.json'),files=files,reviewed_hashes=r['verified_hashes'],
    toy_quality_gate=r['unchanged_root_full_toy_gate']['status'],full_original_Grid100_separate=True)
with (HERE/'FINAL-FROZEN.json').open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
for name,digest in {**files,**r['verified_hashes']}.items():assert sha(name)==digest,name
print(json.dumps(dict(status='VALID',receipt_sha256=sha(HERE/'FINAL-RECEIPT.json'),
    freeze_sha256=sha(HERE/'FINAL-FROZEN.json'),toy_quality_gate=frozen['toy_quality_gate'])))

import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE/'FROZEN.json').exists()
r=json.loads((HERE/'receipt.json').read_text())
assert r['status']=='PASS' and r['evidence']=='VALID'
for p,d in r['reviewed_sha256'].items():assert sha(p)==d,p
maps={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
(HERE/'FROZEN.json').write_text(json.dumps(dict(status='PASS',local_file_sha256=maps,
    reviewed_sha256=r['reviewed_sha256'],receipt_sha256=sha(HERE/'receipt.json')),indent=2)+'\n')
print(json.dumps(dict(status='PASS',receipt_sha256=sha(HERE/'receipt.json'),freeze_sha256=sha(HERE/'FROZEN.json'))))

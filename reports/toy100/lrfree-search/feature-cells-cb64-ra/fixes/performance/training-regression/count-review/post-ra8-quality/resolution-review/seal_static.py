import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE/'STATIC-FROZEN.json').exists()
r=json.loads((HERE/'static-receipt.json').read_text())
assert r['status']=='PASS_STATIC_SOURCE_AND_SCALAR_STATE'
for p,d in r['reviewed_sha256'].items():assert sha(p)==d,p
maps={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
(HERE/'STATIC-FROZEN.json').write_text(json.dumps(dict(status='PASS_STATIC_SOURCE_AND_SCALAR_STATE',
    local_file_sha256=maps,reviewed_sha256=r['reviewed_sha256'],receipt_sha256=sha(HERE/'static-receipt.json'),
    final_owner_and_root_review='PENDING'),indent=2)+'\n')
print(json.dumps(dict(status='PASS_STATIC_SOURCE_AND_SCALAR_STATE',freeze_sha256=sha(HERE/'STATIC-FROZEN.json'))))

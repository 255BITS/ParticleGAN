import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OUT=ROOT/'quality/ra9/independent-source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE/'FROZEN.json').exists() and not (OUT/'FROZEN.json').exists()
r=json.loads((HERE/'receipt.json').read_text())
assert r['status']=='PASS' and sha(HERE/'receipt.json')==sha(OUT/'receipt.json')
for p,d in r['reviewed_sha256'].items():assert sha(p)==d,p
maps={str(p):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
maps.update({str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
frozen=dict(status='PASS',receipt_sha256=sha(OUT/'receipt.json'),local_file_sha256=maps,
    reviewed_sha256=r['reviewed_sha256'],post_exit=True,quality_verdict=None)
for path in (HERE/'FROZEN.json',OUT/'FROZEN.json'):
    with path.open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
for p,d in maps.items():assert sha(p)==d,p
print(json.dumps(dict(status='PASS',receipt_sha256=sha(OUT/'receipt.json'),freeze_sha256=sha(OUT/'FROZEN.json'))))

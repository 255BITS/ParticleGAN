import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE/'FROZEN.json').exists()
prep=json.loads((HERE/'PREPARATION-FROZEN.json').read_text())
for p,d in prep['source_and_input_sha256'].items():assert sha(p)==d,p
r=json.loads((HERE/'result.json').read_text())
assert r['status']=='VALID' and r['queried_rows']==126 and r['query_by_table_distance_pairs']==2520000
local={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
(HERE/'FROZEN.json').write_text(json.dumps(dict(status='VALID',original_quality='FAIL',local_file_sha256=local,
    source_and_input_sha256=prep['source_and_input_sha256'],result_sha256=sha(HERE/'result.json')),indent=2)+'\n')
print(json.dumps(dict(status='VALID',result_sha256=sha(HERE/'result.json'),freeze_sha256=sha(HERE/'FROZEN.json'))))

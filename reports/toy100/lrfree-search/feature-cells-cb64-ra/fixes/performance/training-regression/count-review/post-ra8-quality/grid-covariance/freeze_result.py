import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE/'FROZEN.json').exists()
prep=json.loads((HERE/'PREPARATION-FROZEN.json').read_text())
for p,d in prep['source_and_input_sha256'].items():assert sha(p)==d,p
result=json.loads((HERE/'result.json').read_text())
assert result['status']=='VALID' and result['quality']['original_status']=='FAIL'
assert result['no_draws'] and result['no_training'] and not result['cuda_initialized']
local={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
receipt=dict(status='VALID',original_quality='FAIL',local_file_sha256=local,
             read_only_source_and_input_sha256=prep['source_and_input_sha256'],
             preparation_sha256=sha(HERE/'PREPARATION-FROZEN.json'),result_sha256=sha(HERE/'result.json'))
(HERE/'FROZEN.json').write_text(json.dumps(receipt,indent=2)+'\n')
for p,d in local.items():assert sha(HERE/p)==d,p
print(json.dumps(dict(status='VALID',result_sha256=sha(HERE/'result.json'),freeze_sha256=sha(HERE/'FROZEN.json'))))

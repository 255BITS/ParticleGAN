"""Close affected API proof after child exit; stdlib only, no reruns."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--log',type=Path,required=True)
p.add_argument('--exit-receipt',type=Path,required=True)
p.add_argument('--launch-receipt',type=Path,required=True)
p.add_argument('--root-review',type=Path,required=True)
a=p.parse_args()
i=json.loads(a.inputs.read_text());r=json.loads((a.output/'receipt.json').read_text())
e=json.loads(a.exit_receipt.read_text())
assert e['exit_code']==0 and r['status']=='PASS'
assert r['backend_schema']==10 and r['trainer_schema']==5
assert r['API_sample_calls']==3 and r['CPU_updates_total']==2
assert r['source_and_input_sha256']==i['protected_sha256']
for path,digest in i['protected_sha256'].items():assert sha(path)==digest,path
closed={str(path.resolve()):sha(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
for path in (a.inputs,a.log,a.exit_receipt,a.launch_receipt,Path(__file__).resolve()):
    closed[str(path.resolve())]=sha(path)
assert not (a.output/'FROZEN.json').exists()
frozen=dict(status='PASS',scope='Closed affected genuine backend10/schema2 API proof',
    utc=datetime.now(timezone.utc).isoformat(),receipt_sha256=sha(a.output/'receipt.json'),
    source_and_input_sha256=i['protected_sha256'],private_closed_sha256=closed,
    quality_verdict=None,new_quality_emissions=0,CUDA_initialized=False,
    API_sample_calls=3,CPU_updates_total=2,logs_closed_after_process_exit=True)
(a.output/'FROZEN.json').write_text(json.dumps(frozen,sort_keys=True,indent=2)+'\n')
assert not (a.root_review/'receipt.json').exists() and not (a.root_review/'FROZEN.json').exists(),'retain earlier root receipts'
a.root_review.mkdir(parents=True,exist_ok=True)
root_receipt=dict(r,authoritative_private_receipt=str((a.output/'receipt.json').resolve()),
    authoritative_private_freeze=str((a.output/'FROZEN.json').resolve()),
    authoritative_private_freeze_sha256=sha(a.output/'FROZEN.json'))
(a.root_review/'receipt.json').write_text(json.dumps(root_receipt,sort_keys=True,indent=2)+'\n')
root_frozen=dict(frozen,receipt_sha256=sha(a.root_review/'receipt.json'),
    root_closed_sha256={str((a.root_review/'receipt.json').resolve()):sha(a.root_review/'receipt.json'),
        str((a.output/'FROZEN.json').resolve()):sha(a.output/'FROZEN.json')})
(a.root_review/'FROZEN.json').write_text(json.dumps(root_frozen,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',receipt_sha256=sha(a.root_review/'receipt.json'),
                     FROZEN_sha256=sha(a.root_review/'FROZEN.json'))))

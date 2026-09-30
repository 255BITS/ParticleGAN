"""Authoritative closure after the sealed result writer and child both exited."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'quality/ra11/integration-contract'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

inputs=json.loads((HERE/'API-INPUTS-FROZEN.json').read_text())
receipt=json.loads((PUBLIC/'receipt.json').read_text())
exit_receipt=json.loads((HERE/'EXIT-attempt1.json').read_text())
assert receipt['status']=='PASS' and exit_receipt['exit_code']==0 and exit_receipt['process_absent']
assert not Path(f"/proc/{exit_receipt['pid']}").exists()
assert receipt['API_sample_calls']==3 and receipt['CPU_updates_total']==2
assert len(receipt['records'][0]['rejected_controls'])==12
assert receipt['continuation']['status']=='PASS' and receipt['continuation']['CPU_updates_total']==2
assert receipt['records'][0]['serving']['status']=='PASS'
assert receipt['records'][0]['phase_and_frame']['mean_moves']==914
assert receipt['records'][1]['phase_and_frame']['mean_status']=='veto'
for path,digest in inputs['protected_sha256'].items():assert sha(path)==digest,path
for area in (HERE/'accepted-attempt1',PUBLIC):
    frozen=json.loads((area/'FROZEN.json').read_text())
    for field in ('source_and_input_sha256','private_closed_sha256','root_closed_sha256'):
        for path,digest in frozen.get(field,{}).items():assert sha(path)==digest,path
private={str(path.resolve()):sha(path) for path in sorted(HERE.rglob('*')) if path.is_file()}
public={str(path.resolve()):sha(path) for path in sorted(PUBLIC.rglob('*')) if path.is_file()}
base=dict(status='PASS',scope='Post-exit authoritative affected backend10/schema2 API proof',
    utc=datetime.now(timezone.utc).isoformat(),receipt_sha256=sha(PUBLIC/'receipt.json'),
    source_and_input_sha256=inputs['protected_sha256'],private_closed_sha256=private,
    public_closed_sha256=public,input_seal_sha256=sha(HERE/'API-INPUTS-FROZEN.json'),
    root_GO_sha256=sha(HERE/'ROOT-GO-API.json'),process_absent=True,logs_closed=True,
    backend_schema=10,mean_schema=2,trainer_schema=5,API_sample_calls=3,CPU_updates_total=2,
    numerical_invocations=1,old_generic_cases_repeated=False,CUDA_initialized=False,
    quality_verdict=None,new_quality_emissions=0,
    retained_private_preparation_failure='preexecution-source/review-attempt1.log; AST keyword rendering only')
private_path=HERE/'accepted-attempt1/FINAL-FROZEN.json'
assert not private_path.exists()
private_path.write_text(json.dumps(base,sort_keys=True,indent=2)+'\n')
public_path=PUBLIC/'FINAL-FROZEN.json'
assert not public_path.exists()
public_value=dict(base,authoritative_private_final=str(private_path),
    authoritative_private_final_sha256=sha(private_path))
public_path.write_text(json.dumps(public_value,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status='PASS',receipt_sha256=sha(PUBLIC/'receipt.json'),
    FROZEN_sha256=sha(PUBLIC/'FROZEN.json'),FINAL_FROZEN_sha256=sha(public_path),
    private_FINAL_sha256=sha(private_path),protected=len(inputs['protected_sha256']),closed_private=len(private))))

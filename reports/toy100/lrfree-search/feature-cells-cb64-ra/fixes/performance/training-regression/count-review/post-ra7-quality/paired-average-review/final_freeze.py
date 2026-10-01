"""Seal the completed audit after its redirected process log has closed.

Retains the first live-log freeze and records its precise wrapper defect.
"""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
TARGET=ROOT/'quality/ra8/independent-source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n')
assert not (HERE/'FINAL-FROZEN.json').exists()
receipt=read(HERE/'receipt.json')
assert receipt['status']==receipt['review_status']=='PASS'
assert (HERE/'receipt.json').read_bytes()==(TARGET/'receipt.json').read_bytes()
for name,expected in receipt['verified_hashes'].items():
    assert sha(name)==expected,name
errors={}
for path in (HERE/'FROZEN.json',TARGET/'FROZEN.json'):
    old=read(path)
    mismatches={name:dict(recorded=expected,actual=sha(name))
                for name,expected in old['files'].items() if sha(name)!=expected}
    assert set(mismatches)=={str(HERE/'cpu-attempt2.log')},mismatches
    errors[str(path)]=mismatches
failure=HERE/'failed-freeze-boundary'
failure.mkdir()
write(failure/'receipt.json',dict(status='PRIVATE_FREEZE_WRAPPER_INVALID',
    defect='The successful audit included its redirected log before the final print. Only that closed log changed.',
    preserved_original_freezes={str(p):sha(p) for p in (HERE/'FROZEN.json',TARGET/'FROZEN.json')},
    exact_mismatches=errors,candidate_owner_receipt_and_proof_unchanged=True))
files={str(p):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
files.update({str(p):sha(p) for p in sorted(TARGET.iterdir()) if p.is_file()})
verified=receipt['verified_hashes']
authoritative=dict(status='PASS',evidence_status='VALID',
    scope='Authoritative final source/state seal after the completed helper log closed.',
    receipt_path=str(TARGET/'receipt.json'),receipt_sha256=sha(TARGET/'receipt.json'),
    files=files,reviewed_hashes=verified,
    retained_first_freeze_error=str(failure/'receipt.json'),
    first_freeze_result='Only its own live log digest changed; source/state proof remains PASS.',
    numerical_quality_verdict=None)
write(HERE/'FINAL-FROZEN.json',authoritative)
authoritative['files']={**files,str(HERE/'FINAL-FROZEN.json'):sha(HERE/'FINAL-FROZEN.json')}
write(TARGET/'FINAL-FROZEN.json',authoritative)
for name,expected in authoritative['files'].items():assert sha(name)==expected,name
print(json.dumps(dict(status='PASS',receipt_path=str(TARGET/'receipt.json'),
    receipt_sha256=sha(TARGET/'receipt.json'),freeze_path=str(TARGET/'FINAL-FROZEN.json'),
    freeze_sha256=sha(TARGET/'FINAL-FROZEN.json'),reviewed_files=len(verified))))

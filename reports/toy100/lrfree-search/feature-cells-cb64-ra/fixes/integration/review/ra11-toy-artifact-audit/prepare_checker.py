"""Seal the descriptive checker and composed RA11 identities; stdlib only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
OWNER=ROOT/'integration/review/training-regression/post-ra10-quality/linear-output-production'
ORIGINAL=ROOT/'integration/review/audit_learned.py'
PATTERN=ROOT/'performance/training-regression/count-review/ra9-prospective/audit_checkpoints.py'
PACKAGE=ROOT/'pkg-CB64-RA11'
COMPOSITION=ROOT/'quality/ra11/COMPOSITION.json'
PACKAGE_SHA='1b54cb00461df0ad89fcad59bba1e3012bf94a71ac887ecf708aafdfadc18a93'
ORIGINAL_SHA='683acf472985084a911edd6763d63d411772563d7fcb71df4a547d2a502347ba'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
    assert not (HERE/'CHECKER-FROZEN.json').exists()
    assert sha(ORIGINAL)==ORIGINAL_SHA
    composition=read(COMPOSITION);owner=read(OWNER/'READY.json')
    assert composition['package_sha256']==owner['package_sha256']==PACKAGE_SHA
    assert composition['backend_schema']==owner['backend_schema']==10
    assert composition['trainer_schema']==owner['trainer_schema']==5
    package_files={p.name:sha(p) for p in sorted((PACKAGE/'particlegan').glob('*.py'))}
    assert len(package_files)==31 and package_files==composition['source_sha256']==owner['package_source_sha256']
    digest=hashlib.sha256()
    for name in sorted(package_files):
        digest.update(name.encode()+b'\0'+(PACKAGE/'particlegan'/name).read_bytes()+b'\0')
    assert digest.hexdigest()==PACKAGE_SHA
    protected={str(p):sha(p) for p in (ORIGINAL,PATTERN,ROOT/'integration/review/ra10-toy-artifact-audit/audit_completed.py',ROOT/'integration/review/ra10-toy-artifact-audit/CHECKER-FROZEN.json',COMPOSITION,OWNER/'READY.json',OWNER/'FROZEN.json',Path(owner['config_path']))}
    for name,digest in composition['composed_from'].items():
        assert sha(name)==digest,name
        protected[name]=digest
    protected.update({str(PACKAGE/'particlegan'/name):digest for name,digest in package_files.items()})
    files={str(HERE/name):sha(HERE/name) for name in
        ('DESIGN.md','audit_completed.py','prepare_checker.py','seal_inputs.py','close_audit.py')}
    for name in ('audit_completed.py','prepare_checker.py','seal_inputs.py','close_audit.py'):
        ast.parse((HERE/name).read_text())
    frozen=dict(status='FROZEN_CPU_ONLY_DESCRIPTIVE_CHECKER',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        files=files,protected_file_sha256=protected,package_root=str(PACKAGE),package_sha256=PACKAGE_SHA,
        backend_schema=10,trainer_schema=5,original_authority_sha256=ORIGINAL_SHA,
        numerical_checkpoint_loads_before_seal=0,model_forwards=0,training_updates=0,new_quality_emissions=0,
        RA9_training_parity_asserted=False,quality_verdict=None)
    with (HERE/'CHECKER-FROZEN.json').open('x') as stream:stream.write(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status=frozen['status'],checker_sha256=sha(HERE/'audit_completed.py'),
        freeze_sha256=sha(HERE/'CHECKER-FROZEN.json'),protected_files=len(protected))))

if __name__=='__main__':main()

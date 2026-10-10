"""Bind the unchanged helper to actual frozen root metadata before any RA9 PT read."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OLD=ROOT/'integration/review/training-regression/post-ra8-quality/saved-training-parity'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    assert not (HERE/'SOURCE-FROZEN.json').exists()
    files={}
    def add(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise RuntimeError('Frozen input changed: '+str(path))
        if str(path) in files:assert files[str(path)]==actual
        files[str(path)]=actual
    for name in ('PROTOCOL.md','compare_saved.py','watch_saved.py','helper_contract.py','helper-contract.json',
                 'helper-attempt1.log','prepare_seal.py','launch_watch.py','seal_complete.py'):add(HERE/name)
    check=json.loads((HERE/'helper-contract.json').read_text())
    assert check['status']=='PASS' and check['RA9_numerical_checkpoints_read']==0
    for name in ('compare_saved.py','watch_saved.py','helper_contract.py','SOURCE-FROZEN.json','receipt.json','FROZEN.json'):
        add(OLD/name)
    identities={}
    expected_ready={'RA8':'f2f117d650dbe8eca58c0313d07b661a25ad953bd6b3b30c99507510ca9e5949',
                    'RA9':'ba558fff064f5e5613656a1c1550431091efc9989f4dd34f1d85a58d17d55990'}
    for variant in ('RA8','RA9'):
        ready_path=ROOT/f'quality/{variant.lower()}/READY.json';add(ready_path,expected_ready[variant])
        ready=json.loads(ready_path.read_text())
        for name,value in ready['numerical_source_sha256'].items():add(name,value)
        package=Path(ready['package_root'])/'particlegan'
        for name,value in ready['package_source_sha256'].items():add(package/name,value)
        add(ready['config_path'],ready['config_sha256'])
        lane=ROOT/f'validation-cb64-{variant.lower()}';freeze=lane/'source-freeze.json';add(freeze)
        law=json.loads(freeze.read_text());assert law['status']=='FROZEN_PRE_EXECUTION'
        for name,value in law['local_sources'].items():add(lane/name,value)
        for name,value in law['external_sources'].items():add(name,value)
        identities[variant]=dict(ready_path=str(ready_path),ready_sha256=sha(ready_path),package_sha256=ready['package_sha256'],
                                config_sha256=ready['config_sha256'],lane_source_freeze_sha256=sha(freeze))
    # Existing baseline states are already sealed, so bind their byte hashes
    # without interpreting them. No RA9 numerical checkpoint is opened here.
    for step in (0,100,250,500,750,1000,1250,1500,1750,2000):
        receipt_path=OLD/f'accepted-attempt1/checkpoint-{step:04d}.json';add(receipt_path)
        packet=json.loads(receipt_path.read_text())
        name=str(ROOT/f'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-{step:04d}.pt')
        add(name,packet['checkpoint_sha256'][name])
    value=dict(status='FROZEN_BEFORE_ANY_RA9_NUMERICAL_CHECKPOINT_READ',frozen_utc=datetime.now(timezone.utc).isoformat(),
               files=files,root_identities=identities,helper_AST_reuse=check,RA9_numerical_checkpoints_read=0,
               scope='CPU parse, not numerical replay or quality acceptance')
    with (HERE/'SOURCE-FROZEN.json').open('x') as target:target.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],files=len(files),source_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),
                         frozen_utc=value['frozen_utc'])),flush=True)


if __name__=='__main__':main()

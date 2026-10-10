"""Freeze the one metadata-only audit before any numerical checkpoint parse."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    assert not (HERE/'PREPARATION-FROZEN.json').exists()
    files={}
    def add(path,expected=None):
        path=Path(path).resolve();value=sha(path)
        if expected is not None:assert value==expected,str(path)
        files[str(path)]=value
    for name in ('PROTOCOL.md','inspect_metadata.py','prepare.py','seal.py'):add(HERE/name)
    design=HERE.parent/'local-moment-design'
    seal=json.loads((design/'FROZEN.json').read_text());add(design/'FROZEN.json')
    for path,value in seal['files'].items():add(path,value)
    for path,value in seal['source_sha256'].items():add(path,value)
    lane=ROOT/'validation-cb64-ra9';add(lane/'source-freeze.json')
    screen=lane/'screens';add(screen/'source-freeze.json')
    run=screen/'runs/grid100'
    for name in ('final-state.pt','execution-receipt.json','result.json','job-header.json'):
        add(run/name)
    add('/ml2/hypergan/lrfree-20260926/harness/screen.py')
    value=dict(status='FROZEN_BEFORE_FINAL_NATIVE_PT_METADATA_PARSE',frozen_utc=datetime.now(timezone.utc).isoformat(),
               files=files,checkpoint_numerical_reads=0,model_forwards=0,new_actions=0,new_emissions=0,new_training_steps=0,
               scope='one CPU metadata parse, not replay/geometry or quality measurement',quality_verdict=None)
    with (HERE/'PREPARATION-FROZEN.json').open('x') as target:target.write(json.dumps(value,indent=2)+'\n')
    state=run/'final-state.pt'
    print(json.dumps(dict(status=value['status'],files=len(files),preparation_sha256=sha(HERE/'PREPARATION-FROZEN.json'),
                         helper_sha256=sha(HERE/'inspect_metadata.py'),state_sha256=sha(state),frozen_utc=value['frozen_utc'])),flush=True)


if __name__=='__main__':main()

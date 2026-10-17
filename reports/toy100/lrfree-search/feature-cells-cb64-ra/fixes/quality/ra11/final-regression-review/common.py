"""Stdlib raw-byte bindings for the final RA11 artifact review."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
LANE=ROOT/'validation-cb64-ra11'
PACKAGE=ROOT/'pkg-CB64-RA11'
MONITOR=ROOT/'integration/review/validation-cb64-ra11-monitor'
WATCHER=ROOT/'performance/training-regression/count-review/post-ra10-quality/ra11-canonical-watcher'
ORIGINAL=ROOT/'integration/review/audit_learned.py'
VARIANT='CB64-RA11'
STEPS=(0,100,250,500,750,1000,1250,1500,1750,2000)
GPU='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
PACKAGE_SHA='1b54cb00461df0ad89fcad59bba1e3012bf94a71ac887ecf708aafdfadc18a93'
CONFIG_SHA='b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
AUTHORITY_NAMES=('read','sha','write','require','verify_sources','load_cpu','digest','semantic',
                 'metadata','rng_placement','verify_runtime','audit_training','audit_replay')
PROOF_SEALS=(
 ROOT/'quality/ra11/integration-contract/FINAL-FROZEN.json',
 ROOT/'quality/ra11/grid-gate-review/FROZEN.json',
 ROOT/'integration/review/ra11-toy-artifact-audit/accepted-attempt1/FINAL-FROZEN.json',
 ROOT/'integration/review/ra11-grid-artifact-audit/accepted-attempt1/FINAL-FROZEN.json',
 ROOT/'quality/ra11/lane-review/FROZEN.json',
 ROOT/'quality/ra11/independent-source/FROZEN.json',
 WATCHER/'PREPARATION-FROZEN.json')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def write_new(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def pin(mapping,path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    assert expected is None or digest==expected,str(path)
    assert str(path) not in mapping or mapping[str(path)]==digest,str(path)
    mapping[str(path)]=digest
    return digest
def verify(mapping):
    for path,digest in mapping.items():assert sha(path)==digest,path
def pin_absolute_maps(mapping,value):
    if isinstance(value,dict):
        for path,digest in value.items():
            if isinstance(path,str) and path.startswith('/') and isinstance(digest,str) and len(digest)==64:
                pin(mapping,path,digest)
            elif isinstance(path,str) and path.startswith('/') and isinstance(digest,dict) and isinstance(digest.get('sha256'),str):
                pin(mapping,path,digest['sha256'])
            else:pin_absolute_maps(mapping,digest)
    elif isinstance(value,list):
        for item in value:pin_absolute_maps(mapping,item)
def nodes(path,names):
    selected=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    return selected
def ast_sha(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def process_identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return dict(pid=int(pid),present=False)
    text=path.read_text();tail=text[text.rindex(')')+2:].split()
    return dict(pid=int(pid),present=True,state=tail[0],startticks=tail[19])
def require_exited(pid,startticks):
    value=process_identity(pid)
    assert not value['present'] or value['startticks']!=str(startticks) or value['state']=='Z',value
    return value

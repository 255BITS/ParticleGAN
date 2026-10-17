"""Seal only closed initial endpoint receipts; never the live watcher log."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'accepted-attempt1'
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n')
assert not (HERE/'INITIAL-FROZEN.json').exists()
checked={}
def check_map(values):
    for name,expected in values.items():
        assert sha(name)==expected,name
        checked[name]=expected
for name in ('CHECKER-FROZEN.json','accepted-attempt1/SOURCE-FROZEN.json'):
    path=HERE/name;record=read(path)
    for key in ('files','reviewed_hashes','original_checker'):
        if key in record:check_map(record[key])
    checked[str(path)]=sha(path)
source=read(OUT/'SOURCE-RECEIPT.json')
assert source['status']=='VALID' and source['backend_schema']==7 and source['trainer_schema']==5
rows=[]
for step in (0,100,250):
    path=OUT/f'checkpoint-{step:04d}-FROZEN.json';freeze=read(path)
    assert freeze['status']=='VALID'
    for key in ('files','artifacts'):check_map(freeze[key])
    checked[str(path)]=sha(path)
    row=read(OUT/f'checkpoint-{step:04d}.json')
    assert row['status']=='VALID' and row['quality_verdict'] is None
    assert row['optimizer_updates']==row['new_seeds']==0 and not row['cuda_initialized']
    assert row['paired_average']['rejected_atomic_controls']==['old-backend6','wrong-geometry-policy','boolean-step','future-step']
    rows.append(dict(step=step,paired_average=row['paired_average'],population=row['population'],serving=row['serving']))
for name in ('initial_guard.py','initial-attempt1.log','start_watcher.py','WATCHER.json'):
    checked[str(HERE/name)]=sha(HERE/name)
receipt=dict(status='VALID',scope='Initial sealed RA8 saved-state endpoints; no quality qualification',
    source_guards=len(source['reviewed_hashes']),sealed_steps=[0,100,250],endpoints=rows,
    source_ready_sha256='f2f117d650dbe8eca58c0313d07b661a25ad953bd6b3b30c99507510ca9e5949',
    lane_freeze_sha256='1292ef86f6b16d8928923fda267fb9f1efc6d8ac746a1f091a01397e032a3cf3',
    watcher=read(HERE/'WATCHER.json'),verified_hashes=checked,
    quality_verdict=None,target='Both original strict toy and full Grid100 required.',
    model_forwards=0,gradients=0,optimizer_updates=0,new_emissions=0,new_seeds=0,cuda_initialized=False,
    limits='Saved scalar lease is empirical bounded-stale geometry, not stationarity/equivalence or emitted support.')
write(HERE/'INITIAL-RECEIPT.json',receipt)
write(HERE/'INITIAL-FROZEN.json',dict(status='VALID',files={str(HERE/'INITIAL-RECEIPT.json'):sha(HERE/'INITIAL-RECEIPT.json')},reviewed_hashes=checked))
print(json.dumps(dict(status='VALID',receipt_sha256=sha(HERE/'INITIAL-RECEIPT.json'),freeze_sha256=sha(HERE/'INITIAL-FROZEN.json'))))

"""Narrow helper preflight without reading any RA9 numerical checkpoint."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import json
import torch
from compare_saved import HERE,ORIGINAL,POLICY,EXPIRY,RESOLUTION,STAMP_KEYS,legacy_view,differences,sha

old=ast.parse((ORIGINAL/'compare_saved.py').read_text())
new=ast.parse((HERE/'compare_saved.py').read_text())
names=('sha','tensor_bytes','digest','describe','differences')
def functions(tree):
    return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
a,b=functions(old),functions(new)
assert all(a[name]==b[name] for name in names)
stamp={key:0 for key in STAMP_KEYS}
stamp.update(schema=1,policy=POLICY,rows=1024,required=973,chart_valid=False,duplicate_ok=False,eligible=False)
reference=dict(schema=5,completed_steps=0,recipe=dict(birth_death_cells=64),
    models=dict(prior=dict(z=torch.zeros(1024,2))),
    birth_death=dict(backend_schema=7,settings=dict(cells=64,rank=8,paired_average_policy=POLICY,paired_average_expiry=EXPIRY),
                     paired_average=stamp,snapshot_serial=0,rows_since_eval=0,fill=0,last={},counters=dict(ordinary_moves=0)))
candidate=deepcopy(reference);bd=candidate['birth_death'];bd['backend_schema']=8
bd['settings'].update(cells=128,resolution_policy=RESOLUTION);candidate['recipe']['birth_death_cells']=128
r=legacy_view(reference,'RA8')[0];c=legacy_view(candidate,'RA9')[0]
assert not differences(r,c)
bad=deepcopy(candidate);bad['birth_death']['counters']['ordinary_moves']+=1
assert differences(r,legacy_view(bad,'RA9')[0])[0]['path']=='$/birth_death/counters/ordinary_moves'
bad=deepcopy(candidate);bad['models']['prior']['z'][0,0]=1
assert differences(r,legacy_view(bad,'RA9')[0])[0]['path']=='$/models/prior/z'
assert not torch.cuda.is_initialized()
record=dict(status='PASS',unchanged_original_comparator_AST_functions=list(names),focused_allowlist_controls=3,
            original_comparator_sha256=sha(ORIGINAL/'compare_saved.py'),RA9_numerical_checkpoints_read=0,
            CPU_only=True,cuda_initialized=False,new_training_steps=0,new_emissions=0,numerical_replay=False,
            quality_verdict=None,completed_utc=datetime.now(timezone.utc).isoformat())
with (HERE/'helper-contract.json').open('x') as target:target.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)

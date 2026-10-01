"""Reuse frozen 3K inputs; add predetermined no-signal/opposite count controls."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent
source=HERE.parent/'joint-count/inputs.pt'
expected='d8cb4123942985797be0c491650894163c00d76c05f4dbdd81322919039da5cd'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==expected
holder=ModuleType('global_prepare');holder.__path__=[str(HERE/'pkg-global-count/particlegan')]
sys.modules[holder.__name__]=holder
module=importlib.import_module(holder.__name__+'.feature_cells')
data=torch.load(source,map_location='cpu',weights_only=False)
base=data['cases']['saved_toy_1000']
no_signal=deepcopy(base)
no_signal['fake_features']=base['real_features'][1::2].repeat((2,1))
no_signal['count_sample_scope']='mechanical exact heldout-frequency control; not iid training evidence'
data['cases']['global_no_signal']=no_signal
snap=module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
snap.__dict__.update(deepcopy(base['snapshot']))
inside=(snap.count_categories(base['real_features'][0::2]).remainder(2)==0).nonzero().flatten()
assert len(inside)>0
opposite=deepcopy(base)
opposite['fake_features']=base['real_features'][0::2][inside[0]].expand(len(base['q']),-1).clone()
opposite['count_sample_scope']='mechanical opposite global support signal, existing even-reference row; not iid training evidence'
data['cases']['global_opposite_signal']=opposite
output=HERE/'inputs.pt'
assert not output.exists()
torch.save(data,output)
assert sha(source)==expected and not torch.cuda.is_initialized()
receipt=dict(status='PASS',source_3K_inputs=str(source),source_3K_sha256=expected,
    input_sha256=sha(output),inherited_cases=7,added_cases=['global_no_signal','global_opposite_signal'],
    fitting_changes=0,new_seeds=0,optimizer_updates=0,cuda_initialized=False)
(HERE/'prepare-inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)

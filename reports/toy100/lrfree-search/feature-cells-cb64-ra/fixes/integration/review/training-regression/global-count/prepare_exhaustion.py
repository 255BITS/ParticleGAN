"""One predetermined global quota exhaustion, using the existing two-cell geometry."""
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
holder=ModuleType('global_exhaustion');holder.__path__=[str(HERE/'pkg-global-count/particlegan')]
sys.modules[holder.__name__]=holder
module=importlib.import_module(holder.__name__+'.feature_cells')
data=torch.load(HERE/'inputs.pt',map_location='cpu',weights_only=False)
base=data['cases']['saved_toy_1000']
rng=torch.Generator().set_state(base['planning_rng'])
def cloud(center,rows,width):
    return torch.stack((torch.linspace(center-width,center+width,rows,dtype=torch.float64),
        torch.zeros(rows,dtype=torch.float64)),1)
even=torch.cat((cloud(-3.,154,.04),cloud(3.,358,.04)))
odd=torch.cat((cloud(-3.,26,.017),cloud(3.,486,.017)))
real=torch.empty((1024,2),dtype=torch.float64);real[0::2]=even;real[1::2]=odd
snap=module.FeatureCellSnapshot.fit(real,generator=rng,cells=2,rank=2,chunk=256)
left,right,outside=(torch.tensor([p],dtype=torch.float64) for p in ((-3.,0.),(3.,0.),(6.,0.)))
q=torch.cat((left.expand(64,-1),right.expand(256,-1),outside.expand(704,-1))).clone()
fake=torch.cat((left.expand(20,-1),right.expand(972,-1),outside.expand(32,-1))).clone()
flags,pvalues,_=snap.support(q)
assert int(flags.sum())==704 and bool((pvalues[:320]>.05).all())
snap.cache_queries(q)
value=dict(snapshot=deepcopy(vars(snap)),q=q,flags=flags,pvalues=pvalues,real_features=real,
    fake_features=fake,seed=base['seed'],planning_rng=rng.get_state(),
    count_sample_scope='specified global exhaustion contract: coarse left32 and globaloutside32, not a quality trajectory')
output=HERE/'exhaustion-input.pt';assert not output.exists()
torch.save(dict(cases={'global_certificate_exhaustion':value}),output)
assert not torch.cuda.is_initialized()
receipt=dict(status='PREPARED',input_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    parent_input_sha256=hashlib.sha256((HERE/'inputs.pt').read_bytes()).hexdigest(),
    specified_even_cell_rows=[154,358],specified_odd_cell_rows=[26,486],
    clean_query_cell_rows=[64,256],flagged_outside_rows=704,
    fake_inside_cell_rows=[20,972],fake_outside_rows=32,new_seeds=0,optimizer_updates=0,cuda_initialized=False)
(HERE/'prepare-exhaustion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)

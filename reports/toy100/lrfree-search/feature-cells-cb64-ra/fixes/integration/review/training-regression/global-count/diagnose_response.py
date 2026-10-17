"""Fixed saved-snapshot residual action capacity; no level or seed selection."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from contract_cases import run_plan,plain
HERE=Path(__file__).resolve().parent
holder=ModuleType('global_diagnosis');holder.__path__=[str(HERE/'pkg-global-count/particlegan')]
sys.modules[holder.__name__]=holder
module=importlib.import_module(holder.__name__+'.feature_cells')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[Path(__file__),HERE/'contract_cases.py',HERE/'inputs.pt',HERE/'pkg-global-count/particlegan/feature_cells.py']
before={str(p):sha(p) for p in sources}
data=torch.load(HERE/'inputs.pt',map_location='cpu',weights_only=False)
rows=[]
for name in ('saved_toy_1000','saved_toy_2000'):
    snap,q,flags,pvalues,comparison,child,parent,detail,iso_child,iso_parent,isolation,*_=run_plan(module,data['cases'][name],'cpu')
    categories=detail['query_category_ids']
    global_parents=detail['global_parents']
    phase=detail['global_phase']
    local_deficit=comparison['support']['deficit'][categories[global_parents]]
    observed_inside_deficit=comparison['support']['difference'][categories[global_parents]]<0
    reserved=torch.zeros(len(q),dtype=torch.bool);reserved[child]=True;reserved[parent]=True
    eligible=(~flags)&(pvalues>.05)&(categories.remainder(2)==0)&~reserved
    ids=detail['query_cell_ids']
    parents_by_cell=torch.bincount(ids[eligible],minlength=snap.cells)
    cell_vacancies=(detail['target_counts']-detail['planned_supported_counts']).clamp_min(0)
    physical=torch.minimum(parents_by_cell.clamp_max(module.PARENT_RESERVOIR),cell_vacancies)
    group_vacancies=(snap._group_counts(detail['target_counts'])-snap._group_counts(detail['planned_supported_counts'])).clamp_min(0)
    physical_capacity=int(torch.minimum(snap._group_counts(physical),group_vacancies).sum())
    row=dict(name=name,mass_moves=detail['mass_moves'],local_moves=detail['support_moves'],global_moves=detail['global_moves'],
        ordinary_moves=len(child),ordinary_budget=detail['budget'],remaining_ordinary_budget=detail['budget']-len(child),
        isolation_moves=len(iso_child),global_real_counts=comparison['global_support']['real_counts'],
        global_fake_counts=comparison['global_support']['fake_counts'],global_pvalues=comparison['global_support']['pvalues'],
        raw_global_death_quota=phase['raw_certified_death_capacity'],spent_global_death_quota=phase['spent_certified_death_capacity'],
        residual_global_death_before_phase=phase['residual_certified_death_capacity'],
        raw_global_birth_quota=phase['raw_certified_birth_capacity'],spent_global_birth_quota=phase['spent_certified_birth_capacity'],
        residual_global_birth_before_phase=phase['residual_certified_birth_capacity'],
        global_births_without_local_discovery=int((~local_deficit).sum()),
        global_births_in_observed_inside_deficit_without_local_discovery=int((observed_inside_deficit&~local_deficit).sum()),
        global_birth_cells_without_local_discovery=len(torch.unique(ids[global_parents[~local_deficit]])),
        remaining_unused_eligible_inside_rows=int(eligible.sum()),remaining_physical_birth_capacity=physical_capacity,
        remaining_flagged_rows=int(flags.sum())-int(flags[child].sum()),
        limiting_ordinary_constraint='shared floor(Q*N) budget after aggregate phase')
    rows.append(plain(row));print(json.dumps(plain(row)),flush=True)
assert before=={p:sha(Path(p)) for p in before} and not torch.cuda.is_initialized()
receipt=dict(status='PASS',rows=rows,sources_sha256=before,sources_unchanged=True,
    cuda_initialized=False,new_seeds=0,optimizer_updates=0,
    scope='saved fixed-input count response and residual capacity, no distribution oracle or quality verdict')
(HERE/'response-diagnosis.json').write_text(json.dumps(receipt,indent=2)+'\n')

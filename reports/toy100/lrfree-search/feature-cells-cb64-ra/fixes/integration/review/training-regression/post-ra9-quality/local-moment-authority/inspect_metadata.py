"""Single CPU metadata parse: does final RA9 have certified copy slots to refine?"""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
STATE=ROOT/'validation-cb64-ra9/screens/runs/grid100/final-state.pt'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024*1024):value.update(block)
    return value.hexdigest()


def guard():
    manifest=json.loads((HERE/'PREPARATION-FROZEN.json').read_text())
    failures=[path for path,value in manifest['files'].items() if sha(path)!=value]
    if failures:raise RuntimeError('Frozen source/input changed: '+repr(failures))
    return manifest


def scalar_tree(value):
    # Persisted diagnostics should already be scalar JSON data. No tensor values
    # are inspected, converted or evaluated by this metadata-only helper.
    if value is None or type(value) in (str,int,float,bool):return value
    if type(value) in (list,tuple):return [scalar_tree(v) for v in value]
    if type(value) is dict:
        assert all(type(k) is str for k in value)
        return {k:scalar_tree(v) for k,v in value.items()}
    raise TypeError('Non-scalar persisted metadata: '+type(value).__name__)


def main():
    assert not (HERE/'attempt1/result.json').exists()
    manifest=guard();rng=torch.get_rng_state().clone()
    packet=torch.load(STATE,map_location='cpu',weights_only=False)
    trainer=packet['trainer'];bd=trainer['birth_death'];last=bd['last']
    assert trainer['schema']==5 and bd['backend_schema']==8
    n=trainer['models']['prior']['z'].shape[0]
    assert n==trainer['recipe']['num_particles']==20000
    assert trainer['completed_steps']==7000 and bd['settings']['cells']==128
    assert bd['settings']['resolution_policy']=='even_fit_average_rows_per_effective_rank_floor1_v1'
    assert type(last['step']) is int and 0<last['step']<=trainer['completed_steps']
    assert type(last['snapshot']) is int and last['snapshot']==bd['snapshot_serial']
    for name in ('ordinary_copy_moves','ordinary_novel_birth_moves','ordinary_moves','iso_moves','moves',
                 'ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_budget',
                 'ordinary_discoveries','ordinary_excess_cells','ordinary_deficit_cells'):
        assert type(last[name]) is int and last[name]>=0
    assert last['ordinary_copy_moves']+last['ordinary_novel_birth_moves']==last['ordinary_moves']
    assert last['ordinary_moves']+last['iso_moves']==last['moves']
    assert last['ordinary_moves']<=last['ordinary_budget']<=math.floor(.05*n)
    assert type(last['cells']) is int and last['cells']==min(128,max(1,((n+1)//2)//max(1,last['metric_rank'])))
    assert last['count_categories']==2*last['cells'] and last['count_multiplicity']==3*last['cells']+2
    assert last['count_cutoff']==.05/(3*last['cells']+2)
    slots=last['ordinary_copy_moves']
    decision='REJECT_NO_FINAL_REACTION_COPY_AUTHORITY' if slots==0 else 'RELEVANT_GROUP_AUTHORITY_NOT_IDENTIFIABLE_FROM_SAVED_METADATA'
    metadata=scalar_tree(last);counters=scalar_tree(bd['counters']);stamp=scalar_tree(bd['paired_average'])
    guard();assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    value=dict(status='PASS_METADATA_AUDIT',proposal_decision=decision,completed_steps=trainer['completed_steps'],
        inspected_reaction_step=last['step'],snapshot_serial=bd['snapshot_serial'],ordinary_copy_slots=slots,
        ordinary_novel_births=last['ordinary_novel_birth_moves'],shared_ordinary_budget=last['ordinary_budget'],
        last_reaction=metadata,cumulative_counters=counters,paired_average=stamp,
        actual_cells=last['cells'],metric_rank=last['metric_rank'],groups=last['mass_topology']['groups'],
        conditional_family_multiplicity=last['count_multiplicity'],count_cutoff=last['count_cutoff'],
        group_targeting_identifiable=False,
        scope='One final saved reaction metadata parse; no conclusion about unrecorded earlier group authority',
        unavailable=('historical fitted chart and group/cell mapping','ordinary per-cell quotas and copy child/parent IDs'),
        preserved_count_law=True,new_actions=0,new_training_steps=0,new_optimizer_steps=0,new_emissions=0,
        geometry_refits=0,model_forwards=0,CUDA_used=False,cuda_initialized=False,CPU_only=True,
        rng_draws=0,rng_restores=0,global_CPU_rng_unchanged=True,numerical_replay=False,
        preparation_sha256=sha(HERE/'PREPARATION-FROZEN.json'),preparation_frozen_utc=manifest['frozen_utc'],
        source_input_sha256=manifest['files'],completed_utc=datetime.now(timezone.utc).isoformat(),quality_verdict=None)
    out=HERE/'attempt1';out.mkdir(exist_ok=False)
    with (out/'result.json').open('x') as target:target.write(json.dumps(value,indent=2,allow_nan=True)+'\n')
    print(json.dumps(dict(status=value['status'],proposal_decision=decision,step=last['step'],snapshot=last['snapshot'],
                         copy_slots=slots,cells=last['cells'],rank=last['metric_rank'],groups=last['mass_topology']['groups'],
                         discoveries=last['ordinary_discoveries'],ordinary_moves=last['ordinary_moves'],
                         result_sha256=sha(out/'result.json'))),flush=True)


if __name__=='__main__':main()

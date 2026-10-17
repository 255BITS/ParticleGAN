"""Fixed learned geometry and typed backend7 controls; CPU only."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
PACKAGE=HERE/'pkg-PAIR-AVERAGE'
MAIN=ROOT/'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py'
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(ROOT/'integration/review/training-regression/post-ra4-quality'))
from particlegan import feature_cells as module
from measure_saved_utils import tensor_state_hash
from contract_utils import fixture,ORIGINAL
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=HERE/'gate-receipt.json'
if out.exists():raise SystemExit('Preserve existing receipt.')
sources={str(p):sha(p) for p in [*sorted(PACKAGE.rglob('*.py')),Path(__file__),HERE/'PROTOCOL.md',
    HERE/'contract_utils.py',MAIN,ORIGINAL]}
paths=[ROOT/f'validation-cb64-ra7/learned/training/toy/CB64-RA7/checkpoint-{s:04d}.pt' for s in (500,1000,2000)]
inputs={str(p):sha(p) for p in paths}
namespace=dict(torch=torch,F=F)
definitions=[n for n in ast.parse(MAIN.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='forward']
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(MAIN),'exec'),namespace)
forward=namespace['forward'];rng=torch.get_rng_state().clone();records=[];checks={}
for path in paths:
    state=torch.load(path,map_location='cpu',weights_only=False)['trainer'];before=tensor_state_hash(state)
    w=state['models'];bd_state=state['birth_death']
    with torch.no_grad():
        real=forward(bd_state['reservoir'],w['D'],head=True).double()
        fast=forward(forward(w['prior']['z'],w['G']),w['D'],head=True).double()
        ema=forward(forward(w['ema_prior']['z'],w['ema_G']),w['D'],head=True).double()
        snap=module.FeatureCellSnapshot.fit(real,generator=torch.Generator().set_state(state['cpu_rng']),
            cells=bd_state['settings']['cells'],rank=bd_state['settings']['rank'],chunk=bd_state['settings']['chunk'])
        snap.cache_queries(fast)
        stamp=module.paired_average_geometry(snap,ema,step=state['completed_steps'],snapshot_serial=bd_state['snapshot_serial'])
    assert before==tensor_state_hash(state)
    records.append(stamp)
    print(json.dumps({'event':'saved_fixed_gate',**stamp}),flush=True)
checks['fixed_early_veto_final_pass']=[r['eligible'] for r in records]==[False,False,True]
checks['fixed_intersection_counts']=[r['coherent_rows'] for r in records]==[504,649,985]
checks['fixed_same_group_counts']=[r['same_group_rows'] for r in records]==[944,955,1018]

# Final saved chart, fixed input controls only. Pick the first even real
# eligible row in each real-only group, without labels or query fitting.
categories=snap.count_categories(real);ref_group=snap._mass_topology()[categories//2]
pvalues=snap.support(real)[1]
inside=(categories%2==0)&(pvalues>.05)
choices=torch.stack([((ref_group==g)&inside).nonzero().flatten()[0] for g in range(snap.mass_groups)])
fast_groups=snap._mass_topology()[snap.query_cell_ids]
good=real[choices[fast_groups]].clone()
good_stamp=module.paired_average_geometry(snap,good,step=2000,snapshot_serial=250)
checks['fully_coherent_supported_control']=good_stamp['coherent_rows']==1024 and good_stamp['eligible']
quota_controls=[]
for exceptions in (51,52):
    wrong=good.clone()
    wrong[:exceptions]=real[choices[(fast_groups[:exceptions]+1)%snap.mass_groups]]
    result=module.paired_average_geometry(snap,wrong,step=2000,snapshot_serial=250)
    quota_controls.append(result)
checks['shared_exception_boundary']=quota_controls[0]['coherent_rows']==973 and quota_controls[0]['eligible']
checks['one_over_exception_budget_veto']=quota_controls[1]['coherent_rows']==972 and not quota_controls[1]['eligible']
checks['wrong_group_stays_supported']=all(r['ema_eligible_rows']==1024 for r in quota_controls)
collapsed=real[choices[0]].expand_as(good)
collapse=module.paired_average_geometry(snap,collapsed,step=2000,snapshot_serial=250)
checks['supported_group_collapse_veto']=collapse['ema_eligible_rows']==1024 and not collapse['eligible']
overshoot=snap.mean[None]+100*(good-snap.mean[None])
over=module.paired_average_geometry(snap,overshoot,step=2000,snapshot_serial=250)
checks['large_unsupported_overshoot_veto']=not over['eligible'] and over['ema_eligible_rows']<973
nonfinite=ema.clone();nonfinite[0,0]=float('nan')
badfinite=module.paired_average_geometry(snap,nonfinite,step=2000,snapshot_serial=250)
checks['nonfinite_veto_zero_branch']=not badfinite['eligible'] and badfinite['finite_rows']==1023 and all(badfinite[k]==0 for k in ('groups','coherent_rows','ema_eligible_rows','same_group_rows'))
duplicate=deepcopy(snap);duplicate.duplicate_fraction=.1
dup=module.paired_average_geometry(duplicate,good,step=2000,snapshot_serial=250)
checks['duplicate_chart_veto']=not dup['duplicate_ok'] and not dup['eligible']
degenerate=deepcopy(snap);degenerate.valid_metric=False;degenerate.rank=0
deg=module.paired_average_geometry(degenerate,good,step=2000,snapshot_serial=250)
checks['invalid_chart_veto']=not deg['chart_valid'] and not deg['eligible']

trainer,backend=fixture(state,module)
backend.rows_since_eval=0
backend.paired_average=dict(records[-1])
backend.last=dict(bd_state['last'])
backend.last.update(step=2000,snapshot=250,cells=snap.cells,metric_rank=snap.rank,
    calibration_rows=snap.calibration_rows,duplicate_fraction=snap.duplicate_fraction,
    mass_topology=dict(snap.mass_topology),paired_average=dict(backend.paired_average))
fresh=backend.state_dict();backend.check_state(fresh);backend.check_paired_average_step(fresh,2000)
checks['initial_lease_live']=backend.paired_average_eligible(2000)
checks['future_decision_not_served']=not backend.paired_average_eligible(1999)
backend.rows_since_eval=1023;checks['strict_before_turnover_live']=backend.paired_average_eligible(2007)
backend.rows_since_eval=1024;checks['at_turnover_expired']=not backend.paired_average_eligible(2008)
backend.rows_since_eval=2048;checks['turnover_overshoot_expired']=not backend.paired_average_eligible(2016)
backend.rows_since_eval=0;backend.snapshot_serial+=1
checks['superseded_snapshot_expired']=not backend.paired_average_eligible(2000)
backend.snapshot_serial-=1
_,loaded=fixture(state,module);loaded.load_state_dict(fresh)
checks['load_stamp_live_without_ephemeral_chart']=loaded.snapshot is None and loaded.paired_average_eligible(2000)
checks['load_roundtrip_all_backend_state']=tensor_state_hash(loaded.state_dict())==tensor_state_hash(fresh)
before=tensor_state_hash(loaded.state_dict());rejected=False
try:loaded.load_state_dict(bd_state)
except ValueError:rejected=True
checks['old6_load_atomic']=rejected and before==tensor_state_hash(loaded.state_dict())
malformed=[]
def invalid(name,change,step=None):
    bad=deepcopy(fresh);change(bad)
    before=tensor_state_hash(loaded.state_dict());rejected=False
    try:
        loaded.load_state_dict(bad) if step is None else loaded.check_paired_average_step(bad,step)
    except ValueError:rejected=True
    passed=rejected and before==tensor_state_hash(loaded.state_dict())
    checks['invalid_'+name+'_atomic']=passed;malformed.append(name)

def stamp_change(**values):
    def change(state):
        state['paired_average'].update(values)
        state['last']['paired_average']=dict(state['paired_average'])
    return change
invalid('missing_key',lambda s:s['paired_average'].pop('coherent_rows'))
invalid('bool_count',stamp_change(coherent_rows=True))
invalid('false_eligible',stamp_change(eligible=False))
invalid('too_many_rows',stamp_change(ema_eligible_rows=1025))
invalid('intersection_lower_bound',stamp_change(same_group_rows=1024,ema_eligible_rows=1024,coherent_rows=0,eligible=False))
invalid('intersection_upper_bound',stamp_change(same_group_rows=984))
invalid('nonfinite_nonzero_branch',stamp_change(finite_rows=1023,eligible=False))
invalid('false_chart_rank',stamp_change(chart_valid=False,eligible=False))
invalid('duplicate_disagrees',stamp_change(duplicate_ok=False,eligible=False))
invalid('chart_cells_disagrees',stamp_change(cells=63))
invalid('chart_groups_disagrees',stamp_change(groups=24))
invalid('snapshot_disagrees',stamp_change(snapshot=249))
invalid('wrong_policy',stamp_change(policy='other'))
invalid('wrong_schema',stamp_change(schema=2))
invalid('wrong_requirement',stamp_change(required=972))
invalid('future_step',stamp_change(step=2001),step=2000)

# Typed boundary only: odd801 has401 fitted even references and400 calibration
# rows. No large401-centre fit or new random input is needed for this contract.
odd=module.FeatureCellBirthDeath.__new__(module.FeatureCellBirthDeath)
odd.N=801;odd.settings=dict(backend.settings,cells=401);odd.paired_average=dict(backend.paired_average)
odd_stamp=dict(backend.paired_average,rows=801,required=761,cells=401,calibration_rows=400,
    finite_rows=801,same_group_rows=801,ema_eligible_rows=801,coherent_rows=801)
odd_state=dict(snapshot_serial=250,paired_average=odd_stamp,last=dict(backend.last,cells=401,
    calibration_rows=400,paired_average=odd_stamp))
odd._check_paired_average_state(odd_state)
checks['odd_fit_ceil_calibration_floor_valid']=True
odd_bad=deepcopy(odd_state);odd_bad['paired_average']['cells']=402;odd_bad['last']['cells']=402
try:odd._check_paired_average_state(odd_bad);checks['odd_one_above_fit_bound_rejects']=False
except ValueError:checks['odd_one_above_fit_bound_rejects']=True
checks['small_reference_no_new_gate_method']=not hasattr(module.SmallPopulationReferenceBirthDeath,'paired_average_eligible')
checks['small_reference_policy_retained']=not module.population_policy(128)['finite_resolution_feasible'] and not module.population_policy(799)['finite_resolution_feasible'] and module.population_policy(801)['finite_resolution_feasible']
checks['strict_gate_intersection_source']=all(r['required']==973 for r in records)
assert all(checks.values()),[k for k,v in checks.items() if not v]
assert sources=={p:sha(Path(p)) for p in sources} and inputs=={p:sha(Path(p)) for p in inputs}
assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
out.write_text(json.dumps(dict(status='PASS',records=records,checks=checks,malformed_controls=malformed,
    fixed_quota_controls=quota_controls,source_sha256=sources,input_sha256=inputs,
    all_sources_inputs_checkpoint_tensors_and_global_rng_unchanged=True,cpu_only=True,cuda_initialized=False,
    new_training_steps=0,new_optimizer_steps=0,new_emissions=0,new_seed_experiments=0,quality_verdict=None,
    limits=['geometry anti-blur only; no population/equivalence/quality certificate',
        'turnover lease is bounded stale, no every-update support guarantee',
        'CPU current-chart diagnostics are not historical GPU replay']),indent=2)+'\n')
print(json.dumps({'event':'gate_contract_complete','status':'PASS','checks':len(checks),'receipt':str(out)}),flush=True)

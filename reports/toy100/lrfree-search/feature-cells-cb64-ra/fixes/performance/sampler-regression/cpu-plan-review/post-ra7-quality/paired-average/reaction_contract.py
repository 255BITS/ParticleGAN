"""One matched CPU reaction per saved input; no optimizer/training steps."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise SystemExit('Preserve existing receipt.')
sys.path.insert(0,str(args.package_root))
sys.path.insert(0,str(ROOT/'integration/review/training-regression/post-ra4-quality'))
from particlegan import feature_cells as module
from measure_saved_utils import tensor_state_hash
from contract_utils import fixture, ORIGINAL
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_paths=[*sorted(args.package_root.rglob('*.py')),Path(__file__),HERE/'contract_utils.py',HERE/'PROTOCOL.md',ORIGINAL]
sources={str(p):sha(p) for p in source_paths}
paths=[ROOT/f'validation-cb64-ra7/learned/training/toy/CB64-RA7/checkpoint-{s:04d}.pt' for s in (1250,2000)]
inputs={str(p):sha(p) for p in paths}
rng=torch.get_rng_state().clone();records=[]
method=module.FeatureCellSnapshot.ordinary_transport
for path in paths:
    state=torch.load(path,map_location='cpu',weights_only=False)['trainer'];original=tensor_state_hash(state)
    trainer,bd=fixture(state,module)
    for root in (trainer.G,trainer.D,trainer.ema_G):
        root.register_buffer('owner_neutral_marker',torch.ones(3))
        for p in root.parameters():p.grad=torch.ones_like(p)
    trainer.prior.z.grad=torch.ones_like(trainer.prior.z)
    trainer.ema_G.train();trainer.ema_G[1].eval()
    models=[trainer.G,trainer.D,trainer.ema_G]
    weights_before=[tensor_state_hash(m.state_dict()) for m in models]
    gradients_before=[tensor_state_hash([p.grad for p in m.parameters()]) for m in models]+[tensor_state_hash(trainer.prior.z.grad)]
    modes=[(m,m.training) for root in models for m in root.modules()]
    capture={};copies=[]
    def traced(snapshot,q,flags,comparison,**kwargs):
        result=method(snapshot,q,flags,comparison,**kwargs)
        capture.update(detail=deepcopy(result[2]),q=q.clone(),flags=flags.clone())
        return result
    module.FeatureCellSnapshot.ordinary_transport=traced
    old_move=bd._move
    def moved(t,c,p):
        copies.append((c.clone(),p.clone()));return old_move(t,c,p)
    bd._move=moved
    print(json.dumps({'event':'fixed_reaction_start','step':state['completed_steps']}),flush=True)
    try:event=bd.maybe_apply(trainer,trainer.last_output_sigma)
    finally:module.FeatureCellSnapshot.ordinary_transport=method;bd._move=old_move
    detail=capture['detail'];birth=detail['novel_birth_plan'];children=birth['children'];seeds=birth['source_seed_rows']
    empty=torch.empty(0,dtype=torch.long)
    cc=torch.cat([c for c,p in copies]) if copies else empty
    pp=torch.cat([p for c,p in copies]) if copies else empty
    all_children=torch.cat((cc,children));all_sources=torch.cat((pp,seeds))
    checks=dict(ordinary_budget_honest=event['ordinary_moves']<=event['ordinary_budget']==51,
        phases_sum=event['ordinary_moves']==sum(event[k] for k in ('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_novel_birth_moves')),
        all_children_unique=len(torch.unique(all_children))==len(all_children),
        all_sources_unique=len(torch.unique(all_sources))==len(all_sources),
        no_source_deleted=not bool(torch.isin(all_children,all_sources).any()),
        all_moved=torch.equal(bd.moved_rows.sort().values,all_children.sort().values),
        G_D_EMA_weights_buffers_unchanged=weights_before==[tensor_state_hash(m.state_dict()) for m in models],
        gradients_unchanged=gradients_before==[tensor_state_hash([p.grad for p in m.parameters()]) for m in models]+[tensor_state_hash(trainer.prior.z.grad)],
        module_modes_restored=all(m.training==value for m,value in modes),
        checkpoint_tensors_unchanged=original==tensor_state_hash(state))
    if hasattr(bd,'paired_average'):
        checks['postaction_FAST_cache_current']=torch.equal(bd.snapshot.query_cell_ids,bd.snapshot._assign_metric(bd.snapshot.row_features)[0])
        # Recompute directly, with the already refreshed FAST cache, to confirm
        # the recorded gate sees actual live/EMA post-action coordinates.
        before_stream=bd.stream.get_state().clone();stamp=dict(bd.paired_average)
        bd._record_paired_average(trainer,bd.snapshot)
        checks['postaction_stamp_repeats_exactly']=stamp==bd.paired_average
        checks['gate_adds_no_reaction_rng']=torch.equal(before_stream,bd.stream.get_state())
        fresh=deepcopy(bd.state_dict());_,loaded=fixture(state,module)
        loaded.load_state_dict(fresh)
        checks['backend7_load_preserves_stamp_without_chart']=loaded.snapshot is None and loaded.paired_average==bd.paired_average
        checks['backend7_roundtrip']=tensor_state_hash(fresh)==tensor_state_hash(loaded.state_dict())
        before_bad=tensor_state_hash(loaded.state_dict());old_rejected=False
        try:loaded.load_state_dict(state['birth_death'])
        except ValueError:old_rejected=True
        checks['backend6_atomic_rejection']=old_rejected and before_bad==tensor_state_hash(loaded.state_dict())
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    event_semantic={k:v for k,v in event.items() if k not in ('eval_seconds','work','paired_average','paired_average_forward_rows')}
    numeric=dict(prior=trainer.prior.z.detach(),ema_prior=trainer.ema_prior.z.detach(),
        optimizer=trainer.opt_g.state[trainer.prior.z],history=trainer.opt_g.latent_history,
        lineage=bd.lineage.neighbors,stream=bd.stream.get_state(),
        counters={k:v for k,v in bd.counters.items() if k not in ('feature_distance_cells','projection_products')},
        row_state={k:getattr(bd,k) for k in ('S','W','n','pending','anchor','radius')},
        model_state=[m.state_dict() for m in models],gradients=[[p.grad for p in m.parameters()] for m in models],
        table_gradient=trainer.prior.z.grad,completed_steps=trainer.completed_steps,
        FAST_metric_cache=bd.snapshot.row_features,FAST_cell_ids=bd.snapshot.query_cell_ids)
    record=dict(step=state['completed_steps'],checks=checks,plan_sha256=tensor_state_hash(detail),
        numerical_state_sha256=tensor_state_hash(numeric),original_semantic_event_sha256=tensor_state_hash(event_semantic),
        ordinary=event['ordinary_moves'],copies=event['ordinary_copy_moves'],novel=event['ordinary_novel_birth_moves'],
        isolation=event['iso_moves'],paired_average=getattr(bd,'paired_average',None))
    records.append(record)
    print(json.dumps({'event':'fixed_reaction_pass',**record}),flush=True)
assert sources=={str(p):sha(p) for p in source_paths}
assert inputs=={str(p):sha(p) for p in paths}
assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
receipt=dict(status='PASS',records=records,source_sha256=sources,input_sha256=inputs,
    all_sources_inputs_checkpoint_tensors_and_global_rng_unchanged=True,cpu_only=True,cuda_initialized=False,
    new_optimizer_steps=0,new_training_steps=0,new_emissions=0,new_seed_experiments=0,quality_verdict=None,
    compared_exclusions=['new paired-average metadata','eval_seconds','work','added geometry distance/projection diagnostic counters'],
    fixture_scope='existing RA7 saved weights/FIFO/CPU RNG; forced ready reaction, no historical GPU replay')
args.output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'event':'reaction_contract_complete','status':'PASS','output':str(args.output)}),flush=True)

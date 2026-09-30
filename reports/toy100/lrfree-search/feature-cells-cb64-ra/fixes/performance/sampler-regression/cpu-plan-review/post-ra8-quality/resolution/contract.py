"""Focused finite-fit edges and exact saved-input reactions; CPU only."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import traceback
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)

AREA=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
BASE=ROOT/'pkg-CB64-RA8'
PACKAGE=AREA/'pkg-RESOLUTION'
OLD_UTIL=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra7-quality/paired-average'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    for name in ('SOURCE-FROZEN.json','TESTS-FROZEN.json'):
        freeze=json.loads((AREA/name).read_text())
        for path,expected in freeze['source_and_input_sha256'].items():assert sha(path)==expected,path


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def reaction(state,module,requested,fixture,state_hash):
    fixture_state=deepcopy(state)
    fixture_state['recipe']['birth_death_cells']=requested
    trainer,bd=fixture(fixture_state,module)
    roots=(trainer.G,trainer.D,trainer.ema_G)
    for root in roots:
        root.register_buffer('resolution_contract_marker',torch.ones(3))
        for parameter in root.parameters():parameter.grad=torch.ones_like(parameter)
    trainer.prior.z.grad=torch.ones_like(trainer.prior.z)
    trainer.ema_G.train();trainer.ema_G[1].eval()
    before_models=state_hash([root.state_dict() for root in roots])
    before_grads=state_hash([[parameter.grad for parameter in root.parameters()] for root in roots]
                           +[trainer.prior.z.grad])
    modes=[(child,child.training) for root in roots for child in root.modules()]
    capture={};copies=[]
    original=module.FeatureCellSnapshot.ordinary_transport
    def traced(snapshot,q,flags,law,**kwargs):
        result=original(snapshot,q,flags,law,**kwargs)
        capture.update(plan=deepcopy(result[2]),law=deepcopy(law),flags=flags.clone())
        return result
    module.FeatureCellSnapshot.ordinary_transport=traced
    move=bd._move
    def traced_move(t,children,parents):
        copies.append((children.clone(),parents.clone()))
        return move(t,children,parents)
    bd._move=traced_move
    try:event=bd.maybe_apply(trainer,trainer.last_output_sigma)
    finally:module.FeatureCellSnapshot.ordinary_transport=original;bd._move=move
    assert event is not None and bd.snapshot.cells==64 and bd.snapshot.rank==8
    plan=capture['plan'];birth=plan['novel_birth_plan']
    empty=torch.empty(0,dtype=torch.long)
    cc=torch.cat([children for children,parents in copies]) if copies else empty
    pp=torch.cat([parents for children,parents in copies]) if copies else empty
    children=torch.cat((cc,birth['children']));sources=torch.cat((pp,birth['source_seed_rows']))
    checks=dict(all_children_unique=len(torch.unique(children))==len(children),
        all_parents_and_seeds_unique=len(torch.unique(sources))==len(sources),
        no_source_deleted=not bool(torch.isin(children,sources).any()),
        all_moved_including_novel=torch.equal(bd.moved_rows.sort().values,children.sort().values),
        ordinary_budget=event['ordinary_moves']<=event['ordinary_budget']==51,
        phases_sum=event['ordinary_moves']==sum(event[key] for key in
            ('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_novel_birth_moves')),
        actual_family=event['count_categories']==128 and event['count_multiplicity']==194
            and event['count_cutoff']==.05/194,
        G_D_EMA_weights_buffers_unchanged=state_hash([root.state_dict() for root in roots])==before_models,
        gradients_unchanged=state_hash([[parameter.grad for parameter in root.parameters()] for root in roots]
                                     +[trainer.prior.z.grad])==before_grads,
        modes_unchanged=all(child.training==value for child,value in modes),
        snapshot_requested_metadata=bd.snapshot.requested_cells==requested)
    assert all(checks.values()),[name for name,value in checks.items() if not value]
    backend=deepcopy(bd.state_dict());bd.check_state(backend)
    _,fresh=fixture(fixture_state,module);fresh.load_state_dict(backend)
    assert state_hash(fresh.state_dict())==state_hash(backend)
    assert fresh.snapshot is None and fresh._heads is None and not fresh.latent_geometry._entries
    geometry={name:getattr(bd.snapshot,name) for name in
        ('mean','scale','basis','centers','cell_scale','count_boundary','reference_counts',
         'real_calibration_counts','real_calibration_category_counts','row_features','query_cell_ids')}
    numeric=dict(prior=trainer.prior.z.detach().clone(),ema_prior=trainer.ema_prior.z.detach().clone(),
        moments=deepcopy(trainer.opt_g.state[trainer.prior.z]),history=trainer.opt_g.latent_history.clone(),
        backend=backend,geometry=geometry,models=[deepcopy(root.state_dict()) for root in roots],
        gradients=[[parameter.grad.clone() for parameter in root.parameters()] for root in roots],
        table_gradient=trainer.prior.z.grad.clone(),completed_steps=trainer.completed_steps,
        moved_rows=bd.moved_rows.clone())
    numeric['backend']['last'].pop('eval_seconds',None)
    semantic_event={name:value for name,value in event.items() if name!='eval_seconds'}
    return dict(plan=plan,law=capture['law'],numeric=numeric,event=semantic_event,
                checks=checks,backend=backend,bd=bd,fixture_state=fixture_state,
                ordinary=event['ordinary_moves'],copies=event['ordinary_copy_moves'],
                novel=event['ordinary_novel_birth_moves'],isolation=event['iso_moves'])


def normalize_proposal(numeric):
    normalized=deepcopy(numeric)
    normalized['backend']['backend_schema']=7
    normalized['backend']['settings']['cells']=64
    normalized['backend']['settings'].pop('resolution_policy')
    return normalized


def edge_contract(module,state,state_hash):
    cases=[(128,512,8,64),(128,10000,8,128),(64,512,8,64),(128,3,2,1),
           (128,4,3,1),(128,4,1,4),(128,4,0,4),(401,401,1,401),(17,10000,8,17)]
    records=[]
    for requested,fitted,rank,expected in cases:
        actual=module._fit_cell_count(requested,fitted,rank)
        assert actual==expected
        records.append(dict(requested=requested,fitted_rows=fitted,effective_rank=rank,actual=actual))
    # Fixed saved real-feature prefixes, including a rank-zero duplicate
    # control. Every fit starts a separate savedCPU RNG clone, no new seed.
    original=ROOT/'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py'
    nodes=[node for node in ast.parse(original.read_text()).body
           if isinstance(node,ast.FunctionDef) and node.name=='forward']
    namespace=dict(torch=torch,F=torch.nn.functional)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(original),'exec'),namespace)
    with torch.no_grad():features=namespace['forward'](state['birth_death']['reservoir'][:9],state['models']['D'],head=True)
    fits=[]
    for label,real in [('minimum6',features[:6]),('odd9',features),('rank1',features[:6,:1]),
                       ('degenerate',features[:6].clone().zero_())]:
        generator=torch.Generator().set_state(state['cpu_rng'])
        snap=module.FeatureCellSnapshot.fit(real,generator=generator,cells=128,rank=8,chunk=256)
        expected=module._fit_cell_count(128,len(real[0::2]),snap.rank)
        assert snap.cells==expected and snap.calibration_rows==len(real[1::2])
        law=snap.cell_comparison(real)
        assert law['categories']==2*expected and law['multiplicity']==3*expected+2
        assert law['cutoff']==.05/(3*expected+2)
        if label=='degenerate':
            assert snap.rank==0 and not snap.valid_metric
            assert not any(bool(law[key]['excess'].any() or law[key]['deficit'].any())
                           for key in ('mass','support','global_support'))
            assert bool((snap.support(real)[1]==1).all())
        fits.append(dict(case=label,rows=len(real),rank=snap.rank,cells=snap.cells,
                         calibration_rows=snap.calibration_rows,multiplicity=law['multiplicity']))
    return dict(arithmetic_cases=records,fitted_cases=fits)


def metadata_contract(module,result,state_hash):
    bd=result['bd'];valid=deepcopy(result['backend']);bd.check_state(valid)
    before=state_hash(bd.state_dict());checks=[]
    def invalid(label,change):
        bad=deepcopy(valid);change(bad);rejected=False
        try:bd.load_state_dict(bad)
        except ValueError:rejected=True
        assert rejected and state_hash(bd.state_dict())==before,label
        checks.append(label)
    invalid('old_backend7',lambda state:state.update(backend_schema=7))
    invalid('wrong_resolution_policy',lambda state:state['settings'].update(resolution_policy='old'))
    invalid('requested_not_actual_stamp',lambda state:state['paired_average'].update(cells=128))
    invalid('actual_last_cells',lambda state:state['last'].update(cells=128))
    invalid('category_count',lambda state:state['last'].update(count_categories=256))
    invalid('multiplicity',lambda state:state['last'].update(count_multiplicity=386))
    invalid('cutoff',lambda state:state['last'].update(count_cutoff=.05/386))
    for key in ('fitted_rows','ordinal','categories'):
        invalid('partition_'+key+'_float',lambda state,key=key:state['last']['count_partition'].update(
            {key:float(state['last']['count_partition'][key])}))
    invalid('partition_q_string',lambda state:state['last']['count_partition'].update(q='.05'))
    invalid('partition_extra_field',lambda state:state['last']['count_partition'].update(extra=0))
    invalid('partition_missing_field',lambda state:state['last']['count_partition'].pop('categories'))
    # Pure typed metadata, not fabricated numerical serving evidence. N801,
    # rank1/request401 exercises the ceil-even boundary401 rather than400.
    n=801;fitted=401;rank=1;k=401
    stamp=deepcopy(valid['paired_average'])
    stamp.update(rows=n,required=n-math.floor(.05*n),cells=k,rank=rank,groups=1,
                 calibration_rows=400,finite_rows=n,same_group_rows=0,ema_eligible_rows=0,
                 coherent_rows=0,chart_valid=True,duplicate_ok=True,eligible=False)
    last=dict(step=stamp['step'],snapshot=stamp['snapshot'],paired_average=stamp,
        cells=k,metric_rank=rank,calibration_rows=400,duplicate_fraction=0.,mass_topology=dict(groups=1),
        count_categories=2*k,count_multiplicity=3*k+2,count_cutoff=.05/(3*k+2),
        count_partition=dict(rule='even_fit_score_order_statistic',q=.05,fitted_rows=fitted,
                             ordinal=381,ties='inside',categories=2*k))
    checker=SimpleNamespace(N=n,settings={**bd.settings,'cells':401},paired_average=bd.paired_average)
    module.FeatureCellBirthDeath._check_paired_average_state(checker,
        dict(paired_average=stamp,snapshot_serial=stamp['snapshot'],last=last))
    return dict(atomic_invalid_controls=checks,odd_ceil_typed_control=dict(N=n,fitted_rows=fitted,
                requested=401,rank=rank,actual_cells=k,eligible=False),fresh_roundtrip=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);assert not out.exists()
    verify();sys.path.insert(0,str(BASE));sys.path.insert(0,str(OLD_UTIL))
    sys.path.insert(0,str(ROOT/'integration/review/training-regression/post-ra4-quality'))
    from particlegan import feature_cells as baseline
    from contract_utils import fixture
    from measure_saved_utils import tensor_state_hash as state_hash
    proposal=module_at('particlegan.resolution_feature_cells',PACKAGE/'particlegan/feature_cells.py')
    global_rng=torch.get_rng_state().clone();records=[];edges=metadata=None
    for step in (1250,2000):
        path=ROOT/f'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-{step:04d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)['trainer'];before=state_hash(state)
        print(json.dumps(dict(event='matched_reaction_start',step=step)),flush=True)
        a=reaction(state,baseline,64,fixture,state_hash)
        b=reaction(state,proposal,128,fixture,state_hash)
        comparisons=dict(plan=state_hash(a['plan'])==state_hash(b['plan']),
            conditional_count_law=state_hash(a['law'])==state_hash(b['law']),
            semantic_event=state_hash(a['event'])==state_hash(b['event']),
            full_numerical_state=state_hash(a['numeric'])==state_hash(normalize_proposal(b['numeric'])))
        assert all(comparisons.values()),comparisons
        record=dict(step=step,comparisons=comparisons,baseline_checks=a['checks'],proposal_checks=b['checks'],
            plan_sha256=state_hash(a['plan']),law_sha256=state_hash(a['law']),
            event_sha256=state_hash(a['event']),numerical_sha256=state_hash(a['numeric']),
            ordinary=b['ordinary'],copies=b['copies'],novel=b['novel'],isolation=b['isolation'],
            actual_cells=64,requested_baseline=64,requested_proposal=128)
        records.append(record);print(json.dumps(dict(event='matched_reaction_pass',**record)),flush=True)
        if step==2000:
            edges=edge_contract(proposal,state,state_hash)
            metadata=metadata_contract(proposal,b,state_hash)
        assert state_hash(state)==before,'saved input state mutated'
    assert torch.equal(global_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    verify()
    result=dict(status='PASS',records=records,edges=edges,metadata=metadata,
        source_freeze_sha256=sha(AREA/'SOURCE-FROZEN.json'),tests_freeze_sha256=sha(AREA/'TESTS-FROZEN.json'),
        semantic_exclusions=['backend_schema7->8','settings.cells64->128','new settings.resolution_policy',
                             'birth_death.last.eval_seconds'],
        source_input_tensors_and_global_rng_unchanged=True,cpu_only=True,cuda_initialized=False,
        new_training_steps=0,new_optimizer_steps=0,new_quality_emissions=0,new_seeds=0,quality_verdict=None,
        fixture_scope='existing savedRA8 current weights/FIFO/moments/graph/clonedCPU RNG; forced reaction, no historicalGPU replay',
        finished_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='resolution_contract_complete',status='PASS',output=str(out))),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise

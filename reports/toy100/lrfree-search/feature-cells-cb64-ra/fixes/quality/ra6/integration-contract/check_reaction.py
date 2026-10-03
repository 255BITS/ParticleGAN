"""CPU-only composed RA5 reaction checks using unchanged saved RA4 states.

This triggers one reaction on each existing FIFO, without new real inputs,
training/optimizer steps, evaluator emissions or seeds. Current-device CPU
count jitter is an internal mechanical fixture, not historical GPU replay.
"""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import ast
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import traceback
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant',choices=('RA5','RA6'),required=True)
args=parser.parse_args()
PACKAGE=ROOT/('pkg-CB64-'+args.variant);OWNER=ROOT/'integration/review/training-regression/post-ra4-quality'
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(OWNER))
from particlegan import feature_cells as module
from particlegan import birth_phase as birth
from measure_saved_utils import tensor_state_hash
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def linear(weight,bias):
    layer=torch.nn.Linear.__new__(torch.nn.Linear);torch.nn.Module.__init__(layer)
    layer.in_features=weight.shape[1];layer.out_features=weight.shape[0]
    layer.weight=torch.nn.Parameter(weight.clone());layer.bias=torch.nn.Parameter(bias.clone())
    return layer


def network(weights):
    layers=[]
    for index in (0,2,4):
        layers.append(linear(weights[f'{index}.weight'],weights[f'{index}.bias']))
        if index!=4:layers.append(torch.nn.LeakyReLU(.2))
    return torch.nn.Sequential(*layers)


def fixture(state):
    models=state['models'];z=torch.nn.Parameter(models['prior']['z'].clone())
    ez=torch.nn.Parameter(models['ema_prior']['z'].clone(),requires_grad=False)
    optimizer=state['optimizers'][0]
    table=next(v for v in optimizer['state'].values() if any(isinstance(x,torch.Tensor) and x.shape==z.shape for x in v.values()))
    opt=SimpleNamespace(state={z:deepcopy(table)},latent_history=optimizer['regularizer']['latent']['history'].clone(),
        param_groups=[dict(params=[z])])
    trainer=SimpleNamespace(G=network(models['G']),D=network(models['D']),ema_G=network(models['ema_G']),
        prior=SimpleNamespace(z=z),ema_prior=SimpleNamespace(z=ez),opt_g=opt,device=torch.device('cpu'),dtype=z.dtype,
        recipe=SimpleNamespace(**state['recipe']),completed_steps=state['completed_steps'],
        controller=SimpleNamespace(latent_bandwidth=state['controller']['latent_bandwidth'].clone(),latent_applications=[]),
        last_output_sigma=float(state['output_noise']['log_sigma'].exp()))
    bd=module.FeatureCellBirthDeath(trainer,314159)
    # Reuse an existing CPU stream state; GPU RNG bytes are never loaded into
    # a CPU stream. Readiness is forced solely to exercise this fixed reaction.
    bd.stream.set_state(state['cpu_rng'])
    for key in bd._TENSORS:setattr(bd,key,deepcopy(state['birth_death'][key]))
    bd.fill=len(z);bd.cursor=state['birth_death']['cursor'];bd.rows_since_eval=len(z)
    bd.sample_shape=tuple(state['birth_death']['sample_shape'])
    bd.snapshot_serial=state['birth_death']['snapshot_serial']
    bd.lineage.neighbors=state['birth_death']['lineage_neighbors'].clone()
    bd.counters.update({k:v for k,v in state['birth_death']['counters'].items() if k in bd.counters})
    trainer.birth_death=bd
    return trainer,bd


def reset_hook_contract(trainer,bd,event):
    """Execute the unchanged postreaction trainer hook; no training method."""
    source=ast.parse((PACKAGE/'particlegan/training.py').read_text())
    owner=next(c for c in source.body if isinstance(c,ast.ClassDef) and c.name=='GANTrainer')
    step=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_step')
    node=next(n for n in ast.walk(step) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='self.birth_death is not None'
        and any(isinstance(a,ast.Attribute) and a.attr=='maybe_apply' for a in ast.walk(n)))
    function=ast.FunctionDef(name='existing_postreaction_hook',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=[deepcopy(node)],decorator_list=[])
    namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'existing_reset_hook','exec'),namespace)
    rebase=[];reset=[]
    tester=SimpleNamespace(rebase=lambda params,rows:rebase.append(rows.clone()))
    trainer.lr_settle=SimpleNamespace(testers=[[tester]]);trainer.roles=[['prior']]
    trainer.row_evidence=SimpleNamespace(reset=lambda rows:reset.append(rows.clone()))
    original=bd.maybe_apply;bd.maybe_apply=lambda *args:event
    try:namespace['existing_postreaction_hook'](trainer)
    finally:bd.maybe_apply=original
    return dict(prior_tester_receives_all_moved=len(rebase)==1 and torch.equal(rebase[0],bd.moved_rows),
        row_evidence_receives_all_moved=len(reset)==1 and torch.equal(reset[0],bd.moved_rows),
        novel_rows_revoked=bool(torch.isin(torch.as_tensor(event['novel_birth_children'],dtype=torch.long),rebase[0]).all())
            and bool(torch.isin(torch.as_tensor(event['novel_birth_children'],dtype=torch.long),reset[0]).all()))


composition=ROOT/('quality/'+args.variant.lower()+'/COMPOSITION.json')
expected=json.loads(composition.read_text())['source_sha256']
before={str(p.relative_to(PACKAGE/'particlegan')):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
assert before==expected and sha(PACKAGE/'particlegan/birth_phase.py')==sha(OWNER/'birth_phase.py')
paths=[ROOT/f'validation-ra4/learned/training/toy/CB64-RA4/checkpoint-{s:04d}.pt' for s in (1250,2000)]
inputs={str(p):sha(p) for p in paths+[composition,OWNER/'READY.json']}
global_rng=torch.get_rng_state().clone();records=[];error=None
ordinary_method=module.FeatureCellSnapshot.ordinary_transport
try:
    for path in paths:
        state=torch.load(path,map_location='cpu',weights_only=False)['trainer'];state_hash=tensor_state_hash(state)
        trainer,bd=fixture(state);counter_before=dict(bd.counters)
        prior_before=trainer.prior.z.detach().clone();ema_before=trainer.ema_prior.z.detach().clone()
        model_before={k:tensor_state_hash(v.state_dict()) for k,v in [('G',trainer.G),('D',trainer.D),('ema_G',trainer.ema_G)]}
        parameter_modes=[(m,m.training) for root in (trainer.G,trainer.D,trainer.ema_G) for m in root.modules()]
        step_before=deepcopy(trainer.opt_g.state[trainer.prior.z]['step']);capture={};copies=[]
        def traced_transport(snapshot,q,flags,law,**kwargs):
            result=ordinary_method(snapshot,q,flags,law,**kwargs)
            capture.update(q=q.clone(),flags=flags.clone(),law=deepcopy(law),detail=deepcopy(result[2]))
            return result
        module.FeatureCellSnapshot.ordinary_transport=traced_transport
        original_move=bd._move
        def traced_move(t,c,p):
            copies.append((c.clone(),p.clone()));return original_move(t,c,p)
        bd._move=traced_move
        print(json.dumps(dict(event='reaction_start',step=state['completed_steps'])),flush=True)
        event=bd.maybe_apply(trainer,trainer.last_output_sigma)
        module.FeatureCellSnapshot.ordinary_transport=ordinary_method;bd._move=original_move
        detail=capture['detail'];plan=detail['novel_birth_plan'];newborn=plan['children'];seed=plan['source_seed_rows']
        empty=torch.empty(0,dtype=torch.long)
        all_copy_child=torch.cat([c for c,p in copies]) if copies else empty
        all_copy_parent=torch.cat([p for c,p in copies]) if copies else empty
        all_children=torch.cat((all_copy_child,newborn));all_sources=torch.cat((all_copy_parent,seed))
        unchanged=torch.ones(len(prior_before),dtype=torch.bool);unchanged[all_children]=False
        delta={k:bd.counters[k]-counter_before[k] for k in counter_before}
        cats=detail['query_category_ids'];ids=detail['query_cell_ids'];flags=capture['flags']
        ordinary_copy_children=detail['action_children'][:detail['copy_moves']]
        expected_supported=torch.bincount(ids[~flags],minlength=bd.snapshot.cells)
        expected_supported-=torch.bincount(ids[ordinary_copy_children[~flags[ordinary_copy_children]]],minlength=bd.snapshot.cells)
        expected_supported+=torch.bincount((detail['action_destination_category_ids']//2),minlength=bd.snapshot.cells)
        global_child=detail['global_children'];global_parent=detail['global_parents']
        earlier_child=torch.cat((detail['mass_children'],detail['support_children']))
        earlier_parent=torch.cat((detail['mass_parents'],detail['support_parents']))
        gross_death=(cats[torch.cat((earlier_child,newborn))].remainder(2)==1).sum()
        gross_birth=(cats[earlier_parent].remainder(2)==0).sum()+len(newborn)
        certificate=plan['certificates_after']
        with torch.no_grad():
            lf=birth.learned_latent_features(bd,trainer,trainer.G)(trainer.prior.z[newborn]) if len(newborn) else None
            ef=birth.learned_latent_features(bd,trainer,trainer.ema_G)(trainer.ema_prior.z[newborn]) if len(newborn) else None
        accepted=(not len(newborn) or bool((bd.snapshot.support(lf)[1]>.05).all()
            and (bd.snapshot.support(ef)[1]>.05).all()
            and (bd.snapshot.count_categories(lf)==plan['destination_category_ids']).all()
            and (bd.snapshot.count_categories(ef)==plan['destination_category_ids']).all()))
        checks=dict(reaction_ready_and_returned=event is not None,
            ordinary_shared_budget=event['ordinary_moves']<=event['ordinary_budget']==51,
            all_action_sum=event['moves']==event['ordinary_copy_moves']+event['ordinary_novel_birth_moves']+event['iso_moves'],
            ordinary_phase_sum=event['ordinary_moves']==event['ordinary_mass_moves']+event['ordinary_support_moves']+event['ordinary_global_moves']+len(newborn),
            unique_all_children=len(torch.unique(all_children))==len(all_children),
            unique_copy_parents_and_source_seeds=len(torch.unique(all_sources))==len(all_sources),
            no_parent_or_seed_deleted=not bool(torch.isin(all_sources,all_children).any()),
            all_moved_including_novel=torch.equal(bd.moved_rows.sort().values,all_children.sort().values),
            source_live_untouched=torch.equal(trainer.prior.z[seed],prior_before[seed]),
            source_ema_untouched=torch.equal(trainer.ema_prior.z[seed],ema_before[seed]),
            untouched_live_rows=torch.equal(trainer.prior.z[unchanged],prior_before[unchanged]),
            untouched_ema_rows=torch.equal(trainer.ema_prior.z[unchanged],ema_before[unchanged]),
            novel_live_exact=not len(newborn) or torch.equal(trainer.prior.z[newborn],plan['new_latents']),
            novel_ema_exact=not len(newborn) or torch.equal(trainer.ema_prior.z[newborn],plan['paired_ema_latents']),
            newborn_actual_original_support_and_target=accepted,
            newborn_moments_zero=all(not bool(v[newborn].any()) for v in trainer.opt_g.state[trainer.prior.z].values() if isinstance(v,torch.Tensor) and v.shape==prior_before.shape),
            newborn_history_zero=not bool(trainer.opt_g.latent_history[newborn].any()),
            scalar_optimizer_step_unchanged=torch.equal(trainer.opt_g.state[trainer.prior.z]['step'],step_before),
            newborn_lineage_no_seed_links=bool((bd.lineage.neighbors[newborn]==-1).all()),
            supported_ledger_uses_actual_destinations=torch.equal(expected_supported,detail['planned_supported_counts']),
            gross_death_certificate_true=torch.equal(certificate['spent_death'],gross_death),
            gross_birth_certificate_true=torch.equal(certificate['spent_birth'],gross_birth),
            global_remaining_certificates=len(global_child)<=min(int(certificate['residual_birth']),int(certificate['residual_death'])),
            realised_counters_include_novel=delta['realised_births']==delta['realised_deaths']==event['ordinary_moves'],
            ordinary_counters_include_novel=delta['ordinary_moves']==delta['moves']==event['ordinary_moves'],
            matched_counter_ordinary_copy_only=delta['matched']==event['ordinary_copy_moves'],
            separate_novel_counter=delta['novel_birth_moves']==len(newborn),
            iso_counter_separate=delta['iso_moves']==event['iso_moves'],
            no_semantic_timing=all('seconds' not in key for key in event if key!='eval_seconds') and 'seconds' not in json.dumps(event['novel_birth']),
            model_weights_unchanged=all(tensor_state_hash(v.state_dict())==model_before[k] for k,v in [('G',trainer.G),('D',trainer.D),('ema_G',trainer.ema_G)]),
            all_parameter_gradients_untouched=all(p.grad is None for root in (trainer.G,trainer.D,trainer.ema_G) for p in root.parameters()),
            model_modes_restored=all(m.training==mode for m,mode in parameter_modes),
            saved_checkpoint_tensors_unchanged=tensor_state_hash(state)==state_hash)
        bd.lineage.validate(bd.lineage.neighbors)
        checks.update(reset_hook_contract(trainer,bd,event))
        fresh=deepcopy(bd.state_dict());fresh_hash=tensor_state_hash(fresh)
        _,loaded=fixture(state);loaded.load_state_dict(fresh)
        checks['fresh_backend_checkpoint_roundtrip']=tensor_state_hash(loaded.state_dict())==fresh_hash
        rejected=False;before_reject=tensor_state_hash(loaded.state_dict())
        try:loaded.load_state_dict(state['birth_death'])
        except ValueError:rejected=True
        checks.update(oldRA4_backend_rejects=rejected,old_rejection_atomic=tensor_state_hash(loaded.state_dict())==before_reject)
        json_results={}
        for label,payload in [('whole_diagnostics',bd.diagnostics()),('checkpoint_last',bd.state_dict()['last']),
            ('posthook_event',event)]:
            try:
                json.dumps(payload);json_results[label]=True
            except TypeError:json_results[label]=False
        checks['JSON_expected_for_variant']=all(json_results.values()) if args.variant=='RA6' else not any(json_results.values())
        numeric_state=dict(prior=trainer.prior.z.detach(),ema_prior=trainer.ema_prior.z.detach(),
            optimizer=trainer.opt_g.state[trainer.prior.z],history=trainer.opt_g.latent_history,
            lineage=bd.lineage.neighbors,stream=bd.stream.get_state(),counters=bd.counters,
            row_evidence=dict(S=bd.S,W=bd.W,n=bd.n,pending=bd.pending,anchor=bd.anchor,radius=bd.radius))
        row=dict(step=state['completed_steps'],ordinary_moves=event['ordinary_moves'],copy_moves=event['ordinary_copy_moves'],
            novel_births=len(newborn),isolation_moves=event['iso_moves'],all_moves=event['moves'],
            mass=event['ordinary_mass_moves'],local=event['ordinary_support_moves'],global_copy=event['ordinary_global_moves'],
            moved_rows=bd.moved_rows.clone(),copy_parent_rows=all_copy_parent,source_seed_rows=seed,
            true_global_certificates=certificate,backend_schema=bd.BACKEND_SCHEMA,checks=checks,
            JSON_serialization=json_results,numerical_state_sha256=tensor_state_hash(numeric_state),
            fixed_plan_sha256=tensor_state_hash(detail))
        records.append(plain(row));print(json.dumps(dict(event='reaction_checks',**plain(row))),flush=True)
        assert all(checks.values()),[k for k,v in checks.items() if not v]
    assert sum(r['novel_births'] for r in records)>0,'actual new birth reaction not exercised'
except Exception as failure:
    error=dict(type=type(failure).__name__,message=str(failure),traceback=traceback.format_exc())
    print(json.dumps(dict(event='exception',**error)),flush=True)
finally:module.FeatureCellSnapshot.ordinary_transport=ordinary_method
after={str(p.relative_to(PACKAGE/'particlegan')):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
inputs_after={p:sha(Path(p)) for p in inputs}
checks=dict(package_sources_unchanged=before==after,composition_sources_match=before==expected,
    input_files_unchanged=inputs==inputs_after,global_rng_unchanged=torch.equal(global_rng,torch.get_rng_state()),
    no_cuda_context=not torch.cuda.is_initialized())
result=dict(status='PASS' if error is None and all(checks.values()) else 'ERROR',error=error,records=records,checks=checks,
    source_sha256_before=before,source_sha256_after=after,source_sha256={str(PACKAGE/'particlegan'/name):digest for name,digest in before.items()},
    input_sha256=inputs,local_source_sha256={str(Path(__file__)):sha(Path(__file__)),str(OWNER/'measure_saved_utils.py'):sha(OWNER/'measure_saved_utils.py')},
    cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),new_training_steps=0,new_optimizer_steps=0,new_seeds=0,
    quality_verdict=None,checkpoint_scope='new/old feature-cell backend state; full trainer population checkpoint reviewed separately',
    fixture_scope='existing saved critic/G/current+EMA latent/FIFO; one forced-ready CPU reaction; internal current-device count jitter only',
    input_stream_scope='existing saved CPU RNG bytes; never old GPU stream bytes',
    inherited_postreaction_hook='existing production trainer AST, same all-moved vector to prior rebase and row-evidence reset')
(HERE/(args.variant.lower()+'-reaction.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(event='integration_reaction_complete',status=result['status'],error=error)),flush=True)
raise SystemExit(0 if result['status']=='PASS' else 1)

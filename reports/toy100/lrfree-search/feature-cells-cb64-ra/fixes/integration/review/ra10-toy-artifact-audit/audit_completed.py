"""Read-only CPU audit of all ten completed RA10 toy checkpoints; no forwards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import argparse
import ast
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import traceback
from types import SimpleNamespace
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
ORIGINAL=ROOT/'integration/review/audit_learned.py'
STEPS=(0,100,250,500,750,1000,1250,1500,1750,2000)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify(seal):
    for p,d in seal['source_and_input_sha256'].items():assert sha(p)==d,p

def functions(path,names):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    return nodes

def time_fields(value,prefix='trainer'):
    fields=[]
    if isinstance(value,dict):
        for key,item in value.items():
            name=prefix+'.'+str(key)
            if any(word in str(key).lower() for word in ('seconds','timing','elapsed','perf_counter')):fields.append(name)
            fields.extend(time_fields(item,name))
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value):fields.extend(time_fields(item,prefix+f'[{i}]'))
    return fields

def json_equal(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=True)==json.dumps(b,sort_keys=True,allow_nan=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    seal=read(args.input_freeze);verify(seal)  # Before Torch or PT interpretation.
    assert seal['status']=='COMPLETED_TOY_INPUTS_FROZEN' and seal['steps']==list(STEPS)
    lane=Path(seal['validation']);package=Path(seal['package_root']);run=lane/'learned/training/toy/CB64-RA10'
    assert args.output.resolve().is_relative_to(ROOT/'integration/review')
    args.output.mkdir(parents=True)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rng=torch.get_rng_state().clone()
    ns=dict(torch=torch,argparse=argparse,datetime=datetime,timezone=timezone,hashlib=hashlib,json=json,math=math,
        os=os,Path=Path,struct=struct,sys=sys,traceback=traceback,
        REVIEW=ROOT/'integration/review',STUDY=ROOT,
        args=SimpleNamespace(validation=lane,variant='CB64-RA10',output=args.output),
        LEARNED=lane/'learned',CHECKPOINTS=STEPS,captured_freeze=None,
        EXPECTED_GPU='GPU-72c1b506-891d-b8bc-b353-e020585e1c47')
    names=('read','sha','write','require','verify_sources','load_cpu','digest','semantic','metadata','rng_placement','verify_runtime','audit_training')
    nodes=functions(ORIGINAL,names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ORIGINAL),'exec'),ns)
    original_load=ns['load_cpu'];loaded={}
    def captured_load(path):
        result=original_load(path);loaded[str(Path(path))]=result
        return result  # Exact original authority return, no mutation or changed device tags.
    ns['load_cpu']=captured_load
    integrity=ns['verify_sources']()
    authority=ns['audit_training']('toy',read(args.input_freeze.parent/'toy-job-event.json'))
    assert authority['evidence_status']=='VALID' and authority['primary_status']=='COMPLETE'
    (args.output/'original-authority.json').write_text(json.dumps(authority,indent=2)+'\n')
    feature=ast.parse((package/'particlegan/feature_cells.py').read_text())
    cls=next(n for n in feature.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellBirthDeath')
    method_names={'_check_paired_average_state','check_paired_average_step','paired_average_eligible','_check_mean_actions'}
    definitions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in method_names]
    assert {n.name for n in definitions}==method_names
    resolution=[n for n in feature.body if (isinstance(n,ast.FunctionDef) and n.name=='_fit_cell_count')
        or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CELL_RESOLUTION_POLICY' for t in n.targets))]
    methods=dict(torch=torch,math=math,Fraction=Fraction,Q=.05)
    exec(compile(ast.Module(body=resolution+definitions,type_ignores=[]),'<frozen-backend9-semantic-checks>','exec'),methods)
    mean_tree=ast.parse((package/'particlegan/mean_transport.py').read_text())
    assigns=[n for n in mean_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
        {'Q','POLICY','MEAN_POLICY','PLANNING_POLICY','INSIDE_POLICY','SCALAR_FIELDS','MEAN_KEYS'} for t in n.targets)]
    mean=dict(math=math)
    exec(compile(ast.Module(body=assigns+functions(package/'particlegan/mean_transport.py',('initial_mean_diagnostics','check_mean_diagnostics')),type_ignores=[]),'<frozen-mean-metadata-checks>','exec'),mean)
    records=[]
    curves=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
    for step in STEPS:
        saved,devices=loaded[str(run/f'checkpoint-{step:04d}.pt')]
        state=saved['trainer'];record=saved['record'];bd=state['birth_death'];last=bd['last'];settings=bd['settings']
        n=len(state['models']['prior']['z']);stamp=bd['paired_average'];serial=bd['snapshot_serial']
        assert type(state['schema']) is int and state['schema']==5
        assert type(bd['backend_schema']) is int and bd['backend_schema']==9 and bd['backend']=='feature_cells'
        assert settings['cells']==state['recipe']['birth_death_cells']==128
        assert settings['resolution_policy']==methods['CELL_RESOLUTION_POLICY']
        assert settings['mean_policy']==mean['MEAN_POLICY'] and settings['mean_planning_policy']==mean['PLANNING_POLICY']
        assert settings['mean_inside_policy']==mean['INSIDE_POLICY'] and settings['mean_rank_bound']==8 and settings['mean_pool']==64
        assert settings['count_family']=='original_K_plus_support_2K_plus_global_2_plus_mean_1_common_Q_over_3K_plus_3'
        assert settings['novel_birth_policy']=='paired_even_real_anchor_shared_3K_plus_3_v1'
        assert state['initial_lrs']==[[.0010625,.0085,.0010625],[.00425]]
        assert [[group['lr'] for group in opt['param_groups']] for opt in state['optimizers']]==record['diagnostics']['lr']
        curve=curves[list(STEPS).index(step)]
        assert all(json_equal(record[key],value) for key,value in curve.items())
        fake=SimpleNamespace(N=n,settings=settings,paired_average=stamp,snapshot_serial=serial,
            snapshot=None,fill=bd['fill'],rows_since_eval=bd['rows_since_eval'],dry_run=False)
        methods['_check_paired_average_state'](fake,bd)
        methods['check_paired_average_step'](fake,bd,step)
        methods['_check_mean_actions'](fake,bd)
        mean['check_mean_diagnostics'](last.get('mean_transport'),last=last,paired=stamp,n=n,snapshot_serial=serial,dry_run=False)
        count=bd['counters'];assert count['mean_evals']==serial==count['evals']
        assert 0<=count['mean_witness_fires']<=count['mean_evals'] and count['mean_moves']>=last['mean_transport']['moves']
        assert count['ordinary_moves']==count['moves'] and count['matched']==count['ordinary_moves']-count['novel_birth_moves']
        assert state['row_evidence']['counters']['resets']==count['moves']+count['iso_moves']
        assert state['row_evidence']['counters']['updates']==step
        assert n==1024 and state['recipe']['batch_size']==128
        assert serial==step//8 and stamp['step']==(step//8)*8 and bd['rows_since_eval']==128*(step-stamp['step'])
        assert bd['fill']==min(n,128*step)
        eligible=methods['paired_average_eligible'](fake,step)
        assert eligible==bool(stamp['eligible'] and bd['fill']==n and 0<=bd['rows_since_eval']<n)
        diagnostic=record['diagnostics']['birth_death']
        assert diagnostic['paired_average']==stamp and diagnostic['paired_average_age_real_rows']==bd['rows_since_eval']
        assert diagnostic['last']==last and diagnostic['settings']==settings
        assert not any(k in bd for k in ('snapshot','latent_geometry','moved_rows','mean_packet','fixed_moment'))
        assert time_fields(state)==(['trainer.birth_death.last.eval_seconds'] if serial else [])
        json.dumps(last,allow_nan=False)
        graph=bd['lineage_neighbors'];rows=torch.arange(n)[:,None].expand_as(graph);edges=graph>=0
        assert graph.shape==(n,8) and graph.dtype==torch.long
        assert bool(((graph>=-1)&(graph<n)).all()) and not bool(((graph==rows)&edges).any())
        packed=(rows[edges]*n+graph[edges]).sort().values;reverse=(graph[edges]*n+rows[edges]).sort().values
        assert torch.equal(packed,reverse) and len(torch.unique(packed))==len(packed)
        table=state['lr_settle'][0][1];mask=table['stationary_rows']
        assert table['population_schema']==1 and table['population_q']==.05
        assert table['population_policy']=='two_pair_participation_Q_survival_one_descent_undo_v1'
        assert mask.shape==(n,) and mask.dtype==torch.bool and table['population_active']==(table['last_decisive']==-1)
        if table['population_active']:
            assert int(mask.sum())>=n-math.floor(.05*n) and table['stationary_undo_s']==table['s']/.5
        else:assert table['stationary_undo_s'] is None
        boundary=None
        if serial and stamp['step']==step:
            children=last['ordinary_action_children'];mc=last['ordinary_mean_children'];mp=last['ordinary_mean_parents']
            group=state['optimizers'][0]['param_groups'][1];moments=state['optimizers'][0]['state'].get(group['params'][0],{})
            for value in moments.values():
                if isinstance(value,torch.Tensor) and value.shape==state['models']['prior']['z'].shape:
                    assert torch.equal(value[mc],value[mp])
            history=state['optimizers'][0]['regularizer']['latent']['history']
            assert torch.equal(history[mc],history[mp])
            for key in ('M','Qs','W','S','flag'):assert not bool(state['row_evidence'][key][children].any())
            if children:
                assert bool(table['invalid_block_rows'][children].all())
                # The inherited rebase only clears a currently active mask.
                # Inactive historical masks are not current participation.
                if table['population_active']:assert not bool(mask[children].any())
            for child,parent in zip(mc,mp):assert bool((graph[child]==parent).any())
            boundary=dict(mean_moment_history_inherited=True,own_evidence_reset=True,unfinished_participation_invalidated=True,
                current_mean_lineage_links=True,rows_checked=len(children),mean_rows_checked=len(mc),
                limitation='Isolation IDs and unsaved reaction histories are absent; cumulative reset totals checked separately.')
        item=dict(status='VALID',step=step,checkpoint_sha256=sha(run/f'checkpoint-{step:04d}.pt'),
            whole_state_gpu_typed_sha256=ns['digest'](state,devices),
            semantic_state_gpu_typed_sha256=ns['digest'](ns['semantic'](state),devices),
            backend_schema=9,trainer_schema=5,mean=dict(last['mean_transport']),mean_counters={k:v for k,v in count.items() if k.startswith('mean_')},
            paired_average=dict(stamp),lease_age_real_rows=bd['rows_since_eval'],lease_eligible_now=eligible,
            derived_served_view='EMA' if eligible else 'FAST',population_active=table['population_active'],
            last_phases={k:last.get(k) for k in ('ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves','ordinary_novel_birth_moves','ordinary_mean_moves','ordinary_moves','iso_moves','moves')},
            reaction_boundary_evidence=boundary,quality_verdict=None)
        records.append(item)
        (args.output/f'checkpoint-{step:04d}.json').write_text(json.dumps(item,indent=2)+'\n')
        print(json.dumps(dict(event='completed_checkpoint_valid',step=step,mean_status=item['mean']['status'],mean_moves=item['mean']['moves'])),flush=True)
    verify(seal);assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    result=dict(status='VALID',evidence_status='VALID',steps=list(STEPS),checkpoints=records,source_integrity=integrity,
        original_authority_sha256=sha(ORIGINAL),original_authority_source_unchanged=True,
        original_authority_quality_subgate=authority['toy_quality_gate'],final_record=authority['final'],
        source_and_input_sha256=seal['source_and_input_sha256'],CPU_only=True,cuda_initialized=False,
        model_constructions=0,model_forwards=0,training_updates=0,optimizer_updates=0,new_quality_emissions=0,new_seeds=0,
        RA9_training_parity_asserted=False,quality_verdict=None,
        scope='Completed toy saved-artifact/source validity only; root owns full strict toy gate and separate grid/replay qualification.')
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='VALID',checkpoints=len(records),quality_verdict=None)),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='INVALID',error=str(error),traceback=traceback.format_exc())),flush=True)
        raise

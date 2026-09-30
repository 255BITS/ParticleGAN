"""One CPU read of original MNIST/replay artifacts; final canonical JSON review."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import argparse
import math
import struct
import sys
import traceback
from fractions import Fraction
from types import SimpleNamespace
from common import *
sys.dont_write_bytecode=True

def time_fields(value,prefix='trainer'):
    result=[]
    if isinstance(value,dict):
        for key,item in value.items():
            name=prefix+'.'+str(key)
            if any(word in str(key).lower() for word in ('seconds','timing','elapsed','perf_counter')):result.append(name)
            result.extend(time_fields(item,name))
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value):result.extend(time_fields(item,prefix+f'[{i}]'))
    return result

def validators(torch):
    feature=ast.parse((PACKAGE/'particlegan/feature_cells.py').read_text())
    cls=next(n for n in feature.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellBirthDeath')
    names={'_check_paired_average_state','check_paired_average_step','paired_average_eligible','_check_mean_actions'}
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods}==names
    resolution=[n for n in feature.body if (isinstance(n,ast.FunctionDef) and n.name=='_fit_cell_count') or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CELL_RESOLUTION_POLICY' for t in n.targets))]
    check=dict(torch=torch,math=math,Fraction=Fraction,Q=.05)
    exec(compile(ast.Module(body=resolution+methods,type_ignores=[]),'<qualified-semantic-methods>','exec'),check)
    output_tree=ast.parse((PACKAGE/'particlegan/output_moments.py').read_text())
    output_nodes=[n for n in output_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
        {'Q','POLICY','PROJECTION_POLICY','FIT_CHUNK','MAX_RANK'} for t in n.targets)]
    output={};exec(compile(ast.Module(body=output_nodes,type_ignores=[]),'<qualified-output-constants>','exec'),output)
    mean_tree=ast.parse((PACKAGE/'particlegan/mean_transport.py').read_text())
    assignments=[n for n in mean_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
        {'Q','POLICY','MEAN_POLICY','PLANNING_POLICY','INSIDE_POLICY','SCALAR_FIELDS','MEAN_KEYS'} for t in n.targets)]
    mean=dict(math=math,OUTPUT_MEAN_POLICY=output['POLICY'],PROJECTION_POLICY=output['PROJECTION_POLICY'],MAX_RANK=output['MAX_RANK'])
    exec(compile(ast.Module(body=assignments+nodes(PACKAGE/'particlegan/mean_transport.py',('initial_mean_diagnostics','check_mean_diagnostics')),type_ignores=[]),'<qualified-mean-validator>','exec'),mean)
    return check,output,mean

def state_check(saved,torch,check,output,mean):
    state=saved['trainer'];step=state['completed_steps'];bd=state['birth_death'];settings=bd['settings'];last=bd['last']
    n=len(state['models']['prior']['z']);stamp=bd['paired_average'];serial=bd['snapshot_serial'];counts=bd['counters']
    assert type(state['schema']) is int and state['schema']==5
    assert type(bd['backend_schema']) is int and bd['backend_schema']==10 and bd['backend']=='feature_cells'
    assert settings['cells']==state['recipe']['birth_death_cells']==128
    assert settings['resolution_policy']==check['CELL_RESOLUTION_POLICY']
    assert settings['mean_policy']==mean['MEAN_POLICY'] and settings['mean_planning_policy']==mean['PLANNING_POLICY']
    assert settings['mean_inside_policy']==mean['INSIDE_POLICY'] and settings['mean_rank_bound']==8 and settings['mean_pool']==64
    assert settings['mean_projection_policy']==output['PROJECTION_POLICY'] and settings['mean_frame_fit_chunk']==output['FIT_CHUNK']
    assert settings['count_family']=='original_K_plus_support_2K_plus_global_2_plus_mean_1_common_Q_over_3K_plus_3'
    assert settings['novel_birth_policy']=='paired_even_real_anchor_shared_3K_plus_3_v1'
    assert state['initial_lrs']==[[.0010625,.0085,.0010625],[.00425]] and n==1024 and state['recipe']['batch_size']==128
    fake=SimpleNamespace(N=n,settings=settings,paired_average=stamp,snapshot_serial=serial,snapshot=None,
        fill=bd['fill'],rows_since_eval=bd['rows_since_eval'],dry_run=False)
    check['_check_paired_average_state'](fake,bd);check['check_paired_average_step'](fake,bd,step);check['_check_mean_actions'](fake,bd)
    mean['check_mean_diagnostics'](last.get('mean_transport'),last=last,paired=stamp,n=n,snapshot_serial=serial,dry_run=False,sample_shape=bd['sample_shape'])
    assert counts['mean_evals']==serial==counts['evals'] and 0<=counts['mean_witness_fires']<=counts['mean_evals']
    assert counts['mean_moves']>=last['mean_transport']['moves']
    assert counts['ordinary_moves']==counts['moves'] and counts['matched']==counts['ordinary_moves']-counts['novel_birth_moves']
    assert state['row_evidence']['counters']['resets']==counts['moves']+counts['iso_moves']
    assert state['row_evidence']['counters']['updates']==step
    assert serial==step//8 and stamp['step']==8*(step//8) and bd['rows_since_eval']==128*(step-stamp['step'])
    assert bd['fill']==min(n,128*step)
    eligible=check['paired_average_eligible'](fake,step)
    assert eligible==bool(stamp['eligible'] and bd['fill']==n and 0<=bd['rows_since_eval']<n)
    assert not any(k in bd for k in ('snapshot','latent_geometry','moved_rows','mean_packet','fixed_moment','output_projection','output_moment_frame'))
    assert time_fields(state)==(['trainer.birth_death.last.eval_seconds'] if serial else [])
    json.dumps(last,allow_nan=False)
    graph=bd['lineage_neighbors'];row_ids=torch.arange(n)[:,None].expand_as(graph);present=graph>=0
    assert graph.shape==(n,8) and graph.dtype==torch.long
    assert bool(((graph>=-1)&(graph<n)).all()) and not bool(((graph==row_ids)&present).any())
    edges=(row_ids[present]*n+graph[present]).sort().values;reverse=(graph[present]*n+row_ids[present]).sort().values
    assert torch.equal(edges,reverse) and len(torch.unique(edges))==len(edges)
    table=state['lr_settle'][0][1];mask=table['stationary_rows']
    assert table['population_schema']==1 and table['population_q']==.05 and table['population_policy']=='two_pair_participation_Q_survival_one_descent_undo_v1'
    assert mask.shape==(n,) and mask.dtype==torch.bool and table['population_active']==(table['last_decisive']==-1)
    if table['population_active']:assert int(mask.sum())>=n-math.floor(.05*n) and table['stationary_undo_s']==table['s']/.5
    else:assert table['stationary_undo_s'] is None
    boundary=None
    if serial and stamp['step']==step:
        children=last['ordinary_action_children'];mc=last['ordinary_mean_children'];mp=last['ordinary_mean_parents']
        group=state['optimizers'][0]['param_groups'][1];moments=state['optimizers'][0]['state'].get(group['params'][0],{})
        for value in moments.values():
            if isinstance(value,torch.Tensor) and value.shape==state['models']['prior']['z'].shape:assert torch.equal(value[mc],value[mp])
        history=state['optimizers'][0]['regularizer']['latent']['history'];assert torch.equal(history[mc],history[mp])
        for key in ('M','Qs','W','S','flag'):assert not bool(state['row_evidence'][key][children].any())
        if children:
            assert bool(table['invalid_block_rows'][children].all())
            if table['population_active']:assert not bool(mask[children].any())
        for child,parent in zip(mc,mp):assert bool((graph[child]==parent).any())
        boundary=dict(rows_checked=len(children),mean_rows_checked=len(mc),own_evidence_reset=True,mean_history_moments_inherited=True,unfinished_participation_invalidated=True)
    if 'record' in saved:
        diag=saved['record']['diagnostics']['birth_death']
        assert diag['last']==last and diag['settings']==settings and diag['paired_average']==stamp and diag['paired_average_age_real_rows']==bd['rows_since_eval']
        assert [[g['lr'] for g in opt['param_groups']] for opt in state['optimizers']]==saved['record']['diagnostics']['lr']
    return dict(status='VALID',step=step,trainer_schema=5,backend_schema=10,mean_schema=2,
        mean=dict(last['mean_transport']),lease_eligible_now=eligible,derived_served_view='EMA' if eligible else 'FAST',
        reaction_boundary_evidence=boundary,limitation='No unsaved or isolation row identities reconstructed; historical replay IDs are not current reset evidence.')

def canonical_check(seal):
    ready=read(HERE/'CHECKER-READY.json');summary=read(MONITOR/'summary.json');manifest=read(MONITOR/'READ-ONLY-ARTIFACT-MANIFEST.json')
    assert summary['completed']==summary['total']==16 and summary['source_integrity']['status']=='VALID'
    assert summary['source_integrity']['package_sha256']==PACKAGE_SHA and summary['source_integrity']['config_sha256']==CONFIG_SHA
    records=[]
    expected_options=dict(eval_output_noise=True,strict_streams=True,save_final_state=True,diagnostics=True,evaluation_generate='indexed',serial_backward_argument=True,initialization='batch_feature_zero',image_prior_perturb=False,ring_frozen_control=False)
    for task in ready['tasks']:
        run=LANE/'screens/runs'/task;path=MONITOR/'canonical-receipts/screens/runs'/task/'acceptance-receipt.json'
        row=read(path);raw=read(run/'result.json');execution=read(run/'execution-receipt.json')
        assert row==next(r for r in summary['records'] if r['task']==task)
        assert row['canonical_fixture_validity']=='VALID' and not row['validity_reasons']
        assert row['primary_status']==row['acceptance_status']==row['canonical_gpu_acceptance']==raw['status'] in ('PASS','FAIL')
        assert row['result_sha256']==sha(run/'result.json') and row['execution_receipt_sha256']==sha(run/'execution-receipt.json')
        assert row['source_integrity']['status']=='VALID' and row['candidate_package_sha256']==PACKAGE_SHA and row['config_sha256']==CONFIG_SHA
        assert raw['header']['options']==expected_options and raw['header']['package_sha256']==PACKAGE_SHA
        assert raw['stream_deviations']==0 and execution['process_exit_code']==0
        assert execution['source_integrity_before']['status']==execution['source_integrity_after']['status']=='VALID'
        assert execution['resources']['gpu_uuid']==GPU and execution['resources']['cuda_memory_fraction']==.2
        assert row['expected']==execution['expected'] and raw['completed_steps']==row['expected']['steps']
        curves=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        assert [r['step'] for r in curves]==row['expected']['observation_steps']
        for field in ('final','ema_final','clean_final','native','segments','thresholds','pass_rule'):assert row.get(field)==raw.get(field),task+':'+field
        records.append(dict(task=task,primary_status=row['primary_status'],evidence_status='VALID',acceptance_status=row['acceptance_status'],receipt_sha256=sha(path)))
    actual={str(p.relative_to(LANE)):sha(p) for task in ready['tasks'] for p in sorted((LANE/'screens/runs'/task).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    assert set(actual)==set(manifest['inputs'])
    assert all(actual[p]==d['sha256'] and (LANE/p).stat().st_size==d['bytes'] for p,d in manifest['inputs'].items())
    assert summary['status']==('PASS' if all(r['primary_status']=='PASS' for r in records) else 'FAIL')
    return dict(status='VALID',original_quality_status=summary['status'],records=records,manifest_sha256=sha(MONITOR/'READ-ONLY-ARTIFACT-MANIFEST.json'),
        collector_exception='Only predeclared evaluation_generate plain -> indexed; original numerical/scoring checks unchanged.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-freeze',type=Path,required=True);parser.add_argument('--root-go',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert args.output.resolve().is_relative_to(HERE) and not args.output.exists()
    seal=read(args.input_freeze);verify(seal['source_and_input_sha256'])
    assert seal['status']=='COMPLETED_ORIGINAL_RA11_INPUTS_FROZEN'
    go=read(args.root_go);assert go['input_freeze_sha256']==sha(args.input_freeze) and go['helper_sha256']==sha(Path(__file__)) and go['exactly_one_CPU_artifact_invocation'] is True
    args.output.mkdir(parents=True)
    canonical=canonical_check(seal)  # JSON only; use original canonical outcomes, no collector/PT/scorer rerun.
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);rng=torch.get_rng_state().clone()
    ns=dict(torch=torch,argparse=argparse,datetime=datetime,timezone=timezone,hashlib=hashlib,json=json,math=math,os=os,Path=Path,struct=struct,sys=sys,traceback=traceback,
        REVIEW=ROOT/'integration/review',STUDY=ROOT,args=SimpleNamespace(validation=LANE,variant=VARIANT,output=args.output),LEARNED=LANE/'learned',CHECKPOINTS=STEPS,
        LOSS_KEYS=('loss_d','loss_g','loss_gan','prior_regularization','penalty'),captured_freeze=None,EXPECTED_GPU=GPU)
    exec(compile(ast.Module(body=nodes(ORIGINAL,AUTHORITY_NAMES),type_ignores=[]),str(ORIGINAL),'exec'),ns)
    original_load=ns['load_cpu'];loaded={}
    def captured_load(path):
        key=str(Path(path));assert key not in loaded,key
        result=original_load(path);loaded[key]=result;return result
    ns['load_cpu']=captured_load
    integrity=ns['verify_sources']();events={row['name']:row for row in seal['completed_jobs']}
    training=ns['audit_training']('mnist',events[f'learned-mnist-{VARIANT}'])
    assert training['primary_status']=='COMPLETE' and training['evidence_status']=='VALID'
    aggregate=read(LANE/'learned'/f'replay-{VARIANT}.json')
    assert sha(LANE/'learned'/f'replay-{VARIANT}.json')==events[f'replay-{VARIANT}']['result_sha256']
    assert set(aggregate)=={'toy','mnist'}
    replay={problem:ns['audit_replay'](problem,aggregate[problem]) for problem in ('toy','mnist')}
    assert all(row['evidence_status']=='VALID' for row in replay.values())
    assert len(loaded)==14
    check,output,mean=validators(torch);typed=[]
    for path,(saved,devices) in loaded.items():
        item=state_check(saved,torch,check,output,mean);item['path']=path
        item['whole_state_gpu_typed_sha256']=ns['digest'](saved['trainer'],devices)
        typed.append(item)
    curves=[json.loads(line) for line in (LANE/'learned/training/mnist'/VARIANT/'metrics.jsonl').read_text().splitlines() if line.strip()]
    for curve in curves:
        saved=loaded[str(LANE/'learned/training/mnist'/VARIANT/f"checkpoint-{curve['step']:04d}.pt")][0]
        assert all(json.dumps(saved['record'][key],sort_keys=True)==json.dumps(value,sort_keys=True) for key,value in curve.items())
    verify(seal['source_and_input_sha256']);assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    value=dict(status='VALID',evidence_status='VALID',utc=now(),source_integrity=integrity,total_original_jobs=19,total_canonical_screens=16,
        original_authority_sha256=sha(ORIGINAL),MNIST=training,replay=replay,canonical=canonical,typed_saved_states=typed,
        reused_closed_proofs={str(p):sha(p) for p in PROOF_SEALS},initial_quality_receipt_sha256=sha(ROOT/'quality/results/CB64-RA11.json'),
        MNIST_quality_gate=None,MNIST_quality_outcome='Reported regression retained; artifact VALID is not image-quality PASS.',
        planned_and_actual_PT_loads=14,model_constructions=0,model_forwards=0,training_updates=0,optimizer_updates=0,new_quality_emissions=0,new_seeds=0,cuda_initialized=False,
        root_GO_sha256=sha(args.root_go),input_freeze_sha256=sha(args.input_freeze),source_and_input_sha256=seal['source_and_input_sha256'])
    write_new(args.output/'receipt.json',value)
    print(json.dumps(dict(status='VALID',MNIST_primary=training['primary_status'],replay={k:v['primary_status'] for k,v in replay.items()},canonical_quality=canonical['original_quality_status'],PT_loads=len(loaded))),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='INVALID',error=str(error),traceback=traceback.format_exc())),flush=True)
        raise

"""Draw-free source-bound diagnostics of existing C6/H2 cloud evidence.

No public model/fixture construction, restore, draw, official scorer, training,
or CUDA use. Recorded grades are copied. CDF arithmetic locates discrepancy
in already retained arrays and does not establish a new scientific verdict.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
RAW=Path('/ml2/hypergan/forge-continuous-leaderboard-20261003/hold-continuation-v2')
PREVIOUS=Path('/ml2/hypergan/ParticleGAN-continuous-pg-leaderboard-20261003/reports/forge/continuous-baseline-20261003/hold-results.json')
SOURCE='8021a1c50c4aff90ddea5010d368cffdc857b2f6'
STUDY_PINS={'atlas':'55701698dc8cc0c933b9b2aa0e2f0bd4fb00df86124977b35711fc66e68b5859', 'e22':'7a0ee36f9550d970e3c3a9a337a798405b70891dfcdae3bc135296920ce8d336'}
inputs={}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path,pin=None):
    path=Path(path); h=sha(path)
    if pin is not None: assert h==pin
    inputs[str(path)]=h
    return dict(path=str(path),sha256=h,bytes=path.stat().st_size)


def safe(v):
    if isinstance(v,float) and not math.isfinite(v): return str(v)
    if isinstance(v,torch.Tensor): return v.detach().cpu().tolist()
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    return v


def compare(a,b):
    changed=[]; tensors=0; maximum=0.
    def walk(x,y,path):
        nonlocal tensors,maximum
        if type(x) is not type(y): changed.append(dict(path=path,kind='type')); return
        if isinstance(x,torch.Tensor):
            tensors+=1
            if x.dtype!=y.dtype or x.shape!=y.shape: changed.append(dict(path=path,kind='shape_or_dtype')); return
            same=x==y
            if x.is_floating_point(): same |= torch.isnan(x)&torch.isnan(y)
            if not bool(same.all()):
                mask=torch.isfinite(x)&torch.isfinite(y)
                delta=float((x[mask].double()-y[mask].double()).abs().max()) if bool(mask.any()) else 0.
                maximum=max(maximum,delta)
                changed.append(dict(path=path,kind='tensor',unequal_elements=int((~same).sum()),max_abs_finite_difference=delta))
        elif isinstance(x,dict):
            for k in sorted(set(x)|set(y),key=str):
                if k not in x or k not in y: changed.append(dict(path=path+'/'+str(k),kind='missing'))
                else: walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y): changed.append(dict(path=path,kind='length'))
            for i,(xx,yy) in enumerate(zip(x,y)): walk(xx,yy,path+'/'+str(i))
        elif x!=y and not (isinstance(x,float) and math.isnan(x) and math.isnan(y)):
            changed.append(dict(path=path,kind='scalar',left=safe(x),right=safe(y)))
    walk(a,b,'')
    return dict(equal_with_paired_missing_diagnostic_sentinels=not changed,tensor_leaves=tensors,differing_leaves=len(changed),max_abs_finite_tensor_difference=maximum,changes=changed)


def normal_cdf(z):
    flat=np.asarray(z,dtype=np.float64)
    return np.fromiter((.5*(1+math.erf(float(x)/math.sqrt(2))) for x in flat.flat),dtype=np.float64,count=flat.size).reshape(flat.shape)


def finite_tensor_summary(value):
    counts={'floating_tensors':0,'floating_values':0,'nonfinite_values':0}
    def walk(v):
        if isinstance(v,torch.Tensor) and v.is_floating_point():
            counts['floating_tensors']+=1;counts['floating_values']+=v.numel()
            counts['nonfinite_values']+=int((~torch.isfinite(v)).sum())
        elif isinstance(v,dict):
            for x in v.values():walk(x)
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
    walk(value)
    return counts


def shape_description(x):
    """Analytic two-Gaussian CDF witness; no call to frozen scoring code."""
    x=x.astype(np.float64); means=np.array([[-1.,0.],[1.,0.]])
    label=((x[:,None]-means[None])**2).sum(2).argmin(1)
    components=[]
    for k in range(2):
        block=x[label==k]; mu=block.mean(0); residual=block-mu
        cov=residual.T@residual/len(block)  # descriptive population (ddof=0)
        components.append(dict(nearest_center=k,count=len(block),mass=len(block)/len(x),mean=mu.tolist(),mean_offset=(mu-means[k]).tolist(),centered_covariance=cov.tolist(),eigen_ratios_to_target=np.linalg.eigvalsh(cov/.0625).tolist()))
    witnesses=[]
    n=len(x)
    for j in range(32):
        theta=j*np.pi/32; direction=np.array([np.cos(theta),np.sin(theta)])
        projection=x@direction; order=np.argsort(projection,kind='stable'); ordered=projection[order]
        cdf=.5*normal_cdf((ordered[:,None]-means@direction)/.25).sum(1)
        hi=np.arange(1,n+1)/n-cdf; lo=cdf-np.arange(n)/n
        index_hi=int(hi.argmax()); index_lo=int(lo.argmax())
        high=float(hi[index_hi])>=float(lo[index_lo]); index=index_hi if high else index_lo
        point=float(ordered[index]); empirical=(index+1)/n if high else index/n
        event=projection<=point if high else projection<point
        desired=.5*normal_cdf((point-means@direction)/.25)
        actual=np.array([int((event&(label==k)).sum())/n for k in range(2)])
        witnesses.append(dict(direction_index=j,theta_radians=float(theta),direction=direction.tolist(),coordinate=point,side='empirical_right_minus_target' if high else 'target_minus_empirical_left',empirical_cdf=empirical,target_cdf=float(cdf[index]),absolute_gap=float(hi[index_hi] if high else lo[index_lo]),target_component_contributions=desired.tolist(),empirical_nearest_component_contributions=actual.tolist(),empirical_minus_target_by_component=(actual-desired).tolist(),cdf_count_matches_sorted_side=bool(abs(actual.sum()-empirical)<1e-14)))
    worst=max(witnesses,key=lambda r:r['absolute_gap'])
    return dict(component_assignment_scope='nearest target center; no unobserved mixture labels',components=components,worst_analytic_cdf_location=worst,projection_gaps=[r['absolute_gap'] for r in witnesses])


previous=json.loads(PREVIOUS.read_text());bind(PREVIOUS)
records=[]; loaded={}; snapshot=None
for family in ('atlas','e22'):
    sp=RAW/family/'study.json';bind(sp,STUDY_PINS[family]); study=json.loads(sp.read_text())
    row=next(r for r in study['rows'] if r['family']==family)
    parent=study['parents'][family]; parent_path=Path(parent['path']);bind(parent_path,parent['receipt_sha256'])
    receipt=json.loads(parent_path.read_text()); assert receipt==parent['receipt']
    assert receipt['source']['commit']==SOURCE and receipt['verdict']=='PASS' and receipt['default_protocol_complete']
    result_path=Path(row['result_path']);bind(result_path,row['result_sha256']);result=json.loads(result_path.read_text())
    assert result['status']=='COMPLETE' and result['completed_steps']==1350 and result['full_protocol_complete']
    assert result['verdict']=='FAIL' and result['compound_hold']['hold_passed']==3
    files={'study':bind(sp),'original_receipt':bind(parent_path),'continuation_result':bind(result_path)}
    for prefix,sourcefiles in [('original',parent['artifacts']),('continuation',row['artifacts'])]:
        for role,pin in sourcefiles.items():
            a=bind(pin['path'],pin['sha256']); assert a['bytes']==pin['bytes'];files[prefix+'/'+role]=a
    refs=next(r for r in previous['records'] if r['family']==family)
    for role in ('request','log'):
        files[role]=bind(refs[role]['path'],refs[role]['sha256'])
    with np.load(parent['artifacts']['observations.npz']['path'],allow_pickle=False) as z: old={k:z[k].copy() for k in z.files}
    with np.load(row['artifacts']['appended-observations.npz']['path'],allow_pickle=False) as z: new={k:z[k].copy() for k in z.files}
    states={1200:torch.load(parent['artifacts']['final-state.pt']['path'],map_location='cpu',weights_only=True),1350:torch.load(row['artifacts']['continued-state.pt']['path'],map_location='cpu',weights_only=True)}
    assert states[1200]['completed_steps']==states[1200]['trainer']['completed_steps']==1200
    assert states[1350]['completed_steps']==states[1350]['trainer']['completed_steps']==1350
    assert states[1200]['recipe']==states[1350]['recipe']
    assert safe(states[1200]['recipe'])==receipt['recipe']
    assert receipt['recipe']['total_steps'] is None
    assert states[1200]['trainer']['max_steps']==1200 and states[1350]['trainer']['max_steps']==1350
    trace=[]
    for o in receipt['observations']+result['observations']:
        step=o['step']; arrays=old if step<=1200 else new
        x=arrays[f'step{step}_view0_samples']; t=arrays[f'step{step}_view0_target']
        assert x.shape==(4096,2) and x.dtype==np.float32 and np.isfinite(x).all()
        item=dict(step=step,phase='original' if step<=1200 else 'appended',recorded_passed=o['passed'],recorded_failed_bounds=o['failed_bounds'],recorded_metrics=o['metrics'],target_sha256=hashlib.sha256(t.tobytes()).hexdigest(),samples_sha256=hashlib.sha256(x.tobytes()).hexdigest())
        if step>=900:
            item['descriptive_shape']=shape_description(x)
            item['cdf_location_gap_minus_recorded_ks']=item['descriptive_shape']['worst_analytic_cdf_location']['absolute_gap']-o['metrics']['projection_ks']
            assert abs(item['cdf_location_gap_minus_recorded_ks'])<1e-12
        trace.append(item)
    original_to_new={owner:compare(states[1200]['trainer'][owner],states[1350]['trainer'][owner]) for owner in ('models','optimizers','controller','policy','lr_settle','birth_death','row_evidence','surprise','output_noise','streams','cpu_rng','cuda_rng')}
    endpoints=[]
    for step,state in states.items():
        trainer=state['trainer'];controller=trainer['controller']
        health=finite_tensor_summary({'models':trainer['models'],'optimizer_states':[opt['state'] for opt in trainer['optimizers']]})
        assert health['nonfinite_values']==0
        endpoints.append(dict(step=step,execution_cap=trainer['max_steps'],model_and_optimizer_numeric_health=health,selected_serving=trainer['policy'],backend=trainer.get('backend_selection'),controller_scalars={k:controller[k] for k in ('variant','updates','reopens','closed','mobility','game_trust','game_ratio','payoff_error','alignment','last_cosine')},latent_bandwidth=safe(controller['latent_bandwidth']),birth_death_counters=trainer['birth_death']['counters'],row_evidence_counters=trainer['row_evidence']['counters'],surprise={k:trainer['surprise'][k] for k in ('fires','anchor_events','since_calm','last_ratio','last_ratios','log')},reopen_guard=trainer.get('reopen_guard'),effective_lrs=[[g['lr'] for g in opt['param_groups']] for opt in trainer['optimizers']],named_rng_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in trainer['streams'].items()},data_rng_sha256=hashlib.sha256(state['data_rng'].numpy().tobytes()).hexdigest()))
    records.append(dict(family=family,candidate_id=parent['candidate_id'],case=receipt['case'],original_source=receipt['source'],continuation_source=study['source'],runtime=result['runtime'],recipe=receipt['recipe'],protocol_original=receipt['protocol'],protocol_continuation=result['protocol'],original_gate='PASS',original_study_gate='INCOMPLETE',continuation_gate='FAIL',compound_hold=result['compound_hold'],recorded_restore=result['restore'],recorded_checkpoint_sampler_parity=result['checkpoint_sampler_parity'],recorded_observer_purity=result['observer_purity'],original_inputs_unchanged=result['original_inputs_unchanged'],artifacts=files,paid_seconds=row['paid_wall_seconds'],endpoints=endpoints,endpoint_owner_changes=original_to_new,trace=trace))
    loaded[family]=(receipt,result,old,new,states)
    if snapshot is None:
        snapshot=Path(study['scientific_snapshot'])
        for relative,pin in receipt['source']['files_sha256'].items(): bind(snapshot/relative,pin)
        helper=Path(row['command'][3]);bind(helper,'adcfbd92b39e3affa2626965bf32101bd4ade8a0ad00c9a4e4bbd4a0df4a0b58')


cross={}
for step in (1200,1350):
    left=loaded['atlas'][4][step]; right=loaded['e22'][4][step]
    cross[str(step)]={'owners':{k:compare(left['trainer'].get(k),right['trainer'].get(k)) for k in ('models','optimizers','controller','policy','lr_settle','birth_death','row_evidence','surprise','reopen_guard','output_noise','streams','cpu_rng','cuda_rng','recipe','backend_selection')},'data_rng':compare(left['data_rng'],right['data_rng'])}
array_comparisons=[]
for label,index in [('original',2),('appended',3)]:
    left,right=loaded['atlas'][index],loaded['e22'][index];assert set(left)==set(right)
    array_comparisons.append(dict(scope=label,array_count=len(left),all_shape_dtype_bytes_equal=all(a.shape==right[k].shape and a.dtype==right[k].dtype and a.tobytes()==right[k].tobytes() for k,a in left.items()),scalar_values=sum(a.size for a in left.values())))
for f in ('atlas','e22'):
    old,new=loaded[f][2:4]
    assert all(old['step1200_view0_target'].tobytes()==new[f'step{s}_view0_target'].tobytes() for s in (1250,1300,1350))
for history in previous['engineering_history']['records']:
    bind(history['study_path'],history['study_sha256'])
    for artifact in history['artifacts'].values():bind(artifact['path'],artifact['sha256'])
bind(previous['raw_summary']['path'],previous['raw_summary']['sha256'])
for path,pin in inputs.items():assert sha(path)==pin
assert not torch.cuda.is_initialized()
packet=dict(schema='pg_best_baseline_retained_debug_v1',scope='Existing C6 original and H2 persistence evidence; original noisy Atlas19 is a separate cohort',operations={'models':0,'restores':0,'new_draws':0,'official_scorer_calls':0,'optimizer_updates':0,'cuda_initialized':False},input_files_unchanged=True,records=records,cross_family_endpoint_comparisons=cross,cross_family_array_comparisons=array_comparisons,source={'scientific_commit':SOURCE,'verified_scientific_files':148,'snapshot':str(snapshot),'helper_sha256':'adcfbd92b39e3affa2626965bf32101bd4ade8a0ad00c9a4e4bbd4a0df4a0b58'},evaluation_identity={'protocol_seed':24002,'primary_sampling_seed':34002,'displayed_target_seed':134005,'historical_comparison_reference_seed':991,'sliced_w1_projection_seed':992,'projection_cdf':'analytic fixed32directions; no RNG','target_scale_metric':'RMS normalization of comparison reference, not law-rescaling'},retained_engineering_history=previous['engineering_history'],retained_continuation_cost=previous['cost'],bounded_followups=[{'status':'PROPOSED_NOT_EXECUTED','kind':'observer_trace','purpose':'Record selected fast/average serving, per-role effective/applied update norms, prior row scale/hold, critic payoff and controller/cumulative event counters at every future scored boundary; state/RNG parity control before spending. It preserves all gates and adds no tuning axis.'},{'status':'PROPOSED_NOT_EXECUTED','kind':'endpoint_factor_attribution','purpose':'Use an independently named frozen-state diagnostic comparing the actual paired G/prior average and fast serving and saved controller geometry under identical evaluation RNG; never substitute an alternate score for the primary FAIL. Missing1250/1300 checkpoints first need explicit acquisition, not invented arrays.'}],missing=['1250 and1300 complete model/prior/controller/optimizer checkpoints','Per-update applied-rate/row-hold/critic payoff/loss/gradient stream','Selected-serving flag at intermediate scored boundaries','Paired average-serving sample arrays'],limits=['Recorded counters exclude new discrete birth/death moves, surprise fires or reopens during1201–1350; continuous generator/prior/controller/row-control dynamics remain coupled.','Matched arrays, models and optimizer memories do not imply complete policy-state equivalence or equal unsaved intermediate actions.','Center/CDF co-movement describes the failure but does not prove its generator, prior, critic or sampler cause.','No current retained parity evidence indicates observer corruption; absence of a full unobserved150-update GPU twin prevents a universal parity claim.','No historical noisy19/19, clean-MoG, speed, default or eight-case qualification is inferred.'],input_index=[{'path':p,'sha256':h} for p,h in sorted(inputs.items())])
(OUT/'retained-hold-debug.json').write_text(json.dumps(safe(packet),indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'records':2,'input_files':len(inputs),'arrays_equal':array_comparisons,'new_updates_draws_models_scorers':0,'all_inputs_unchanged':True}))

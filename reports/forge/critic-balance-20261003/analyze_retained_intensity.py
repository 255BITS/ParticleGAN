"""Draw-free saved-state/array diagnosis; no public fixture/model construction."""
from pathlib import Path
import hashlib,json,math,struct
import numpy as np
import torch

torch.set_num_threads(1)
ROOT=Path('/ml2/hypergan/forge-critic-balance-20261003-v2')
OUT=Path('/ml2/hypergan/critic-balance-failure-analysis-20261003')
CASE='image-develop-img_intensity2-source-transpose12'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fingerprint(x):
 h=hashlib.sha256()
 def visit(v):
  h.update(type(v).__name__.encode()+b'\0')
  if isinstance(v,torch.Tensor):
   t=v.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,np.ndarray):
   h.update(str((v.dtype,v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):visit(k);visit(v[k])
  elif isinstance(v,(list,tuple)):
   for q in v:visit(q)
  elif isinstance(v,float):h.update(struct.pack('>d',v))
  else:h.update(repr(v).encode())
 visit(x);return h.hexdigest()

def artifact(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}

rows={}; receipts={}; arrays={}; states={}; studies={}
for f in ('atlas','e22'):
 p=ROOT/f/'study.json';s=json.loads(p.read_text());studies[f]=s
 t=next(t for t in s['trials'] if t['family']==f)
 row=next(c for c in t['cases'] if c['id']==CASE);rp=Path(row['receipt_path']);r=json.loads(rp.read_text());rows[f]=row;receipts[f]=r
 assert sha(rp)==row['receipt_sha256']
 for n,a in r['artifacts'].items():
  p=rp.parent/n;assert sha(p)==a['sha256'] and p.stat().st_size==a['bytes']
 snapshot=Path(s['execution_source']['snapshot_path'])
 for n,value in r['source']['files_sha256'].items():assert sha(snapshot/n)==value
 arrays[f]={k:v.copy() for k,v in np.load(rp.parent/'observations.npz',allow_pickle=False).items()}
 states[f]=torch.load(rp.parent/'final-state.pt',map_location='cpu',weights_only=False)

r=receipts['atlas'];a=arrays['atlas'];state=states['atlas'];t=state['api_state'];c=t['controller']
compact=[]
for step in (0,25,50,100,300,600):
 x=a[f'step{step}_view0_samples'];target=a[f'step{step}_view0_target'];patch=x[:,:,2:6,2:6];mask=np.ones((8,8),bool);mask[2:6,2:6]=False
 o=next(o for o in r['observations'] if o['step']==step)
 # Additional descriptive summaries of captured draws; original verdict/metrics are unchanged.
 d=np.sqrt(np.square(x.astype(np.float64)[:,None]-target.astype(np.float64)[None]).mean((2,3,4)))
 compact.append({'step':step,'original_metrics':o['metrics'],'original_failed_bounds':o['failed_bounds'],
                 'nearest_assignment_counts':np.bincount(d.argmin(1),minlength=2).tolist(),
                 'all_arrays_finite':bool(np.isfinite(x).all() and np.isfinite(target).all()),
                 'pixel_mean':float(x.mean()),'pixel_min':float(x.min()),'pixel_max':float(x.max()),
                 'rms_to_blank':float(np.sqrt(np.square(x.astype(np.float64)).mean())),
                 'patch_mean':float(patch.mean()),'patch_max':float(patch.max()),
                 'background_mean':float(x[:,:,:, :][:,:,mask].mean()),
                 'patch_mean_sigmoid_logit_derivative':float((patch*(1-patch)).mean()),
                 'fraction_pixels_le_1e_4':float((x<=1e-4).mean())})

same_groups=[];different_groups=[]
for key in sorted(set(t)|set(states['e22']['api_state'])):
 if key in t and key in states['e22']['api_state'] and fingerprint(t[key])==fingerprint(states['e22']['api_state'][key]):same_groups.append(key)
 else:different_groups.append(key)
obs_same=all((x['step'],x['metrics'],x['passed'],x['failed_bounds'],x['views'])==(y['step'],y['metrics'],y['passed'],y['failed_bounds'],y['views']) for x,y in zip(r['observations'],receipts['e22']['observations']))
assert obs_same and len(r['observations'])==len(receipts['e22']['observations'])==25
npz_same=set(a)==set(arrays['e22']) and all(fingerprint(a[k])==fingerprint(arrays['e22'][k]) for k in a)
assert npz_same
assert all(not o['passed'] and o['failed_bounds']==['distribution_tv','finite_template_tv','hq','modes'] for o in r['observations'])
recipe_differences={k:{'atlas':r['recipe'].get(k),'e22':receipts['e22']['recipe'].get(k)} for k in sorted(set(r['recipe'])|set(receipts['e22']['recipe'])) if r['recipe'].get(k)!=receipts['e22']['recipe'].get(k)}

grad=c['previous_gradient'];opt=t['optimizers'][0];bias=opt['state'][opt['param_groups'][0]['params'][-1]]
reg=t['optimizers'][1]['regularizer']['record']
final_rate=[[g['lr'] for g in opt['param_groups']] for opt in t['optimizers']]
owner_summary={'public_completed_steps':t['completed_steps'],'selected_serving':t['policy']['served_source'],
 'learned_training_output_sigma':t['policy']['last_output_sigma'],'primary_output_noise_added':0,
 'actual_backend_atlas':t['backend_selection'],'atlas_guard':t['reopen_guard'],
 'initial_lrs':t['initial_lrs'],'terminal_effective_lrs':final_rate,
 'terminal_critic_fraction_of_nominal':final_rate[1][0]/t['initial_lrs'][1][0],
 'terminal_smoothed_payoff_error':c['payoff_error'],'critic_damping_function_at_terminal_error':1/(1+c['payoff_error']**2),
 'effective_rate_vs_error_boundary_warning':'Saved last optimizer LR was assigned before the final controller observation; these endpoint quantities need not give exactly the same factor.',
 'controller_updates':c['updates'],'controller_reopens':c['reopens'],'controller_mobility':c['mobility'],
 'surprise_fires':t['surprise']['fires'],'surprise_anchor_events':t['surprise']['anchor_events'],
 'birth_death_counters':t['birth_death']['counters'],'row_evidence_counters':t['row_evidence']['counters'],
 'last_generator_gradient':{'count':grad.numel(),'norm':float(grad.norm()),'max_abs':float(grad.abs().max()),'mean_abs':float(grad.abs().mean())},
 'output_bias_optimizer':{k:float(bias[k]) for k in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq')},
 'critic_regularizer':{k:reg.get(k) for k in ('calls','observed_steps','lr_max','lr_last','anchor_started','last_sur','sur_base','w','alpha','ema_updates','formulation')},
 'missing_artifacts':['Per-update GAN generator/discriminator losses','Per-step effective role rates and served-selection trace between retained sample boundaries','Intermediate model/optimizer checkpoints before the step25 collapse','Matched current-source old-D-rate control'],
 'local_signature':'Captured foreground sigmoid outputs imply almost zero local output-logit sensitivity. The retained final G gradient is small but nonzero; its negative output-bias gradient points toward a brighter bias under minimization. Large retained AMSGrad second moment is a recovery-scale observation, not an initial-collapse cause.'}

family_records=[]
for f in ('atlas','e22'):
 row=rows[f];s=studies[f];r1=receipts[f];p=Path(row['receipt_path']);family_records.append({'family':f,'trial_id':next(t['id'] for t in s['trials'] if t['family']==f),'study':artifact(ROOT/f/'study.json'),
 'receipt':artifact(p),'log':artifact(Path(row['log_path'])),'artifacts':{n:{**v,'path':str(p.parent/n)} for n,v in r1['artifacts'].items()},
 'source_commit':r1['source']['commit'],'execution_source_digest':s['execution_source']['digest'],'verified_public_python_files':len(r1['source']['files_sha256']),
 'recipe':r1['recipe'],'runtime':r1['runtime'],'original_status':r1['status'],'original_verdict':r1['verdict'],'default_protocol_complete':r1['default_protocol_complete'],
 'completed_updates':r1['completed_updates'],'study_gate':row['study_gate'],'acquisition_hold':row['acquisition_hold'],'protocol':r1['protocol'],
 'requested_recipe_overrides':r1['requested_recipe_overrides'],'paid_supervisor_seconds':row['paid_wall_seconds'],'acquisition_seconds':r1['elapsed_seconds'],
 'all_scored_checks':len(r1['observations']),'passing_checks':sum(o['passed'] for o in r1['observations']),'post_update_checks':len(r1['observations'])-1,
 'terminal_failed_bounds':r1['observations'][-1]['failed_bounds'],'final_metrics':r1['observations'][-1]['metrics'],
 'remaining_case_statuses':{z['id']:z['status'] for z in next(t for t in s['trials'] if t['family']==f)['cases'] if z['id']!=CASE}})

old=OUT/'frozen-hold-analysis.json';carry=studies['atlas']['spec']['engineering_carryover'];paid=sum(rows[f]['paid_wall_seconds'] for f in rows)
report={'schema':'particlegan_draw_free_critic_balance_intensity_diagnosis_v1','status':'COMPLETE_READ_ONLY','ordinary_updates':0,'new_samples':0,'model_constructions':0,'cuda_contexts':0,
 'scientific_source_commit':'a956c6fc447bbe1314ff43ec224825a0e941eb55','scientific_base':'4749b2780add539df4bd8d2dd1d3cc9f002f77ad',
 'case_id':CASE,'goal':r['case']['goal'],'thresholds':r['case']['thresholds'],'sampling':r['case']['sampling'],'family_records':family_records,
 'denominator':{'families':2,'required_cases_each':8,'required_cells':16,'original_FAIL':2,'study_FAIL':2,'UNKNOWN':14,'capacity_SUPPORTED':16,'qualified_whole_configs':0},
 'cross_family':{'observation_metrics_flags_views_exact':obs_same,'observations':25,'metric_scalar_comparisons':sum(len(o['metrics']) for o in r['observations']),
 'npz_bytes_identical':sha(Path(rows['atlas']['receipt_path']).parent/'observations.npz')==sha(Path(rows['e22']['receipt_path']).parent/'observations.npz'),
 'npz_array_count':len(a),'npz_scalar_count':sum(v.size for v in a.values()),'all_retained_array_bytes_equal':npz_same,
 'models':{k:{'equal':fingerprint(t['models'][k])==fingerprint(states['e22']['api_state']['models'][k]),'atlas_typed_sha256':fingerprint(t['models'][k]),'e22_typed_sha256':fingerprint(states['e22']['api_state']['models'][k])} for k in t['models']},
 'identical_trainer_state_groups':same_groups,'different_or_family_specific_trainer_state_groups':different_groups,
 'caller_data_generator_equal':fingerprint(state['data_generator'])==fingerprint(states['e22']['data_generator']),
 'recipe_differences':recipe_differences,'whole_checkpoint_equal':fingerprint(state)==fingerprint(states['e22']),
 'limit':'No equality claim for wall-clock logs, global RNG, complete checkpoints or unreached native backends. Matching recorded outputs and internal groups are not an independent replication or full policy equivalence.'},
 'selected_saved_array_summaries':compact,'saved_owner_summary':owner_summary,
 'previous_hold_analysis':artifact(old),'engineering_carryover':carry,'new_scientific_supervisor_seconds':paid,'total_prior_paid_seconds':paid+carry['paid_seconds'],
 'source_claims':[{'file':'benchmarks/toy_audit/api_images.py','sha256':r['source']['files_sha256']['benchmarks/toy_audit/api_images.py'],'lines':[541,568,719],
 'claim':'Per-image RMSE neighborhoods determine HQ/mode/mass; sigmoid host output. Mean RMSE is diagnostic.'},
 {'file':'particlegan/continuous.py','sha256':r['source']['files_sha256']['particlegan/continuous.py'],'lines':[314,330],
 'claim':'Smoothed positive G-minus-D loss balance controls reciprocal-square critic payoff damping.'},
 {'file':'particlegan/policy.py','sha256':r['source']['files_sha256']['particlegan/policy.py'],'lines':[650,658,1197],
 'claim':'Role stationarity scale is followed by critic-only payoff damping; policy selects fast or averaged serving.'},
 {'file':'particlegan/feature_cells.py','sha256':r['source']['files_sha256']['particlegan/feature_cells.py'],'lines':[37,44],
 'claim':'Finite resolution selects a reference-kNN fallback at small populations; large native hosts can activate distinct feature-cell law.'}],
 'next_hypothesis':{'status':'ACCEPTED_DECLARATION_NOT_EXECUTED','requested_shared_overrides':{'lr':.00265625,'prior_lr_mult':3.,'d_lr_mult':4.5},
 'ordinary_initialization':'Fresh original seed/host initialization, not a continuation or transplant of the failed trained state',
 'change':'Half nominal G and learned-output-noise step rate; retain nominal prior and D rates .00796875/.011953125 across supported role-owned hosts',
 'rationale':'Target the observed early near-black sigmoid excursion and small retained terminal recovery gradient without replacing the public objective, sampler or controller.',
 'limits':'Rate changes alter endogenous policy trajectories. No assertion that the old critic rate caused failure, no monotonicity or success prediction. Acquisition may slow. Original gates, horizons, seed and full16 denominator remain fixed.',
 'capacity':'Requires sixteen new candidate-bound zero-update construction/sampler outcomes with current source/Recipe/serving and fresh optimizer/controller ownership; old capacity verdicts cannot be borrowed.',
 'budget':'Parent retains the original 15360-second campaign ceiling and exact 59.116680497769266 prior debit; this report creates no reservation or new allowance.'}}
(OUT/'intensity-failure-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'report':str(OUT/'intensity-failure-analysis.json'),'sha256':sha(OUT/'intensity-failure-analysis.json'),'family_count':len(family_records),'checks':25,'array_values':report['cross_family']['npz_scalar_count'],'new_supervisor_seconds':paid,'prior_total':report['total_prior_paid_seconds']}))

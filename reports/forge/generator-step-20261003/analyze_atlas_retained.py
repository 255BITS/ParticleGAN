"""Saved-array/state descriptions only: no model, sampler, scorer or update."""
from pathlib import Path
import hashlib,json,struct
import numpy as np
import torch

torch.set_num_threads(1)
OUT=Path('/ml2/hypergan/generator-step-failure-analysis-20261003')
CASE='image-develop-img_intensity2-source-transpose12'
BASE=Path('/ml2/hypergan/forge-generator-step-20261003/atlas')
OLD=Path('/ml2/hypergan/forge-critic-balance-20261003-v2/atlas')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def artifact(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def fingerprint(x):
 h=hashlib.sha256()
 def visit(v):
  h.update(type(v).__name__.encode()+b'\0')
  if isinstance(v,torch.Tensor):
   a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):visit(k);visit(v[k])
  elif isinstance(v,(list,tuple)):
   for a in v:visit(a)
  elif isinstance(v,float):h.update(struct.pack('>d',v))
  else:h.update(repr(v).encode())
 visit(x);return h.hexdigest()
def load(base,expected_study_sha=None,expected_receipt_sha=None):
 p=base/'study.json';s=json.loads(p.read_text())
 if expected_study_sha:assert sha(p)==expected_study_sha
 trial=next(t for t in s['trials'] if t['family']=='atlas');row=next(c for c in trial['cases'] if c['id']==CASE)
 p=Path(row['receipt_path']);r=json.loads(p.read_text());assert sha(p)==row['receipt_sha256']
 if expected_receipt_sha:assert sha(p)==expected_receipt_sha
 for name,v in r['artifacts'].items():assert sha(p.parent/name)==v['sha256'] and (p.parent/name).stat().st_size==v['bytes']
 snapshot=Path(s['execution_source']['snapshot_path'])
 for name,v in r['source']['files_sha256'].items():assert sha(snapshot/name)==v
 state=torch.load(p.parent/'final-state.pt',map_location='cpu',weights_only=False)
 arrays={k:v.copy() for k,v in np.load(p.parent/'observations.npz',allow_pickle=False).items()}
 return s,trial,row,r,state,arrays
s,trial,row,r,state,arrays=load(BASE,'02ed35c689ae1a6f5df11bab6e42b2a07bb389d6ea3db146f3357e226411e2f7','3faae2709afbea454b747ecde87169015792bf1f9eb2f2561b6e090f8d8a40ad')
old_s,old_trial,old_row,old_r,old_state,old_arrays=load(OLD)
new=state['api_state'];old=old_state['api_state']
assert r['completed_updates']==600 and r['status']=='COMPLETE' and r['default_protocol_complete'] and r['verdict']=='FAIL'
assert len(r['observations'])==25 and sum(o['passed'] for o in r['observations'])==0
assert row['acquisition_hold']=={'reason':'no five-check acquisition window','status':'FAIL'}
assert all(o['failed_bounds']==['distribution_tv','finite_template_tv','hq','modes'] for o in r['observations'])

def selected_description(a,r):
 result=[]
 for step in (0,25,50,100,300,600):
  x=a[f'step{step}_view0_samples'];patch=x[:,:,2:6,2:6];o=next(o for o in r['observations'] if o['step']==step)
  result.append({'step':step,'recorded_metrics':o['metrics'],'recorded_failed_bounds':o['failed_bounds'],
                 'all_samples_finite':bool(np.isfinite(x).all()),'pixel_mean':float(x.mean()),
                 'rms_to_blank':float(np.sqrt((x.astype(np.float64)**2).mean())),
                 'patch_mean':float(patch.mean()),'patch_max':float(patch.max()),
                 'patch_mean_sigmoid_derivative':float((patch*(1-patch)).mean()),
                 'fraction_pixels_le_1e_4':float((x<=1e-4).mean())})
 return result

def owner_summary(a):
 c=a['controller'];g=c['previous_gradient'];opts=a['optimizers'];bid=opts[0]['param_groups'][0]['params'][-1];bias=opts[0]['state'][bid]
 assert float(g[-1])==float(bias['exp_avg'])
 z=a['models']['prior']['z'];return {'completed_steps':a['completed_steps'],'selected_serving':a['policy']['served_source'],
  'learned_training_output_sigma':a['policy']['last_output_sigma'],'initial_lrs':a['initial_lrs'],
  'terminal_effective_lrs':[[v['lr'] for v in opt['param_groups']] for opt in opts],
  'terminal_critic_fraction_of_nominal':opts[1]['param_groups'][0]['lr']/a['initial_lrs'][1][0],
  'terminal_smoothed_payoff_error':c['payoff_error'],'damping_function_at_terminal_error':1/(1+c['payoff_error']**2),
  'boundary_note':'The optimizer LR was assigned before the last payoff observation; ratio and final-error damping need not agree exactly.',
  'last_G_gradient':{'count':g.numel(),'norm':float(g.norm()),'max_abs':float(g.abs().max()),'signed_output_bias':float(g[-1])},
  'output_bias_optimizer':{k:float(bias[k]) for k in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq')},
  'prior_raw_geometry':{'rows':z.shape[0],'dimensions':z.shape[1],'coordinate_mean':z.mean(0).tolist(),'coordinate_std_population':z.std(0,unbiased=False).tolist(),'frobenius_norm':float(z.norm())},
  'controller_latent_bandwidth':c['latent_bandwidth'].tolist(),'controller_mobility':c['mobility'],'controller_updates':c['updates'],'controller_reopens':c['reopens'],
  'birth_death_counters':a['birth_death']['counters'],'row_evidence_counters':a['row_evidence']['counters'],
  'surprise':{k:a['surprise'][k] for k in ('fires','anchor_events','last_ratio','log')},'atlas_guard':a['reopen_guard'],'backend_selection':a['backend_selection']}
new_owner,old_owner=owner_summary(new),owner_summary(old)
model_diffs={}
for name in ('G','D','prior','ema_G','ema_prior'):
 ks=list(new['models'][name]);assert ks==list(old['models'][name])
 diffs=[new['models'][name][k].double()-old['models'][name][k].double() for k in ks]
 model_diffs[name]={'identical':fingerprint(new['models'][name])==fingerprint(old['models'][name]),'parameters':sum(v.numel() for v in diffs),
                    'max_abs_difference':max(float(v.abs().max()) for v in diffs),
                    'l2_endpoint_difference':float(torch.stack([v.square().sum() for v in diffs]).sum().sqrt())}
recipe_changed={key:{'previous':old_r['recipe'].get(key),'new':r['recipe'].get(key)} for key in sorted(set(old_r['recipe'])|set(r['recipe'])) if old_r['recipe'].get(key)!=r['recipe'].get(key)}
source_changed={key:{'previous':old_r['source']['files_sha256'].get(key),'new':r['source']['files_sha256'].get(key)} for key in sorted(set(old_r['source']['files_sha256'])|set(r['source']['files_sha256'])) if old_r['source']['files_sha256'].get(key)!=r['source']['files_sha256'].get(key)}
new_desc,old_desc=selected_description(arrays,r),selected_description(old_arrays,old_r)
report={'schema':'particlegan_generator_step_atlas_draw_free_diagnosis_v1','status':'COMPLETE_READ_ONLY','family':'atlas','case_id':CASE,
 'scientific_source_commit':r['source']['commit'],'execution_source_digest':s['execution_source']['digest'],
 'new_evidence':{'study':artifact(BASE/'study.json'),'receipt':artifact(Path(row['receipt_path'])),'log':artifact(Path(row['log_path'])),
 'artifacts':{k:{**v,'path':str(Path(row['receipt_path']).parent/k)} for k,v in r['artifacts'].items()},'runtime':r['runtime'],'resolved_recipe':r['recipe'],
 'verified_python_files':len(r['source']['files_sha256']),'case':r['case'],'protocol':r['protocol'],
 'scientific_status':r['status'],'original_gate':r['verdict'],'study_gate':row['study_gate'],'acquisition_hold':row['acquisition_hold'],
 'completed_updates':r['completed_updates'],'default_protocol_complete':r['default_protocol_complete'],'checks':25,'post_update_checks':24,'passed_checks':0,
 'final_metrics':r['observations'][-1]['metrics'],'final_failed_bounds':r['observations'][-1]['failed_bounds'],
 'unknown_cases':[{'id':c['id'],'status':c['status']} for c in trial['cases'][1:]],
 'new_paid_supervisor_seconds':row['paid_wall_seconds'],'new_acquisition_seconds':r['elapsed_seconds'],'unmeasured_reservation_seconds':s['unmeasured_interrupt_reservation_seconds'],
 'existing_debit':s['spec']['prior_carryover'],'cost_limit':'Old engineering/scientific debit is retained separately; this descriptive comparison adds no pooled result, time ranking or allowance.'},
 'selected_new_saved_array_summaries':new_desc,'new_owner_summary':new_owner,
 'predecessor_descriptive_comparison':{'scientific_source_commit':old_r['source']['commit'],'receipt':artifact(Path(old_row['receipt_path'])),
 'original_gate':old_r['verdict'],'study_gate':old_row['study_gate'],'selected_saved_array_summaries':old_desc,'owner_summary':old_owner,
 'changed_resolved_recipe_fields':recipe_changed,'changed_receipt_source_files':source_changed,
 'initial_arrays_bitwise_equal':all(fingerprint(arrays[k])==fingerprint(old_arrays[k]) for k in arrays if k.startswith('step0_')),
 'caller_data_rng_final_equal':fingerprint(state['data_generator'])==fingerprint(old_state['data_generator']),
 'named_training_streams_final_equal':fingerprint(new['streams'])==fingerprint(old['streams']),
 'final_model_endpoint_differences':model_diffs,
 'limits':'Matching initial captured arrays and final stream states do not establish a complete paired trajectory. All three declared rate multipliers and helper source identity differ; effective rates/gradients are endogenous. These are descriptive comparisons, not an isolated generator/noise or D/prior counterfactual.'},
 'numerical_interpretation':'Slower nominal G/noise steps reduce training output sigma and produce less-saturated captured foreground than the predecessor, but every retained primary check still rejects all draws. The larger nonzero final G gradient and positive output-bias gradient preclude asserting missing gradients alone as the established cause. Foreground local sensitivity remains near zero; exact causal entry/escape mechanism is unresolved.',
 'missing_artifacts':['Per-update G/D adversarial losses','Per-update effective role rates','Intermediate models/optimizer state before step25','Fixed-G prior intervention and alternate served/EMA samples (not acquired)'],
 'e22_status':'NOT_INSPECTED_PENDING_ROOT_IMMUTABLE_NOTICE','new_training_updates':0,'new_samples':0,'model_constructions':0,'model_restores':0,'rescoring_calls':0,'cuda_contexts':0,
 'analysis_script':artifact(Path(__file__))}
(OUT/'atlas-intensity-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'report':str(OUT/'atlas-intensity-analysis.json'),'sha256':sha(OUT/'atlas-intensity-analysis.json'),'source_changed':list(source_changed),'final_prior_l2_difference':model_diffs['prior']['l2_endpoint_difference'],'initial_arrays_equal':report['predecessor_descriptive_comparison']['initial_arrays_bitwise_equal'],'named_streams_equal':report['predecessor_descriptive_comparison']['named_training_streams_final_equal'],'bias':new_owner['output_bias_optimizer']}))

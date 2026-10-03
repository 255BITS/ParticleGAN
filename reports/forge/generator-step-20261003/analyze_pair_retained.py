"""Exact frozen two-family byte/flag comparisons only; no model or scorer calls."""
from pathlib import Path
import hashlib,json,struct
import numpy as np
import torch

torch.set_num_threads(1)
BASE=Path('/ml2/hypergan/forge-generator-step-20261003')
OUT=Path('/ml2/hypergan/generator-step-failure-analysis-20261003')
CASE='image-develop-img_intensity2-source-transpose12'
PINS={'atlas':('02ed35c689ae1a6f5df11bab6e42b2a07bb389d6ea3db146f3357e226411e2f7','3faae2709afbea454b747ecde87169015792bf1f9eb2f2561b6e090f8d8a40ad'),
      'e22':('6b0fcfd09df7d5b197cefc608a8ae59ef6583a8179e12685a6e117e9acaae5fa','d20dd05b23c10a2ecaac760da441a6f53a44ef67e5041246ab1dd17f92682e90')}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def artifact(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def fp(x):
 h=hashlib.sha256()
 def v(x):
  h.update(type(x).__name__.encode()+b'\0')
  if isinstance(x,torch.Tensor):
   t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(np.ascontiguousarray(x).tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):v(k);v(x[k])
  elif isinstance(x,(list,tuple)):
   for y in x:v(y)
  elif isinstance(x,float):h.update(struct.pack('>d',x))
  else:h.update(repr(x).encode())
 v(x);return h.hexdigest()

studies={};rows={};receipts={};states={};arrays={};records=[]
for f in ('atlas','e22'):
 sp=BASE/f/'study.json';s=json.loads(sp.read_text());assert sha(sp)==PINS[f][0];studies[f]=s
 t=next(t for t in s['trials'] if t['family']==f);row=next(c for c in t['cases'] if c['id']==CASE);rp=Path(row['receipt_path']);r=json.loads(rp.read_text());assert sha(rp)==PINS[f][1]==row['receipt_sha256'];rows[f]=row;receipts[f]=r
 assert t['status']=='FAIL' and r['status']=='COMPLETE' and r['verdict']=='FAIL' and r['default_protocol_complete'] is True and r['completed_updates']==600
 assert r['requested_recipe_overrides']=={'lr':.00265625,'prior_lr_mult':3.,'d_lr_mult':4.5}
 assert len(r['observations'])==25 and all(not o['passed'] and o['failed_bounds']==['distribution_tv','finite_template_tv','hq','modes'] for o in r['observations'])
 for k,v in r['artifacts'].items():assert sha(rp.parent/k)==v['sha256'] and (rp.parent/k).stat().st_size==v['bytes']
 snapshot=Path(s['execution_source']['snapshot_path'])
 for k,v in r['source']['files_sha256'].items():assert sha(snapshot/k)==v
 states[f]=torch.load(rp.parent/'final-state.pt',map_location='cpu',weights_only=False)
 arrays[f]={k:v.copy() for k,v in np.load(rp.parent/'observations.npz',allow_pickle=False).items()}
 records.append({'family':f,'trial_id':t['id'],'study':artifact(sp),'receipt':artifact(rp),'log':artifact(Path(row['log_path'])),
 'artifacts':{k:{**v,'path':str(rp.parent/k)} for k,v in r['artifacts'].items()},'source_commit':r['source']['commit'],'execution_source_digest':s['execution_source']['digest'],
 'verified_python_files':len(r['source']['files_sha256']),'runtime':r['runtime'],'resolved_recipe':r['recipe'],'protocol':r['protocol'],
 'original_status':r['status'],'original_gate':r['verdict'],'study_gate':row['study_gate'],'full_original_protocol':r['default_protocol_complete'],'completed_updates':600,
 'acquisition_hold':row['acquisition_hold'],'checks':25,'post_update_checks':24,'passed_checks':0,'final_metrics':r['observations'][-1]['metrics'],
 'final_failed_bounds':r['observations'][-1]['failed_bounds'],'unknown_case_ids':[c['id'] for c in t['cases'][1:] if c['status']=='UNKNOWN'],
 'new_paid_supervisor_seconds':row['paid_wall_seconds'],'new_acquisition_seconds':r['elapsed_seconds'],'unmeasured_reservation_seconds':s['unmeasured_interrupt_reservation_seconds']})
a=states['atlas']['api_state'];b=states['e22']['api_state'];ar=receipts['atlas'];er=receipts['e22']
observation_equal=all(fp({k:o[k] for k in ('step','metrics','passed','failed_bounds','views')})==fp({k:p[k] for k in ('step','metrics','passed','failed_bounds','views')}) for o,p in zip(ar['observations'],er['observations']))
array_equal=set(arrays['atlas'])==set(arrays['e22']) and all(fp(x)==fp(arrays['e22'][k]) for k,x in arrays['atlas'].items())
same=[];different=[]
for k in sorted(set(a)|set(b)):
 (same if k in a and k in b and fp(a[k])==fp(b[k]) else different).append(k)
models={k:{'equal':fp(a['models'][k])==fp(b['models'][k]),'atlas_typed_sha256':fp(a['models'][k]),'e22_typed_sha256':fp(b['models'][k]),'parameters':sum(v.numel() for v in a['models'][k].values())} for k in a['models']}
assert observation_equal and array_equal and all(v['equal'] for v in models.values())
prior=OUT/'atlas-intensity-analysis.json'
report={'schema':'particlegan_generator_step_pair_draw_free_diagnosis_v1','status':'COMPLETE_READ_ONLY','scientific_source_commit':ar['source']['commit'],
 'source_digest':studies['atlas']['execution_source']['digest'],'case_id':CASE,'records':records,'atlas_detailed_comparison':artifact(prior),
 'required_denominator':{'families':2,'cases_each':8,'cells':16,'full_original_FAIL':2,'study_FAIL':2,'UNKNOWN':14,'capacity_SUPPORTED':16,'fully_qualified_configs':0},
 'equivalence':{'observations_metrics_flags_views_exact':observation_equal,'observation_count':25,'metric_scalars':sum(len(o['metrics']) for o in ar['observations']),
 'all_retained_npz_array_bytes_equal':array_equal,'npz_count':len(arrays['atlas']),'npz_scalar_values':sum(v.size for v in arrays['atlas'].values()),
 'npz_file_bytes_equal':sha(Path(rows['atlas']['receipt_path']).parent/'observations.npz')==sha(Path(rows['e22']['receipt_path']).parent/'observations.npz'),
 'gif_bytes_equal':sha(Path(rows['atlas']['receipt_path']).parent/'goal.gif')==sha(Path(rows['e22']['receipt_path']).parent/'goal.gif'),
 'final_models':models,'identical_api_state_groups':same,'different_or_family_specific_groups':different,
 'caller_data_rng_equal':fp(states['atlas']['data_generator'])==fp(states['e22']['data_generator']),
 'complete_checkpoint_equal':fp(states['atlas'])==fp(states['e22']),
 'recipe_differences':{k:{'atlas':ar['recipe'].get(k),'e22':er['recipe'].get(k)} for k in sorted(set(ar['recipe'])|set(er['recipe'])) if ar['recipe'].get(k)!=er['recipe'].get(k)},
 'limits':'Matching captured output/owners is not an independent replication or full policy equivalence. Whole Recipe/backend/guard/global RNG/timing differ, and native backend cases remain unreached.'},
 'interpretation':{'failure':'Both families retain full600 originalFAIL/studyFAIL and all25 four-bound failures. All final draws miss both finite template neighborhoods; this is brightness collapse.',
 'descriptive_change_from_predecessor':'Patch local sensitivity/brightness is less suppressed at captured steps, training output sigma is smaller, and terminal G gradient is larger, without any primary gate pass.',
 'source_limit':'Public training/provider/package/scorer files common to the preceding scientific cohort have identical hashes; helper/source identity and all three declared rate fields differ. This observation does not isolate one causal owner.',
 'families_have_reason':'Current32-row imagehost selects Atlas reference-kNN fallback and E22 reference behavior. Distinct20000-row native Atlas feature-cell and E22 reference questions, plus potential guard behavior, remain unattempted; neither family borrows native success.'},
 'costs':{'new_scientific_supervisor_seconds_by_family':{f:rows[f]['paid_wall_seconds'] for f in rows},'new_pair_scientific_supervisor_seconds':sum(rows[f]['paid_wall_seconds'] for f in rows),
 'previous_engineering_debit_seconds':4.757908704923466,'previous_scientific_debit_seconds_by_family':{'atlas':27.0030500178691,'e22':27.3557217749767},
 'prior_debit_seconds':59.116680497769266,'campaign_charged_seconds':59.116680497769266+sum(rows[f]['paid_wall_seconds'] for f in rows),
 'original_campaign_ceiling_seconds':15360.,'speed_ranking':False,'reservation_created_by_analysis':False},
 'next_hypothesis':{'status':'PROPOSED_NOT_EXECUTED','shared_overrides':{'lr':.00265625,'prior_lr_mult':3.,'d_lr_mult':2.25},
 'mechanical_delta':'Halve nominal D from .011953125 to .0059765625; preserve current nominal G/noise .00265625 and prior .00796875. Every public mechanism remains enabled.',
 'rationale':'Remaining early dark collapse with large positive smoothed G-minus-D payoff imbalance and stronger D nominal rate relative to G motivates a slower-critic contrast at the completed slow-G initialization/rates.',
 'causal_limit':'This is a test of critic-step sensitivity, not proof the critic caused collapse or that payoff damping failed. Effective D can respond endogenously and need not halve; no monotonicity or success prediction.',
 'protocol':'Two fresh whole configurations/all16cells, same original hosts/seed/priors/gates/horizons/evaluation law, first two smoke prerequisites, unchanged five-confirmation/five-later-hold requirement and first-nonpass stop.',
 'admission':'Requires fresh candidate-bound capacity/source/Recipe/serving evidence, explicit root acceptance and finite remaining-cost accounting. No old Q1 verdict or success may be reused.',
 'budget':'No new reservation/allowance is made. Any accepted test remains inside the original15360 ceiling with113.99425188452005 prior paid seconds retained exactly once.',
 'why_distinct':'This slow-G/P3/D2.25 full trio is neither the failed .0053125/P1.5/D2.25 tuple nor the failed .00265625/P3/D4.5 tuple; not an unchanged rerun or seed sweep.'},
 'ordinary_training_updates':0,'model_constructions':0,'model_restores':0,'new_samples':0,'rescoring_calls':0,'cuda_contexts':0,'analysis_script':artifact(Path(__file__))}
(OUT/'pair-intensity-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'report':str(OUT/'pair-intensity-analysis.json'),'sha256':sha(OUT/'pair-intensity-analysis.json'),'exact_metrics':observation_equal,'exact_arrays':array_equal,'identical_owner_groups':same,'different_owner_groups':different,'new_pair_paid':report['costs']['new_pair_scientific_supervisor_seconds']}))

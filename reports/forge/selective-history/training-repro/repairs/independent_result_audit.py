"""Read-only scientific receipts audit. Writes only independent-result-audit.json."""
import json,hashlib,math
from pathlib import Path
import torch
BASE=Path('/ml2/hypergan/combination-training-20261015')
ATTEMPTS=BASE/'queue/bcap-selective-history-repair-screen-v1'
def read(p):return json.loads(p.read_text())
def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean_task(t):return {k:v for k,v in t.items() if k!='field_ownership'}
def passes(point,thresholds):
 for k,op,b in thresholds:
  x=point.get(k)
  if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):return False
  if not {'>=':x>=b,'<=':x<=b,'==':x==b}[op]:return False
 return True
ARM={'bcap-transport-penalty2-v1':'P','bcap-selective-history-halfjoint-v1':'H'}
rows=[];requests=[];source_check=None
for d in sorted(ATTEMPTS.iterdir()):
 if not d.is_dir():continue
 wrap=read(d/'request.json');q=wrap['request'];job=wrap['job'];task=job['task_id'];spec=q['tasks'][task];arm=ARM[q['candidate']['id']]
 terminal=read(d/'terminal.json');result=read(d/'result.json');raw=read(d/'raw-result.json');graded=read(d/'graded-result.json');grade=graded['grades'][task]
 assert 'task_results' in result, (arm,task,'not finalized')
 requests.append((arm,task,q,job));e=raw['evidence'];curve=e['observations'];n=24 if task=='two_pole' else 96;steps=spec['execution']['steps'];thresholds=spec['evaluation']['thresholds'];expected=sorted({math.ceil(i*steps/n) for i in range(1,n+1)})
 complete=[p['step'] for p in curve]==expected;ps=[passes(p,thresholds) for p in curve];suffix=0
 for ok in reversed(ps):
  if not ok:break
  suffix+=1
 independent='PASS' if complete and suffix>=5 and passes(e['live'],thresholds) else 'FAIL' if complete else 'INCOMPLETE'
 assert independent==grade['gate_status'];assert grade['evaluator_result']['convergence']['passing_suffix']==suffix
 assert graded['raw_hash']==stable(raw);assert graded['source_digest']==q['source']['digest']
 assert terminal['attempt_status']=='completed';assert terminal['grading']==graded
 guards=e['guards'];assert guards['all_finite'] and guards['hooks_exercised'] and guards['unintended_rng_deviations']==0
 assert set(guards['optimizer_updates'].values())=={steps};assert e['provenance_checkpoint']['completed_steps']==steps
 assert q['protocol']['seed']==0 and job['science']['seed']==0
 saved=e['saved_observer_outputs'];saved_path=d/saved['path'];assert sha(saved_path)==saved['sha256'] and saved_path.stat().st_size==saved['bytes']
 artifact=e['provenance_checkpoint'];provenance_path=d/'provenance/provenance-state.pt';assert sha(provenance_path)==artifact['artifact_manifest']['files']['provenance-state.pt']['sha256']
 saved_samples=torch.load(saved_path,map_location='cpu',weights_only=False);assert [x['step'] for x in saved_samples]==expected
 checkpoint=torch.load(provenance_path,map_location='cpu',weights_only=False)
 if task=='two_pole':
  for sample in saved_samples:assert sample['views'][0]['samples'].shape==(12,1) and sample['views'][0]['target'].shape==(12,1)
  opt=checkpoint['optimizers']['generator'][0]
 else:
  for sample in saved_samples:assert sample['samples'].shape==(4096,2)
  opt=checkpoint['trainer']['optimizers'][0]
 history=opt.get('component_history');hs=None
 if arm in ['H']:
  assert history['mode']=={'H':'adversarial'}[arm];assert history['placement']=='before_normalization' and not history['pending'];assert history['signal']['roles']==['generator']
  packet=history['signal']['packets']['generator'];assert packet['count']==steps
  hs={'mode':history['mode'],'placement':history['placement'],'roles':history['signal']['roles'],'count':packet['count'],'zero_steps':packet['zero_steps'],'last':packet['last']}
 else:assert history is None
 init=raw.get('initialization',raw.get('applied',{}).get('initialization',{}))
 tr=result['task_results'][0];assert tr['cost']['execution_seconds']==terminal['elapsed_seconds']
 row={'arm':arm,'candidate':q['candidate']['id'],'attempt':d.name,'task':task,'candidate_revision':q['candidate_revision'],'request_id':wrap['request_id'],'gate':independent,'complete':complete,'steps':steps,'observations':len(curve),'passing_observations':sum(ps),'passing_suffix':suffix,'confirmed_step':grade['evaluator_result']['convergence']['confirmed_step'],'final_gate_metrics':{k:e['live'][k] for k,_,_ in thresholds},'thresholds':thresholds,'guards':guards,'cost':tr['cost'],'terminal_status':terminal['attempt_status'],'grade_raw_hash_matches':True,'source_digest':q['source']['digest'],'receipt_hashes':{f:sha(d/f) for f in ['request.json','raw-result.json','graded-result.json','terminal.json','result.json']},'history':hs,'initialization':init,'stream_hashes':artifact['named_stream_state_sha256'],'saved_outputs':{**saved,'count':len(saved_samples),'absolute_path':str(saved_path)},'provenance_sha256':sha(provenance_path),'actual_existing_media':[str(x) for x in d.rglob('*') if x.suffix in ['.gif','.png','.mp4']]}
 rows.append(row)
 if source_check is None:
  s=q['source'];sp=Path(s['snapshot_path']);bad=[n for n,h in s['files'].items() if sha(sp/n)!=h]
  assert not bad;assert stable(s['files'])==s['digest']
  source_check={'digest':s['digest'],'origin_commit':s['origin_commit'],'path':str(sp),'verified_files':len(s['files']),'mismatches':bad}
assert len(rows)==4 and len({(r['arm'],r['task']) for r in rows})==4
ref=requests[0][2]
shared={key:all(q[key]==ref[key] for _,_,q,_ in requests) for key in ['source','runtime','protocol','compute_profiles','rng']};assert all(shared.values())
comparisons={}
for task in ['two_pole','ring16_acquisition']:
 qs=[q for _,t,q,_ in requests if t==task];rs=[r for r in rows if r['task']==task]
 comparison={'task_declaration_equal_excluding_ownership':all(clean_task(q['tasks'][task])==clean_task(qs[0]['tasks'][task]) for q in qs),'initialization_receipts_identical':all(r['initialization']==rs[0]['initialization'] for r in rs),'final_named_stream_states_identical':all(r['stream_hashes']==rs[0]['stream_hashes'] for r in rs)}
 assert all(comparison.values());comparisons[task]=comparison
recipes={a:q['candidate']['resolved_recipe'] for a,_,q,_ in requests};keys=set().union(*[set(x) for x in recipes.values()]);diff={k:{a:r.get(k) for a,r in recipes.items()} for k in keys if len({json.dumps(r.get(k),sort_keys=True) for r in recipes.values()})>1}
assert set(diff)=={'lr','d_lr_mult','reg_coeff','generator_history_mode'}
out={'status':'AUDIT_COMPLETE_NO_REPAIR_SURVIVOR','scope':'research_diagnostic_only','source_verification':source_check,'shared_request_fields_equal':shared,'comparisons':comparisons,'recipe_differences':diff,'attempt_count':len(rows),'retry_count':0,'seed':0,'declared_campaign_budget_seconds':1200,'current_outstanding_reservations_seconds':0,'actual_execution_seconds':sum(r['cost']['execution_seconds'] for r in rows),'all_full_original_updates':True,'independently_recomputed_grades':True,'repair_survivors':[],'rows':sorted(rows,key=lambda r:('PH'.index(r['arm']),r['task']))}
(BASE/'repairs/independent-result-audit.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
for r in out['rows']:print(r['arm'],r['task'],r['gate'],'suffix',r['passing_suffix'],'cost',r['cost']['execution_seconds'],'history',r['history'])

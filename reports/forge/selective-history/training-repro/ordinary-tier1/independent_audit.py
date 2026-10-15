"""Audit saved ordinary receipts only. No model execution or optimizer updates."""
from pathlib import Path
import json,hashlib,math,sys
BASE=Path('/ml2/hypergan/combination-training-20261015')
CAMPAIGN='bcap-current-baseline-ordinary-tier1-20261015-v1'
ATTEMPTS=BASE/'queue'/CAMPAIGN
OUT=BASE/'ordinary-tier1'
def read(p):return json.loads(p.read_text())
def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
INV=read(BASE/'ordinary-drafts/task-and-gate-inventory.json');EXPECTED={t['task']:t for t in INV['tasks']}
def pass_bounds(x,thresholds):
 for k,op,b in thresholds:
  v=x.get(k)
  if type(v) not in [float,int] or not math.isfinite(v):return False
  if not {'>=':v>=b,'<=':v<=b,'==':v==b}[op]:return False
 return True
rows=[];first=None
for d in sorted(ATTEMPTS.iterdir()):
 if not d.is_dir():continue
 w=read(d/'request.json');q=w['request'];task=w['job']['task_id'];assert task in EXPECTED
 if first is None:
  first=q;sp=Path(q['source']['snapshot_path']);assert stable(q['source']['files'])==q['source']['digest'];assert all(sha(sp/n)==h for n,h in q['source']['files'].items())
  sys.path.insert(0,str(sp))
  from experiments.forge.views import grade_result
 if not (d/'terminal.json').exists() or not (d/'result.json').exists():
  rows.append({'task':task,'attempt':d.name,'status':'RUNNING'});continue
 result=read(d/'result.json');terminal=read(d/'terminal.json')
 if 'task_results' not in result:rows.append({'task':task,'attempt':d.name,'status':'UNFINALIZED'});continue
 assert q['candidate']['id']=='recipe-prior-refactor-bcap-pure-v1';assert w['job']['science'].get('evidence_use') is None
 assert q['protocol']['seed']==0 and q['through_tier']==1
 inv=EXPECTED[task];spec=q['tasks'][task];assert spec['execution']==inv['full_frozen_task_definition']['execution'];assert spec['evaluation']==inv['full_frozen_task_definition']['evaluation']
 rr=result['task_results'][0];original=read(d/'graded-result.json') if (d/'graded-result.json').exists() else terminal.get('grading',{})
 row={'task':task,'importance':inv['importance'],'attempt':d.name,'retry_of':result.get('retry_of'),'terminal_status':terminal['attempt_status'],'cost':rr.get('cost',{}),'receipt_hashes':{n:sha(d/n) for n in ['request.json','result.json','terminal.json']},'source_digest':q['source']['digest']}
 if (d/'raw-result.json').exists() and 'grades' in original:
  raw=read(d/'raw-result.json');assert original['raw_hash']==stable(raw);grade=grade_result(spec,raw);assert grade==original['grades'][task], (task,'independent frozen evaluator mismatch')
  evidence=raw['evidence'];kind=spec['evaluation']['kind'];row.update(status=grade['gate_status'],metrics=grade.get('metrics'),evaluator_result=grade.get('evaluator_result'),reasons=grade.get('reasons'),independent_frozen_regrade_exact=True,receipt_hashes={**row['receipt_hashes'],**{n:sha(d/n) for n in ['raw-result.json','graded-result.json']}})
  if kind=='transfer_sustained':
   curve=evidence['observations'];assert [x['step'] for x in curve]==inv['exact_steps'];flags=[pass_bounds(x,inv['thresholds']) for x in curve];suffix=0
   for ok in reversed(flags):
    if not ok:break
    suffix+=1
   manual='PASS' if suffix>=5 and pass_bounds(evidence['live'],inv['thresholds']) else 'FAIL';assert manual==row['status'];row['manual_passing_suffix']=suffix
  elif kind in ['gaussian_smoke','word_smoke']:
   curve=evidence['observations'];confirm=evidence['confirmations'];assert [x['step'] for x in curve]==inv['exact_steps'] and [x['step'] for x in confirm]==inv['exact_steps'];hits=[x['step'] for x,y in zip(curve,confirm) if pass_bounds(x,inv['thresholds']) and pass_bounds(y['metrics'],inv['thresholds'])];row['manual_confirmed_steps']=hits
   assert (bool(hits))==(row['status']=='PASS');assert evidence['completed_steps']==inv['steps']
  row['guards']=evidence.get('guards');row['provenance_completed_steps']=evidence.get('provenance_checkpoint',{}).get('completed_steps');row['completed_steps']=evidence.get('completed_steps');row['executed_updates']=evidence.get('executed_updates')
  artifacts=[]
  def verify_files(root,manifest):
   for name,meta in manifest['files'].items():
    f=Path(root)/name;assert sha(f)==meta['sha256'];artifacts.append({'path':str(f),'sha256':meta['sha256']})
  if evidence.get('provenance_checkpoint'):
   p=evidence['provenance_checkpoint'];verify_files(p['artifact_root'],p['artifact_manifest'])
  if evidence.get('artifact_root') and evidence.get('artifact_manifest'):verify_files(evidence['artifact_root'],evidence['artifact_manifest'])
  if evidence.get('saved_observer_outputs'):
   s=evidence['saved_observer_outputs'];f=d/s['path']
   if not f.exists():f=Path(evidence['artifact_root'])/s['path']
   assert sha(f)==s['sha256'];artifacts.append({'path':str(f),'sha256':s['sha256']})
  row['verified_artifacts']=artifacts
  prior_attempt={'two_pole':'2f3e6cc58af14ee1aea7302ab180e3fa','ring16_acquisition':'e79666f6b8b94b6d84600dbaea2cdaad'}.get(task)
  if prior_attempt:
   old=read(BASE/'queue/bcap-selective-history-pole-ring-v2'/prior_attempt/'raw-result.json');row['fresh_vs_diagnostic_baseline_curve_identical']=evidence['observations']==old['evidence']['observations'];row['fresh_vs_diagnostic_baseline_live_identical']=evidence['live']==old['evidence']['live'];row['diagnostic_used_for_qualification']=False
 else:row.update(status=rr.get('gate_status',rr.get('status','INCOMPLETE')),independent_frozen_regrade_exact=False,note='No complete independently graded raw result; retained as incomplete/error, never inferred PASS.')
 rows.append(row)
state=read(BASE/'queue/queue/state.json');by_attempt={r['attempt']:r for r in rows};selected=[];superseded=[]
for task in EXPECTED:
 jobdef=next(j for j in first['jobs'] if task in j.get('task_ids',[j['task_id']]));job=state['jobs'][jobdef['compatibility_key']];chosen=(job.get('result') or {}).get('attempt_id') or (job.get('worker') or {}).get('attempt') if job['status'] in ['terminal','running'] else None
 if chosen in by_attempt:selected.append(by_attempt[chosen])
 else:selected.append({'task':task,'importance':EXPECTED[task]['importance'],'status':'PENDING_NO_CURRENT_RESULT'})
for row in rows:
 if row.get('retry_of'):
  old=row['retry_of']['attempt_id'];oldpath=ATTEMPTS/old/'result.json';assert stable(read(oldpath))==row['retry_of']['result_hash'];assert read(ATTEMPTS/old/'request.json')['request']==read(ATTEMPTS/row['attempt']/'request.json')['request'];assert by_attempt[old]['terminal_status'] in ['error','timeout','cancelled'];superseded.append(old)
assert len(superseded)<=1,'Only one operational retry authorized'
finished=all(r['status'] not in ['RUNNING','UNFINALIZED','PENDING_NO_CURRENT_RESULT'] for r in selected);required=[r for r in selected if r.get('importance')=='required'];passed=sum(r['status']=='PASS' for r in required)
output={'status':'COMPLETE' if finished else 'PARTIAL_NO_QUALIFICATION_CLAIM','campaign':CAMPAIGN,'scope':'ordinary_qualification','candidate':'recipe-prior-refactor-bcap-pure-v1','required_passes':passed,'required_total':6,'fresh_tier1_qualified':finished and len(required)==6 and passed==6,'diagnostic_clock_separate':True,'declared_campaign_budget_seconds':2520,'actual_execution_seconds':sum(r.get('cost',{}).get('execution_seconds',0) for r in rows),'source_digest':first['source']['digest'] if first else None,'attempt_count':len(rows),'task_count':len(EXPECTED),'superseded_attempts_retained':superseded,'selected_current_attempts':[{k:r.get(k) for k in ['task','attempt','status']} for r in selected],'rows':rows}
(OUT/'independent-result-audit.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2))
for r in rows:print(r['task'],r['status'],r.get('cost',{}).get('execution_seconds'))

"""Summarize durable terminal ordinary receipts without running evaluations."""
import hashlib,json,time
from pathlib import Path
root=Path('/ml2/hypergan/particlegan-selective-history-20261015')
out=Path('/ml2/hypergan/combination-training-20261015'); stage=out/'ordinary-tier1'
manifest=json.loads((stage/'launch-manifest.json').read_text())
state=json.loads((out/'queue/queue/state.json').read_text())
campaign=manifest['campaign']; declared=manifest['roster'][0]['jobs']; rows=[]; current=[]
for definition in declared:
 key=definition['key']; job=state['jobs'][key]
 assert job['status']=='terminal', (definition['task'],job['status'])
 for item in job['attempts']:
  directory=Path(item['path']); aid=item['attempt_id']; result=json.loads((directory/'result.json').read_text())
  wrapper=json.loads((directory/'request.json').read_text()); request=wrapper['request']
  assert request['source']['digest']==manifest['source_digest'] and request['protocol']['seed']==0
  assert request['campaign_id']==campaign and request['through_tier']==1
  assert len(result['task_results'])==1
  row=result['task_results'][0]; durable=root/'reports/forge/attempts'/aid
  assert all((durable/n).is_file() for n in ('request.json','result.json','evidence.json'))
  assert (directory/'result.json').read_bytes()==(durable/'result.json').read_bytes()
  hashes={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ('request.json','raw-result.json','graded-result.json','terminal.json','result.json') if (directory/n).exists()}
  observation_steps=[]
  for line in (directory/'run.log').read_text().splitlines():
   try:event=json.loads(line)
   except ValueError:continue
   if event.get('event')=='observation':observation_steps.append(event.get('step'))
  selected=result['attempt_id']==job['result']['attempt_id']
  record={'candidate':request['candidate']['id'],'study':request['study']['id'],'request_id':wrapper['request_id'],'attempt_id':aid,'task':row['task_id'],'job_key':key,'gate_status':row['gate_status'],'raw_status':row['raw_status'],'reason':row['reason'],'wall_seconds':row['cost']['wall_seconds'],'device':row['cost']['device'],'metrics':row.get('metrics',{}),'convergence':row.get('convergence'),'current_selected_outcome':selected,'retry_of':result.get('retry_of'),'qualification_required':row['task_id']!='clockfree_audit_measurement_v1','external_artifact_directory':str(directory),'durable_receipt_directory':str(durable),'file_sha256':hashes,'observations_logged':len(observation_steps),'latest_logged_observation_step':observation_steps[-1] if observation_steps else None,'final_update_count_known':row['raw_status']=='completed','saved_tensor_files':[str(p.relative_to(directory)) for p in sorted(directory.rglob('*.pt'))]}
  rows.append(record)
  if selected:current.append(record)
assert len(current)==7 and len({r['task'] for r in current})==7
charges=[c for c in state['charges'] if c['owner']['campaign']==campaign]
assert {c['attempt_id'] for c in charges}=={r['attempt_id'] for r in rows}
actual=sum(c['seconds'] for c in charges); ledger=state['campaigns'][campaign]
assert abs(actual-ledger['spent_seconds'])<1e-8 and ledger['reserved_seconds']==0
assert actual<=ledger['definition']['budget_seconds']
required=[r for r in current if r['qualification_required']]
passed=sum(r['gate_status']=='PASS' for r in required); assert len(required)==6
qualified=passed==6
summary={'schema':'selective_history_ordinary_tier1_completion_v1','status':'COMPLETE','completed_at_unix':time.time(),'campaign':campaign,'candidate':manifest['roster'][0]['candidate'],'source_digest':manifest['source_digest'],'source_origin_commit':'071a8d5614caf632b11ead731a6bb5fdd8a2afc8','runtime':manifest['runtime'],'scientific_attempts':len(rows),'completed_attempts':sum(r['raw_status']=='completed' for r in rows),'terminal_attempts':len(rows),'scientific_retries':0,'operational_retries':len(rows)-7,'spent_seconds':actual,'campaign_cap_seconds':ledger['definition']['budget_seconds'],'remaining_seconds':ledger['definition']['budget_seconds']-actual,'reserved_seconds':ledger['reserved_seconds'],'qualification_input':True,'scientific_scope':'ordinary_tier1','required_passes':passed,'required_total':6,'six_required_verdict':'PASS' if qualified else ('INCOMPLETE' if any(r['gate_status']=='INCOMPLETE' for r in required) else 'FAIL'),'fresh_tier1_qualified':qualified,'clock_diagnostic_separate':True,'selected_outcomes':[{'task':r['task'],'attempt_id':r['attempt_id'],'gate_status':r['gate_status'],'raw_status':r['raw_status']} for r in current],'operational_retry_status':'EXHAUSTED' if len(rows)>7 else ('CONDITIONALLY_AUTHORIZED_PENDING_SCHEDULING_REPAIR' if any(r['raw_status']=='timeout' for r in current) else 'NOT_NEEDED'),'observer_environment':{'PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS':None,'meaning':'Observed original word child unset; unchanged for retry'},'jobs':sorted(rows,key=lambda r:(r['task'],r['attempt_id']))}
(stage/'completion.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('jobs','runtime','selected_outcomes')},indent=2))

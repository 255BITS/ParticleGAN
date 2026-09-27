"""Audit available terminal simple-probe artifacts; emit own archive queue only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
R=Path(__file__).resolve().parent;E=R.parent
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/20260927T032610Z/research-simple-eleven-wave1/20260927T032610Z-1996550/repo/reports/reviewed-probe-output'));a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('simple_runtime_audit',R/'runtime_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
index=E/'research-screen-queue/simple-probe-preparation/prepared-index.json';prepared=read(index)['rows']
completed=[];pending=[];errors=[]
for row in prepared:
 name=Path(row['directory']).name
 matches=[f.parent for f in a.source_root.glob('*/'+name+'/result.json') if (f.parent/'initialization-receipt.json').is_file() and (f.parent/'prepared-source-sha256.json').is_file()]
 assert len(matches)<=1,(name,matches)
 if not matches:pending.append(dict(candidate=row['candidate'],directory=name));continue
 source=matches[0];out=R/'completed'/(name+'-runtime-audit.json');out.parent.mkdir(exist_ok=True)
 try:
  if out.exists():
   result=read(out)
   assert result['audit_source_sha256']==sha(R/'runtime_audit.py')
   assert result['source']==str(source)
   for artifact_name,digest in result['artifacts'].items():assert sha(source/artifact_name)==digest,artifact_name
  else:
   result=audit.audit(source,row,index)
 except Exception as error:
  import traceback
  record=dict(candidate=row['candidate'],source=str(source),error=repr(error),traceback=traceback.format_exc())
  errors.append(record);(R/(name+'-audit-error.json')).write_text(json.dumps(record,indent=2)+'\n');continue
 if not out.exists():
  out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  out.with_suffix('.md').write_text('# '+row['candidate']+' runtime audit\n\nRetained-artifact audit PASS. Strict quality '+result['quality_status']+': '+str(result['passing_observations'])+'/24, first arrival '+str(result['first_arrival'])+', final streak '+str(result['passing_suffix'])+'; final '+str(result['final']['modes'])+'/8, HQ '+str(result['final']['hq'])+'.\n\nSource seal, own CPU proof, raw initial CUDA models/buffers/optimizer parameters, original construction randomness and all2400 rate actions/Adam calls match.\n\n'+'\n\n'.join(result['limits'])+'\n')
 archive_id='research-'+name+'-new-init'
 completed.append(dict(id=archive_id,candidate=row['candidate'],audit=str(out),audit_sha256=sha(out),source=str(source),quality_status=result['quality_status'],passing_observations=result['passing_observations'],passing_suffix=result['passing_suffix'],first_arrival=result['first_arrival'],final=result['final'],eligibility=result['tested_configuration_eligibility'],archive_argv=['python3',str(E/'archive_research_screen.py'),'--audit',str(out),'--id',archive_id,'--eligibility',result['tested_configuration_eligibility']]))
archived={r['id'] for r in read(E/'research-results.json')['results']} if (E/'research-results.json').exists() else set()
for row in completed:row['already_archived']=row['id'] in archived
report=dict(pending_archive=[row for row in completed if not row['already_archived']],status='AUDIT_ERRORS' if errors else 'AUDITED_AVAILABLE_TERMINALS',prepared_index_sha256=sha(index),auditor_sha256=sha(R/'runtime_audit.py'),source_root=str(a.source_root),completed=completed,pending=pending,errors=errors,shared_results_modified=False)
(R/'pending-archive-index.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],completed=len(completed),pending=len(pending),errors=len(errors),index=str(R/'pending-archive-index.json'))))
assert 'torch' not in sys.modules
if errors:raise SystemExit(1)

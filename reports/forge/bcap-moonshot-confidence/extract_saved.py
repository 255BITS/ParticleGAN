"""Saved-only certified confidence counters and scalar paired metrics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository', type=Path, required=True)
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--output', type=Path)
args=parser.parse_args()
ROOT=args.repository.resolve()
ARCHIVE=args.artifacts.resolve()
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('confidence_saved_publisher',ROOT/'reports/forge/bcap-develop-integration/publish.py')
pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
state=json.loads((ARCHIVE/'phase3-queue/queue/state.json').read_text())
progress=json.loads((ARCHIVE/'phase3-progress.json').read_text())
roles={v:k for k,v in progress['requests'].items()}
rows=[]
for job in state['jobs'].values():
 if job['status']!='terminal' or not job.get('result'):
  continue
 result=job['result'];attempt=pub.certified_attempt(ROOT,result['attempt_id'])
 assert result==attempt['result']
 role=next(roles[r] for r in job['subscribers'] if r in roles)
 for row in result['task_results']:
  saved, proof=pub.checkpoint(row)
  counters=[]
  def visit(value,path='state'):
   if isinstance(value,dict):
    if value.get('mode')=='block_mmd_v1' and isinstance(value.get('stats'),dict):
     stats=value['stats'];calls=stats['calls']
     counters.append(dict(path=path,stats=stats,mean_mobility=stats['mobility_sum']/calls if calls else None,
      zero_fraction=stats['zero_calls']/calls if calls else None,
      high_fraction=stats['high_calls']/calls if calls else None,
      mean_signal=stats['signal_sum']/calls if calls else None,
      mean_standard_error=stats['standard_error_sum']/calls if calls else None,
      mean_blocks=stats['blocks']/calls if calls else None,mean_rows=stats['rows']/calls if calls else None))
    for key,child in value.items():
     if key not in ('models','role_parameters','streams','initialization'):
      visit(child,path+'.'+str(key))
   elif isinstance(value,(tuple,list)):
    for i,child in enumerate(value): visit(child,path+'['+str(i)+']')
  if saved is not None:visit(saved)
  if counters:
   assert all(c['stats']==counters[0]['stats'] for c in counters), 'counter copies disagree'
   assert counters[0]['stats']['calls']==proof['completed_steps'], 'saved mobility clock differs'
  observed=row.get('evidence',{}).get('observations',[])
  terminal_components={k:v for k,v in (observed[-1] if observed else {}).items()
   if k.startswith('component_') and isinstance(v,list)}
  metrics={k:v for k,v in row.get('metrics',{}).items() if isinstance(v,(int,float,bool,str)) or v is None}
  grade=row.get('evaluator_result',{})
  failures=[x for x in grade.get('metrics',[]) if isinstance(x,dict) and x.get('status')!='PASS']
  rows.append(dict(role=role,task_id=row['task_id'],gate_status=row['gate_status'],attempt_id=result['attempt_id'],
   metrics=metrics,evaluator_summary={k:v for k,v in grade.items() if k not in ('metrics','metric_checks')},
   failed_final_bounds=failures,checkpoint=proof,confidence=counters,terminal_components=terminal_components,
   cost=row.get('cost',{}),reason=row.get('reason'),source_commit=attempt['request']['source']['origin_commit'],
   source_digest=attempt['request']['source']['digest']))
output=dict(schema_version=1,scope='saved_only_confidence_analysis',qualification_input=False,
 phase=progress['phase'],source_commit=progress['source_commit'],rows=rows,
 optimizer_updates_added=0,sampling_draws_added=0)
(args.output or ARCHIVE/'saved-analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
for row in rows:
 conf=row['confidence']
 summary=conf[0] if conf else {}
 print(row['role'],row['task_id'],row['gate_status'],'metrics='+json.dumps(row['metrics']),
 'gate='+json.dumps(row['evaluator_summary']),
 'mobility='+json.dumps({k:v for k,v in summary.items() if k!='stats'}),flush=True)

"""Last two authorized refinements of the slow-critic final-only cadence result."""
from copy import deepcopy
from pathlib import Path
import search

CARDS=[
 {'name':'slow_g_every2_budget5','overrides':{'g_every':2,'steps':6000},
  'resources':'changed update cadence + more outer steps (D6000/G3000)'},
 {'name':'slow_g_every4_budget6','overrides':{'g_every':4,'steps':7200},
  'resources':'changed update cadence + more outer steps (D7200/G1800)'},
]
out=Path('/tmp/pr36-stress-solvability')
report=search.initialize()
task=next(task for task in search.TASKS if task['name']=='stress_slow_critic')
search.write_json(out/'targeted_final_frozen.json',{'task':task,'cards':CARDS,'max_episodes':2,
  'reason':'g_every2 at3600 reached final PASS butonly2passing observations. Extend time or allow morecriticupdates perG, preserveDlearningrateperturbation. Stopatfirstsustainedwitness.'})
for card in CARDS:
 spec=deepcopy(task);spec.update(deepcopy(card['overrides']))
 assert spec['d_lr_mult']==.75 and spec['thresholds']==task['thresholds']
 print('START',len(report['rows'])+1,card['name'],card['resources'],flush=True)
 result=search.vector_tasks.run_episode(spec,search.vector_tasks.fixed_policy('cosine'),fixed=True)
 row={'card':card,'task':task['name'],'spec':spec,'resources':card['resources'],
      'result':result,'shortfall':search.shortfall(result,spec)}
 report['rows'].append(row);search.write_json(out/'results.json',report);search.render(report)
 print('DONE',card['name'],result.get('status'),'suffix',result.get('convergence',{}).get('passing_suffix'),
       'live',result.get('live'),'update_counts',result.get('update_counts'),'seconds',result.get('seconds'),flush=True)
 if result.get('error'):print(result['error'],flush=True)
 if result.get('status')=='PASS' and result.get('convergence',{}).get('confirmed_step') is not None:
  print('STOP sustained witness found',flush=True)
  break
print('COMPLETE targeted_final',len(report['rows']),flush=True)

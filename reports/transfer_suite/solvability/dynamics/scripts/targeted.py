"""Bounded, predeclared follow-up on the remaining slow-critic bottleneck."""
from copy import deepcopy
import json
from pathlib import Path
import search

CARDS=[
 {'name':'density4096_slow','overrides':{'particles':4096,'batch':256,'steps':6000,'lr':.0006,
   'betas':[0.,.999],'prior_reg':1.,'reg_coeff':1.,'reg_kappa':1.,'prior_lr_mult':10.},
  'resources':'more particles + larger batch + more updates'},
 {'name':'slow_prior_lr3','overrides':{'prior_lr_mult':3.},'resources':'same architecture/data/update budget'},
 {'name':'slow_prior_lr3_budget3','overrides':{'prior_lr_mult':3.,'steps':3600},'resources':'more updates'},
 {'name':'slow_g_every2_budget3','overrides':{'g_every':2,'steps':3600},
  'resources':'changed update cadence + more outer steps (D3600/G1800)'},
]
out=Path('/tmp/pr36-stress-solvability')
report=search.initialize()
task=next(task for task in search.TASKS if task['name']=='stress_slow_critic')
search.write_json(out/'targeted_frozen.json',{'task':task,'cards':CARDS,'max_episodes':4,
  'reason':'Only slow-critic remains without a sustained training witness among the practical ranking families.'})
for card in CARDS:
 if any(row['card']['name']==card['name'] for row in report['rows']):continue
 spec=deepcopy(task);spec.update(deepcopy(card['overrides']))
 assert spec['d_lr_mult']==.75 and spec['thresholds']==task['thresholds']
 for key in ('means','covariances','masses','hidden','layers','d_hidden','d_layers','fourier'):
  assert spec[key]==task[key]
 print('START',len(report['rows'])+1,card['name'],'stress_slow_critic',card['resources'],flush=True)
 result=search.vector_tasks.run_episode(spec,search.vector_tasks.fixed_policy('cosine'),fixed=True)
 row={'card':card,'task':task['name'],'spec':spec,'resources':card['resources'],
      'result':result,'shortfall':search.shortfall(result,spec)}
 report['rows'].append(row);search.write_json(out/'results.json',report);search.render(report)
 print('DONE',card['name'],result.get('status'),'suffix',result.get('convergence',{}).get('passing_suffix'),
       'live',result.get('live'),'update_counts',result.get('update_counts'),'seconds',result.get('seconds'),flush=True)
 if result.get('error'):print(result['error'],flush=True)
print('COMPLETE targeted',len(report['rows']),flush=True)

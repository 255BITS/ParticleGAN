import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, '/ml2/hypergan/ParticleGAN-stress-solvability')
import torch
from benchmarks.transfer_suite import stress_tasks, vector_tasks
from benchmarks.locked_shared.baseline import write_json

torch.set_num_threads(1)
OUT=Path('/tmp/pr36-stress-solvability')
TASKS=[deepcopy(t) for t in stress_tasks.TASKS if t['tier']=='ranking']+[deepcopy(stress_tasks.RESERVED_TASKS[0])]
CARDS=[
 {'name':'baseline'},
 {'name':'prior_reg_zero','prior_reg':0.},
 {'name':'prior_reg_0p2','prior_reg':.2},
 {'name':'prior_reg_1','prior_reg':1.},
 {'name':'prior_reg_3','prior_reg':3.},
 {'name':'prior_lr_1','prior_lr_mult':1.},
 {'name':'prior_lr_3','prior_lr_mult':3.},
 {'name':'prior_lr_30','prior_lr_mult':30.},
 {'name':'prior_reg1_lr1','prior_reg':1.,'prior_lr_mult':1.},
 {'name':'prior_reg1_lr3','prior_reg':1.,'prior_lr_mult':3.},
 {'name':'beta2_0p99','betas':[0.,.99]},
 {'name':'beta2_0p9','betas':[0.,.9]},
 {'name':'cap1_k1','cap_coeff':1.,'cap_kappa':1.},
 {'name':'cap2_k1','cap_coeff':2.,'cap_kappa':1.},
 {'name':'cap5_k1p25','cap_coeff':5.,'cap_kappa':1.25},
 {'name':'lr1p5','lr_multiplier':1.5},
 {'name':'fourier3','fourier':3},
 {'name':'fourier4','fourier':4},
 {'name':'particles1024','particles':1024},
 {'name':'particles4096','particles':4096},
 {'name':'fourier3_particles1024','fourier':3,'particles':1024},
 {'name':'budget2','steps_multiplier':2},
 {'name':'budget3','steps_multiplier':3},
 {'name':'stockish_extended','fourier':3,'width_multiplier':1.5,'layers_add':1,'particles':1024,
  'lr_multiplier':.6,'cap_coeff':1.,'cap_kappa':1.,'prior_reg':1.,'steps_multiplier':3},
]

def configure(original,card):
 spec=deepcopy(original)
 for key in ('prior_reg','prior_lr_mult','betas','fourier','particles'):
  if key in card:spec[key]=deepcopy(card[key])
 if spec['reg_arm']=='b_cap':
  spec['reg_coeff']=card.get('cap_coeff',spec['reg_coeff'])
  spec['reg_kappa']=card.get('cap_kappa',spec['reg_kappa'])
 spec['lr']*=card.get('lr_multiplier',1.)
 spec['steps']*=card.get('steps_multiplier',1)
 for key in ('hidden','d_hidden'):
  spec[key]=round(spec[key]*card.get('width_multiplier',1.))
 for key in ('layers','d_layers'):
  spec[key]+=card.get('layers_add',0)
 assert spec['thresholds']==original['thresholds']
 for key in ('means','covariances','masses','batch','d_lr_mult','d_every','g_every'):
  assert spec.get(key)==original.get(key),(card['name'],key)
 return spec

def classify(card):
 extra=[]
 if card.get('steps_multiplier',1)!=1:extra.append('more updates')
 if any(key in card for key in ('fourier','particles','width_multiplier','layers_add')):extra.append('changed capacity')
 return ' + '.join(extra) if extra else 'same architecture/data/update budget'

def shortfall(result,spec):
 values=result.get('live',{})
 return sum(max(0., (bound-values.get(key,-1e6)) if op=='>=' else (values.get(key,1e6)-bound))/max(abs(bound),.01)
            for key,op,bound in spec['thresholds'])

def render(report):
 rows=sorted(report['rows'],key=lambda r:(-int(r['result'].get('convergence',{}).get('confirmed_step') is not None),
              -int(r['result'].get('status')=='PASS'),r['shortfall'],r['card']['name'],r['task']))
 lines=['# Stress solvability search','','All attempts: fixed seed0, original thresholds,24 live checks, final passing suffix>=5. EMA separate. Seen cadence is development evidence; this search is not fresh transfer.','',
 '| Card | Task | Resources | Final | Stable suffix | SW1 | Mass TV | HQ | Cov error | Min eigen | Seconds |',
 '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 for row in rows:
  r=row['result'];m=r.get('live',{});f=lambda k:f"{m[k]:.4f}" if k in m else 'ERROR'
  lines.append(f"| {row['card']['name']} | {row['task']} | {row['resources']} | {r.get('status')} | {r.get('convergence',{}).get('passing_suffix',0)}/24 | {f('sw1_normalized')} | {f('mass_tv')} | {f('hq')} | {f('component_covariance_error')} | {f('component_min_eigen_ratio')} | {r.get('seconds',0):.2f} |")
 (OUT/'README.md').write_text('\n'.join(lines)+'\n')

def initialize():
 if (OUT/'results.json').exists():return json.loads((OUT/'results.json').read_text())
 fp=vector_tasks.fingerprint()
 root=Path('/ml2/hypergan/ParticleGAN-stress-solvability')
 for relative in ('benchmarks/transfer_suite/stress_tasks.py',):fp['source_sha256'][relative]=hashlib.sha256((root/relative).read_bytes()).hexdigest()
 for relative,expected in fp['source_sha256'].items():
  source=root/relative;target=OUT/'source_snapshot'/relative
  assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 report={'protocol':fp,'source_commit':'afe615264221eda47c5d1b7fd2cf4082552e30e9','original_tasks':TASKS,
         'screen_cards':CARDS,'rows':[], 'seen_cadence_is_development':True,
         'target_positive_controls':{}}
 for spec in TASKS:
  target=vector_tasks.sample_target(spec,4096,torch.Generator().manual_seed(0),spec['steps'])
  metrics=vector_tasks.score_samples(target,spec,spec['steps'])
  report['target_positive_controls'][spec['name']]={'metrics':metrics,'pass':vector_tasks.passes(metrics,spec['thresholds']),
    'meaning':'Target distribution scoring sanity only, not a trained GAN witness.'}
 write_json(OUT/'results.json',report);write_json(OUT/'frozen_screen.json',{'cards':CARDS,'tasks':TASKS,'protocol':fp})
 return report

def run(report,card,task):
 if any(row['card']==card and row['task']==task['name'] for row in report['rows']):return
 spec=configure(task,card)
 print('START',len(report['rows'])+1,card['name'],task['name'],classify(card),flush=True)
 result=vector_tasks.run_episode(spec,vector_tasks.fixed_policy('cosine'),fixed=True,allow_reserved=spec['split']=='reserved')
 row={'card':card,'task':task['name'],'spec':spec,'resources':classify(card),'result':result,'shortfall':shortfall(result,spec)}
 report['rows'].append(row);write_json(OUT/'results.json',report);render(report)
 m=result.get('live',{});c=result.get('convergence',{})
 print('DONE',card['name'],task['name'],result.get('status'),'suffix',c.get('passing_suffix'),'metrics',
       {key:m.get(key) for key,_,_ in spec['thresholds']},'seconds',result.get('seconds'),flush=True)
 if result.get('error'):print(result['error'],flush=True)

if __name__=='__main__':
 args=argparse.ArgumentParser();args.add_argument('--phase',choices=['screen','cross'],default='screen');args.add_argument('--cards')
 options=args.parse_args();report=initialize()
 if options.phase=='screen':
  for card in CARDS:run(report,card,TASKS[0])
 else:
  cards=json.loads(Path(options.cards).read_text())
  for card in cards:
   for task in TASKS:run(report,card,task)
 print('COMPLETE',options.phase,'total_episodes',len(report['rows']),flush=True)

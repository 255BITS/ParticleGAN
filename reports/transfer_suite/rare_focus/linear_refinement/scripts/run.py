"""Serial pointwise raw/quadratic-skip architecture experiment; fixed original recipe."""
import argparse
from copy import deepcopy
import gzip,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-recipe')
import torch
from benchmarks.transfer_suite import suite,vector_tasks
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.solvability_search import write,render
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES,constructor

ROOT=Path('/tmp/pr36-valid-linear-final')
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['screen','cross'],required=True);parser.add_argument('--names',nargs='*');args=parser.parse_args()
all_names=[a['name'] for a in ARCHITECTURES]
selected=args.names or all_names
assert len(selected)==len(set(selected)) and set(selected)<=set(all_names)
assert args.phase!='screen' or selected==all_names
cards=[dict(name=name,architecture=next(deepcopy(a) for a in ARCHITECTURES if a['name']==name)) for name in selected]
tasks=(['vector_unequal_mass'] if args.phase=='screen' else ['vector_two_broad','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral'])
originals={s['name']:s for s in suite.manifest()['tasks']}
checks={c['name']:c for c in json.loads((ROOT/'architecture_checks.json').read_text())['cards']}
for card in cards:
 card['parameter_count']=checks[card['name']]['parameters']
 card['overrides']=dict(d_hidden=card['architecture']['hidden'],d_layers=2)
output=ROOT/args.phase;output.mkdir(exist_ok=False);(output/'episodes').mkdir()
plan=dict(candidates=cards,tasks=tasks,formulation='original Rp logistic b_cap3/kappa1.25 prior_reg.05 no particle L2',recipe='original Adam(0,.99),G LR .001,D LR .0015,prior LR .01,cosine,1:1 updates',scope='research discriminator architecture only',seed=0, cross_rule='Only sustained rare winners receive the other five data toys. This is the final four-card refinement; no further cards.')
write(output/'plan.json',plan)
protocol=suite.snapshot(output);protocol['experiment_source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('run.py','check_architectures.py')};write(output/'protocol.json',protocol)
torch.set_num_threads(1)
records=[]
for card in cards:
 for task in tasks:
  original=deepcopy(originals[task]);spec=original|card['overrides']|{'research_discriminator':deepcopy(card['architecture'])}
  assert original['betas']==[0.,.99] and original['particles']==256 and original['batch']==128
  assert original['lr']==.001 and original['d_lr_mult']==1.5 and original['prior_lr_mult']==10.
  assert original['reg_arm']=='b_cap' and original['reg_coeff']==3. and original['reg_kappa']==1.25 and original['prior_reg']==.05
  suite.verify_source(protocol)
  print(json.dumps({'event':'START','card':card['name'],'task':task,'steps':spec['steps']}),flush=True)
  saved=vector_tasks.SimpleMLPDiscriminator
  vector_tasks.SimpleMLPDiscriminator=constructor(card['architecture'])
  try:result=vector_tasks.run_episode(spec,vector_tasks.fixed_policy('cosine'),fixed=True)
  finally:vector_tasks.SimpleMLPDiscriminator=saved
  verdict=test_verdict(spec,result)
  record=dict(candidate=card,original_spec=original,spec=spec,policy=vector_tasks.fixed_policy('cosine'),verdict=verdict,seconds=result['seconds'])
  payload=record|{'result':result,'source_sha256':protocol['source_sha256'],'experiment_source_sha256':protocol['experiment_source_sha256']}
  raw=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode();relative=f"episodes/{card['name']}__{task}.json.gz"
  (output/relative).write_bytes(gzip.compress(raw,mtime=0))
  record.update(artifact=relative,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),live=result.get('live'),ema=result.get('ema'))
  records.append(record);write(output/'index.json',{'records':records});render(output,records)
  print(json.dumps({'event':'DONE','card':card['name'],'task':task,'status':verdict['status'],'suffix':verdict.get('convergence',{}).get('passing_suffix'),'live':result.get('live'),'seconds':result['seconds'],'error':result.get('error')}),flush=True)
suite.verify_source(protocol)
print('COMPLETE',args.phase,len(records),flush=True)

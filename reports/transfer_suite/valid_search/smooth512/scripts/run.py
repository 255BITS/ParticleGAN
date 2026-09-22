"""Targeted fixed-recipe 512-particle smooth-D witness check."""
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
from benchmarks.transfer_suite.smooth_critic_research import constructor

ROOT=Path('/tmp/pr36-valid-smooth512')
CARDS=[dict(name=f'softplus5_d{width}x2_f2_p512',architecture=dict(name='axis_softplus5',features='axis',activation='softplus',beta=5.,hidden=width,layers=2,fourier=2),overrides=dict(particles=512,d_hidden=width,d_layers=2)) for width in (64,96)]
TASKS=['vector_two_broad','vector_unequal_mass','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral']
parser=argparse.ArgumentParser();parser.add_argument('--width',type=int,choices=[64,96],required=True);parser.add_argument('--phase',choices=['rare','cross'],required=True);args=parser.parse_args()
card=deepcopy(CARDS[[64,96].index(args.width)]);tasks=['vector_unequal_mass'] if args.phase=='rare' else [n for n in TASKS if n!='vector_unequal_mass']
originals={s['name']:s for s in suite.manifest()['tasks']}
declaration=dict(cards=CARDS,screen_tasks=['vector_unequal_mass'],cross_tasks=[n for n in TASKS if n!='vector_unequal_mass'],selection='Run both rare cases; stop if neither sustains. If either sustains, complete all six for the passing D with longer final suffix, then smaller final normalized shortfall, then smaller width.',resource_profile=dict(particles=512,batch=128,hidden=64,layers=2,z_dim=4),recipe='Original beta99/Rp logistic/b_cap3/kappa1.25/prior_reg.05,cosine,G/D/prior LR .001/.0015/.01,1:1 updates; original outer budgets',seed=0)
if not (ROOT/'declaration.json').exists():write(ROOT/'declaration.json',declaration)
else:assert json.loads((ROOT/'declaration.json').read_text())==declaration
output=ROOT/f'{args.phase}{args.width}';output.mkdir(exist_ok=False);(output/'episodes').mkdir();write(output/'plan.json',dict(candidate=card,tasks=tasks))
protocol=suite.snapshot(output);protocol['experiment_source_sha256']={'run.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};protocol['source_commit']='981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45';write(output/'protocol.json',protocol)
torch.set_num_threads(1);rows=[]
for task in tasks:
 original=deepcopy(originals[task]);spec=original|card['overrides']|{'research_discriminator':deepcopy(card['architecture'])}
 assert spec['betas']==[0.,.99] and spec['batch']==128 and spec['particles']==512 and spec['hidden']==64 and spec['layers']==2
 assert spec['lr']==.001 and spec['d_lr_mult']==1.5 and spec['prior_lr_mult']==10.
 assert spec['reg_arm']=='b_cap' and spec['reg_coeff']==3. and spec['reg_kappa']==1.25 and spec['prior_reg']==.05
 suite.verify_source(protocol);print(json.dumps(dict(event='START',card=card['name'],task=task,steps=spec['steps'])),flush=True)
 saved=vector_tasks.SimpleMLPDiscriminator;vector_tasks.SimpleMLPDiscriminator=constructor(card['architecture'])
 try:result=vector_tasks.run_episode(spec,vector_tasks.fixed_policy('cosine'),fixed=True)
 finally:vector_tasks.SimpleMLPDiscriminator=saved
 verdict=test_verdict(spec,result);record=dict(candidate=card,original_spec=original,spec=spec,policy=vector_tasks.fixed_policy('cosine'),resource_profile=declaration['resource_profile'],verdict=verdict,seconds=result['seconds'])
 payload=record|dict(result=result,source_sha256=protocol['source_sha256'],experiment_source_sha256=protocol['experiment_source_sha256'])
 raw=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode();file=f"episodes/{card['name']}__{task}.json.gz";(output/file).write_bytes(gzip.compress(raw,mtime=0))
 record.update(artifact=file,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),live=result.get('live'),ema=result.get('ema'));rows.append(record);write(output/'index.json',{'records':rows});render(output,rows)
 print(json.dumps(dict(event='DONE',card=card['name'],task=task,status=verdict['status'],suffix=verdict.get('convergence',{}).get('passing_suffix'),live=result.get('live'),seconds=result['seconds'],error=result.get('error'))),flush=True)
suite.verify_source(protocol)

"""Adaptive follow-up to the observed beta5 near miss; original recipe frozen."""
from copy import deepcopy
from datetime import datetime, timezone
import gzip, hashlib, importlib.util, json, math
from pathlib import Path
import sys
sys.path.insert(0, '/ml2/hypergan/ParticleGAN-pr36-valid-d')
import torch
from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.solvability_search import write, render
from benchmarks.transfer_suite.smooth_critic_research import constructor
from particlegan import GradientPenalty
ROOT=Path('/tmp/pr36-valid-softplus-refinement')
ORIGIN=Path('/ml2/hypergan/ParticleGAN-pr36-valid-recipe/benchmarks/transfer_suite/smooth_critic_research.py')
LOCAL=Path('/ml2/hypergan/ParticleGAN-pr36-valid-d/benchmarks/transfer_suite/smooth_critic_research.py')
HARD=['vector_unequal_mass','vector_unequal_width','vector_overlap']
REST=['vector_two_broad','vector_anisotropic','vector_spiral']
CARDS=[dict(name=f'softplus{beta}_d{width}_l{depth}_f2', architecture=dict(features='axis',activation='softplus',beta=float(beta)),overrides=dict(d_hidden=width,d_layers=depth,fourier=2)) for beta,width,depth in [(3,64,2),(10,64,2),(5,96,2),(5,128,2),(5,64,3),(5,128,3)]]
for card in CARDS: card['architecture']['name']=card['name']
PLAN=dict(created_utc=datetime.now(timezone.utc).isoformat(), stage='adaptive refinement; frozen six-card screen before new results', seed=0, candidates=CARDS, screen_tasks=HARD, validation_tasks=REST, selection='Select two by rare-mass sustained PASS first, then total hard sustained passes, then lower mean final normalized bound shortfall, then confirmation fraction, then name. Complete other three data cases for selected cards; never change their settings per case.', trigger='Observed original axis Softplus(beta5) D64x2/F2 passes unequal_width; unequal_mass narrowly fails TV .166699>.15 and SW1 .181933>.18, with all other final bounds passing.', source_origin=str(ORIGIN), source_origin_sha256=hashlib.sha256(ORIGIN.read_bytes()).hexdigest(), fixed='Original G, Rp logistic b_cap3/kappa1.25, prior_reg.05, no particle L2, Adam(0,.99), LR.001, Dmult1.5, prior_mult10, cosine, 256 particles, batch128, all original per-task budgets (spiral1600; others1200). Only D activation beta/width/depth vary.', gates='Unchanged original live thresholds; complete24 observations and at least5 final passing. EMA separate. No diagnostic/stress or extra seeds.')
assert not (ROOT/'plan.json').exists()
write(ROOT/'plan.json',PLAN)
assert LOCAL.read_bytes()==ORIGIN.read_bytes()
# Same seed checks only: copy parity and finite second-order cap backprop.
module_spec=importlib.util.spec_from_file_location('original_smooth_critic',ORIGIN)
origin_module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(origin_module)
torch.set_num_threads(1)
checks=[]
for card in CARDS:
 kwargs=dict(in_dim=2,hidden_dim=card['overrides']['d_hidden'],n_hidden=card['overrides']['d_layers'],fourier=2)
 torch.manual_seed(0);model=constructor(card['architecture'])(**kwargs)
 torch.manual_seed(0);replica=origin_module.constructor(card['architecture'])(**kwargs)
 assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),replica.state_dict().values()))
 gen=torch.Generator().manual_seed(0);real=torch.randn(16,2,generator=gen);fake=torch.randn(16,2,generator=gen)
 outputs=[]
 for network in (model,replica):
  x=fake.clone().requires_grad_();out=network(x)
  penalty=GradientPenalty('b_cap',coeff=3.,kappa=1.25)(network,real,fake,step=1,generator=torch.Generator().manual_seed(0))
  loss=out.mean()+penalty;loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
  outputs.append((out.detach(),penalty.detach(),x.grad.clone(),[p.grad.clone() for p in network.parameters()]))
 assert all(torch.equal(a,b) for a,b in zip(outputs[0][:3],outputs[1][:3]))
 assert all(torch.equal(a,b) for a,b in zip(outputs[0][3],outputs[1][3]))
 checks.append(dict(candidate=card['name'],discriminator_parameters=sum(p.numel() for p in model.parameters()),source_copy_exact=True,state_output_input_gradient_penalty_parameter_gradient_exact=True,finite_backward=True))
write(ROOT/'architecture_checks.json',checks)
protocol=suite.snapshot(ROOT);protocol['experiment_source_sha256']={'run.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};write(ROOT/'protocol.json',protocol)
originals={s['name']:s for s in suite.manifest()['tasks']}
write(ROOT/'task_specs.json',[originals[n] for n in HARD+REST])
(ROOT/'episodes').mkdir();records=[]
def episode(card,task,phase):
 original=deepcopy(originals[task]);spec=original|deepcopy(card['overrides'])|{'research_discriminator':deepcopy(card['architecture'])}
 assert set(k for k in original if spec[k]!=original[k]) <= {'d_hidden','d_layers','fourier'}
 assert original['betas']==[0.,.99] and original['particles']==256 and original['batch']==128
 assert original['lr']==.001 and original['d_lr_mult']==1.5 and original['prior_lr_mult']==10.
 assert original['reg_arm']=='b_cap' and original['reg_coeff']==3. and original['reg_kappa']==1.25 and original['prior_reg']==.05
 suite.verify_source(protocol)
 print(json.dumps(dict(event='START',candidate=card['name'],task=task,phase=phase,steps=spec['steps'])),flush=True)
 saved=vector_tasks.SimpleMLPDiscriminator;vector_tasks.SimpleMLPDiscriminator=constructor(card['architecture'])
 try: result=vector_tasks.run_episode(spec,vector_tasks.fixed_policy('cosine'),fixed=True)
 finally: vector_tasks.SimpleMLPDiscriminator=saved
 verdict=test_verdict(spec,result)
 record=dict(candidate=card,original_spec=original,spec=spec,policy=vector_tasks.fixed_policy('cosine'),fixed=True,phase=phase,verdict=verdict,seconds=result['seconds'])
 payload=record|dict(result=result,source_sha256=protocol['source_sha256'],experiment_source_sha256=protocol['experiment_source_sha256'])
 raw=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode();relative=f"episodes/{card['name']}__{task}.json.gz"
 (ROOT/relative).write_bytes(gzip.compress(raw,mtime=0))
 record.update(artifact=relative,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),live=result.get('live'),ema=result.get('ema'))
 records.append(record);write(ROOT/'index.json',dict(records=records));render(ROOT,records)
 print(json.dumps(dict(event='DONE',candidate=card['name'],task=task,phase=phase,status=verdict['status'],suffix=verdict.get('convergence',{}).get('passing_suffix'),shortfall=verdict['shortfall'],live=result.get('live'),seconds=result['seconds'],error=result.get('error'))),flush=True)
for card in CARDS:
 for task in HARD: episode(card,task,'screen')
def rank(card):
 rows=[r for r in records if r['candidate']['name']==card['name']]
 rare=next(r for r in rows if r['spec']['name']=='vector_unequal_mass')
 return (not rare['verdict']['passed'],-sum(r['verdict']['passed'] for r in rows),sum(r['verdict']['shortfall'] for r in rows)/len(rows),sum(r['verdict']['confirmation_fraction'] for r in rows)/len(rows),card['name'])
ranked=sorted(CARDS,key=rank);selected=ranked[:2]
write(ROOT/'selection.json',dict(rule=PLAN['selection'],ranking=[dict(candidate=c['name'],key=rank(c)) for c in ranked],selected=[c['name'] for c in selected]))
print(json.dumps(dict(event='SELECTED',candidates=[c['name'] for c in selected])),flush=True)
for card in selected:
 for task in REST: episode(card,task,'full_six_completion')
suite.verify_source(protocol)
print('COMPLETE',len(records),flush=True)

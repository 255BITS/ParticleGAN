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
from benchmarks.transfer_suite.spectral_critic_research import constructor
from benchmarks.transfer_suite.smooth_critic_research import constructor as smooth_constructor
from particlegan import GradientPenalty
ROOT=Path('/tmp/pr36-rare-spectral-round3')
ORIGIN=Path('/ml2/hypergan/ParticleGAN-pr36-valid-recipe/benchmarks/transfer_suite/smooth_critic_research.py')
LOCAL=Path('/ml2/hypergan/ParticleGAN-pr36-valid-d/benchmarks/transfer_suite/smooth_critic_research.py')
HARD=['vector_unequal_mass']
REST=['vector_unequal_width','vector_overlap','vector_two_broad','vector_anisotropic','vector_spiral']
CARD_VALUES=[('b5_raw2',5,2.,1.,[1.,1.]),('b5_raw4',5,4.,1.,[1.,1.]),
 ('b5_fourier025',5,1.,.25,[1.,1.]),('b5_fourier05',5,1.,.5,[1.,1.]),
 ('b5_raw2_fourier05',5,2.,.5,[1.,1.]),('b5_harmonic2_025',5,1.,1.,[1.,.25]),
 ('b5_harmonic2_05',5,1.,1.,[1.,.5]),('b6_raw2',6,2.,1.,[1.,1.]),
 ('b6_fourier05',6,1.,.5,[1.,1.]),('b6_harmonic2_025',6,1.,1.,[1.,.25])]
CARDS=[dict(name=name,architecture=dict(name=name,features='axis',activation='softplus',beta=float(beta),raw_gain=raw_gain,fourier_gain=fourier_gain,harmonic_gains=harmonic_gains),overrides=dict(d_hidden=96,d_layers=2,fourier=2)) for name,beta,raw_gain,fourier_gain,harmonic_gains in CARD_VALUES]
PLAN=dict(created_utc=datetime.now(timezone.utc).isoformat(), stage='third adaptive refinement: fixed spectral feature balance; ten-card rare-only screen frozen before results', seed=0, candidates=CARDS, screen_tasks=HARD, validation_tasks=REST, selection='Run all ten rare-mass cards; every sustained rare-mass winner completes the other five valid data cases, using the exact same architecture and recipe across cases. No further cards or budget extensions. Rank rare failures for reporting by final normalized bound shortfall, then final passing suffix, then name; do not promote final-only passes.', trigger='Forensic replay of D96x2/F2 Softplus5 found genuine rare-component minor-axis contraction under expected Rp critic gradient, with G locally full-rank. Test whether fixed input-feature amplitude balance changes periodic derivative dominance. This is a hypothesis, not a demonstrated causal remedy.', source_origin=str(ORIGIN), source_origin_sha256=hashlib.sha256(ORIGIN.read_bytes()).hexdigest(), fixed='Original G, Rp logistic b_cap3/kappa1.25, prior_reg.05, no particle L2, Adam(0,.99), LR.001, Dmult1.5, prior_mult10, cosine, 256 particles, batch128, all original per-task budgets (spiral1600; others1200). Only D fixed raw/Fourier/harmonic feature amplitudes vary around Softplus beta5/6. Width96/depth2/Fourier2 are fixed. All feature transforms remain native differentiable; cap gradients remain in original data coordinates.', gates='Unchanged original live thresholds; complete24 observations and at least5 final passing. EMA separate. No diagnostic/stress or extra seeds.')
assert not (ROOT/'plan.json').exists()
write(ROOT/'plan.json',PLAN)
assert LOCAL.read_bytes()==ORIGIN.read_bytes()
torch.set_num_threads(1)
checks=[]
# Neutral architecture must exactly reproduce the existing smooth critic, including derivatives.
for beta in (5.,6.):
 card=dict(name='neutral_check',features='axis',activation='softplus',beta=beta,raw_gain=1.,fourier_gain=1.,harmonic_gains=[1.,1.])
 torch.manual_seed(0);model=constructor(card)(2,96,2,2)
 torch.manual_seed(0);replica=smooth_constructor(card)(2,96,2,2)
 assert all(torch.equal(model.state_dict()[k],v) for k,v in replica.state_dict().items())
 gen=torch.Generator().manual_seed(0);real=torch.randn(16,2,generator=gen);fake=torch.randn(16,2,generator=gen)
 outputs=[]
 for network in (model,replica):
  x=fake.clone().requires_grad_();out=network(x)
  penalty=GradientPenalty('b_cap',coeff=3.,kappa=1.25)(network,real,fake,step=1,generator=torch.Generator().manual_seed(0))
  (out.mean()+penalty).backward()
  outputs.append((out.detach(),penalty.detach(),x.grad.clone(),[p.grad.clone() for p in network.parameters()]))
 assert all(torch.equal(a,b) for a,b in zip(outputs[0][:3],outputs[1][:3]))
 assert all(torch.equal(a,b) for a,b in zip(outputs[0][3],outputs[1][3]))
 checks.append(dict(name=f'neutral_beta{beta}',state_output_input_gradient_penalty_parameter_gradient_exact=True))
for card in CARDS:
 torch.manual_seed(0);model=constructor(card['architecture'])(2,96,2,2)
 gen=torch.Generator().manual_seed(0);real=torch.randn(16,2,generator=gen);fake=torch.randn(16,2,generator=gen,requires_grad=True)
 output=model(fake);first=torch.autograd.grad(output.sum(),fake,create_graph=True)[0]
 second=torch.autograd.grad(first.sum(),fake)[0]
 assert torch.isfinite(output).all() and torch.isfinite(first).all() and torch.isfinite(second).all()
 # Activate cap on a static test copy by increasing the output weight; this is NEVER used in training.
 with torch.no_grad():model.net[-1].weight.mul_(100.)
 penalty=GradientPenalty('b_cap',coeff=3.,kappa=1.25)(model,real,fake.detach(),step=1,generator=torch.Generator().manual_seed(0))
 assert torch.isfinite(penalty) and penalty.item()>0
 penalty.backward()
 gradients=[p.grad for p in model.parameters() if p.grad is not None]
 assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum()>0 for g in gradients)
 checks.append(dict(candidate=card['name'],discriminator_parameters=sum(p.numel() for p in model.parameters()),finite_native_input_hessian=True,active_b_cap_double_backprop=True,static_test_output_weight_multiplier=100.,static_test_penalty=penalty.item()))
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
selected=[card for card in CARDS if next(r for r in records if r['candidate']['name']==card['name'])['verdict']['passed']]
write(ROOT/'selection.json',dict(rule=PLAN['selection'],selected=[c['name'] for c in selected]))
print(json.dumps(dict(event='SELECTED',candidates=[c['name'] for c in selected])),flush=True)
for card in selected:
 for task in REST: episode(card,task,'full_six_completion')
suite.verify_source(protocol)
print('COMPLETE',len(records),flush=True)

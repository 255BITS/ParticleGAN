"""Frozen vector host resources and scoring, current public RP5 learner."""
from public_worker import ROOT,digest,rates
from pathlib import Path
import argparse,hashlib,json,math,os,time,zipfile
import torch
from particlegan import GANTrainer,get_recipe
from particlegan.training import input_noise_std,output_noise_std
from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.public_default_verification import vector_discriminator
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.toy100.device import apply_device_policy
from lib.toy_models import SimpleMLPGenerator
from static_sources import dependency_closure

O=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
plan=ROOT/'benchmarks/transfer_suite/plans/default_comparison.json';profile=ROOT/'reports/transfer_suite/unadjusted/leading_profile.json'
original=next(j['spec'] for j in json.loads(plan.read_text()) if j['spec']['name']==a.task)
assert original['runner']=='vector'
card=json.loads(profile.read_text())['discriminators'].get(a.task)
spec=dict(original)
if card:spec.update(d_hidden=card.get('width',card.get('hidden')),d_layers=card['layers'],fourier=card['fourier'],research_discriminator=card)
fixture_info=json.loads((O/'vector-fixtures.json').read_text())['fixtures'][a.task]
fixture_path=ROOT/fixture_info['local_path']
assert hashlib.sha256(fixture_path.read_bytes()).hexdigest()==fixture_info['sha256']
cfg=vector_tasks.resolve(spec);assert cfg['d_every']==cfg['g_every']==1
recipe=get_recipe(total_steps=None,continuous_precision='rp5',game_update='secant_resolvent',adam_eager_state=True,
                  num_particles=cfg['particles'],z_dim=cfg['z_dim'],batch_size=cfg['batch'])
# Host source pinning; unrelated C6 experiment paths remain provenance, not imports.
lock=json.loads((O/'vector-host-lock.json').read_text())
for name,want in lock['source_sha256'].items():
 if name.startswith('experiments/'):continue
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name
entries=[str(x.relative_to(ROOT)) for x in [*sorted((ROOT/'particlegan').glob('*.py')),Path(__file__),O/'public_worker.py',O/'static_sources.py',O/'source_helpers.py',O/'rp5.md',O/'vector-declaration.md',O/'vector-host-lock.json',O/'vector-support-copy.json',O/'vector-fixtures.json',O/'vector-backend-contract.md',O/'vector-fixture-correction.md',O/'vector-fixture-correction-v2.md',fixture_path,plan,profile]]
sources,external=dependency_closure(ROOT,entries)
a.output.mkdir(parents=True,exist_ok=False)
decl=dict(candidate='API-RP5',gate='broader_'+a.task,original_spec=original,spec=cfg,card=card,recipe=recipe.to_dict(),serial_backward=True,
 initialization='CPU prior/G/D construction, canonical CPU parameter fixture copied byte-exact before CUDA/GANTrainer; explicit global/prior seed0',fixture=fixture_info,
 streams=dict(data='CUDA seed0',latent='CUDA seed1',penalty='CUDA seed2',training_noise='API private CUDA seed5',generator_real='fresh data batch from same stream once per accepted update',evaluation='CUDA latent990,target991,projection992; fixed paired output seed2303; isolated'),
 runtime_scope='CPU initialization plus published CUDA data/evaluation backend; explicitly resolves supervisor scaffold ambiguity; no CPU reference score borrowed',
 evaluation_noise_difference='Declared isolated output seed2303 retained; historical K3P used global402. Any matched comparator must use this declared observation namespace too.',
 quality='unchanged protocol.test_verdict and complete24-observation/final5 requirement',
 source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},external_imports=external)
(a.output/'declaration.json').write_text(json.dumps(decl,indent=2)+'\n')
with zipfile.ZipFile(a.output/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sources.items():z.writestr(name,data)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(0)
prior=recipe.make_prior(init_std=.5,generator=torch.Generator(device='cpu').manual_seed(0))
g=SimpleMLPGenerator(cfg['z_dim'],cfg['hidden'],cfg['layers'],2);d=vector_discriminator(cfg,card)
assert all(p.device.type=='cpu' for m in (prior,g,d) for p in m.parameters())
fixture=torch.load(fixture_path,map_location='cpu',weights_only=False)
parameter_groups=[list(g.parameters())+list(prior.parameters()),list(d.parameters())]
with torch.no_grad():
 for params,values in zip(parameter_groups,fixture):
  assert len(params)==len(values)
  for parameter,value in zip(params,values):
   assert parameter.shape==value.shape
   parameter.copy_(value)
actual=[[{'shape':list(t.shape),'sha256':hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()} for t in group] for group in parameter_groups]
assert actual==fixture_info['initial_parameter_groups'], 'canonical fixture mismatch before training'
(a.output/'canonical-initialization.json').write_text(json.dumps({'all_parameter_hashes_match':True,'actual':actual,'fixture_sha256':fixture_info['sha256']},indent=2)+'\n')
initial_models={k:digest(m.state_dict()) for k,m in [('prior',prior),('G',g),('D',d)]}
g=g.cuda();d=d.cuda();prior=prior.cuda();runtime=apply_device_policy('cuda:0')
stream=torch.Generator(device='cuda:0').manual_seed(0)
t=GANTrainer(recipe,g,d,prior=prior,seed=0,serial_backward=True,
             latent_generator=torch.Generator(device='cuda:0').manual_seed(1),penalty_generator=torch.Generator(device='cuda:0').manual_seed(2),
             optimizer_options={'foreach':False,'fused':False})
(a.output/'runtime.json').write_text(json.dumps({**runtime,'torch':str(torch.__version__),'cuda':torch.version.cuda,'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}},indent=2)+'\n')
initial={'trainer':t.state_dict(),'data_rng':stream.get_state()};torch.save(initial,a.output/'initial-state.pt')
(a.output/'initial.json').write_text(json.dumps({'complete':digest(initial),'models_cpu':initial_models,'models_cuda':{k:digest(v) for k,v in t.state_dict()['models'].items()}},indent=2)+'\n')
expected={math.ceil(i*cfg['steps']/24) for i in range(1,25)};observations=[];started=time.monotonic()
@torch.no_grad()
def measure(ema=False):
 with torch.random.fork_rng(devices=[0]):
  model,prior=(t.ema_G,t.ema_prior) if ema else (t.G,t.prior)
  torch.manual_seed(402)
  latent=prior.sample(vector_tasks.EVAL_SAMPLES,generator=torch.Generator(device='cuda:0').manual_seed(990))[0]
  fake=t._generate(model,latent,output_noise_std(recipe,t.completed_steps),torch.Generator(device='cuda:0').manual_seed(2303))
  return vector_tasks.score_samples(fake,cfg,t.completed_steps)
try:
 with (a.output/'metrics.jsonl').open('w',buffering=1) as obs,(a.output/'learning-rates.jsonl').open('w',buffering=1) as lr:
  for step in range(1,cfg['steps']+1):
   real=vector_tasks.sample_target(cfg,cfg['batch'],stream,step)
   real_g=lambda:vector_tasks.sample_target(cfg,cfg['batch'],stream,step)
   t.step(real,generator_real=real_g,collect_stats=step in expected)
   lr.write(json.dumps({'step':step,**rates(t),'input_noise':input_noise_std(recipe,step-1),'output_noise':output_noise_std(recipe,step-1),'precision':t.precision.state,'game':t.game_stats})+'\n')
   if step in expected:
    before=digest([t.state_dict(),stream.get_state()]);point=dict(step=step,**measure(),ema=measure(True));assert digest([t.state_dict(),stream.get_state()])==before
    observations.append(point);obs.write(json.dumps(point)+'\n');print(json.dumps(point),flush=True)
 verdict=test_verdict(cfg,dict(live=observations[-1],observations=observations))
 assert verdict['convergence']['complete']
 # Existing final-five requirement, never endpoint-only success.
 status='PASS' if verdict['passed'] and verdict['convergence']['passing_suffix']>=5 else 'FAIL'
 metrics={'verdict':verdict,'final':observations[-1],'policy':t.precision.state,'updates':t.completed_steps}
 torch.save(dict(trainer=t.state_dict(),data_rng=stream.get_state()),a.output/'final-state.pt')
except Exception as e:status='ERROR';metrics={'error':repr(e)}
row=dict(candidate='API-RP5',gate='broader_'+a.task,status=status,seconds=time.monotonic()-started,metrics=metrics,artifact=str(a.output.resolve()))
(a.output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
with (ROOT.parent/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
(a.output/'artifact-sha256.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.iterdir() if x.is_file() and x.name!='artifact-sha256.json'},indent=2)+'\n')
print(json.dumps(row),flush=True)

"""Frozen small ring, actual RP5 API, preserved shared data/latent order."""
from public_worker import ROOT,digest,rates
from pathlib import Path
import argparse,hashlib,json,os,time,zipfile
import torch
from particlegan import GANTrainer,get_recipe
from particlegan.training import input_noise_std,output_noise_std
from benchmarks.locked_shared import mode_hold as host
from benchmarks.transfer_suite.protocol import test_verdict
from static_sources import dependency_closure

O=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
plan=ROOT/'benchmarks/transfer_suite/plans/default_comparison.json'
spec=next(j['spec'] for j in json.loads(plan.read_text()) if j['spec']['name']=='mode_hold')
recipe=get_recipe(total_steps=None,continuous_precision='rp5',game_update='secant_particle_exploration',adam_eager_state=True,num_particles=12,z_dim=4,batch_size=128)
ring=json.loads((O/'rp9-source-lock.json').read_text())
for name,want in ring['source_sha256'].items():
 if name.startswith('particlegan/'):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name
fixture=json.loads((O/'mode-hold-fixture.json').read_text())
for name,want in fixture['current_helpers'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name
entries=[str(p.relative_to(ROOT)) for p in [*sorted((ROOT/'particlegan').glob('*.py')),Path(__file__),O/'static_sources.py',O/'public_worker.py',O/'source_helpers.py',O/'rp9.md',O/'rp9-gate-order-amendment.md',O/'mode-hold-declaration.md',O/'mode-hold-fixture.json',O/'rp9-source-lock.json',plan]]
entries.extend(r['local_copy'] for r in fixture['archived_sources'].values())
sources,external=dependency_closure(ROOT,entries)
a.output.mkdir(parents=True,exist_ok=False)
decl=dict(candidate='API-RP9',gate='broader_mode_hold',spec=spec,recipe=recipe.to_dict(),serial_backward=True,fixture=fixture,source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},external_imports=external)
(a.output/'declaration.json').write_text(json.dumps(decl,indent=2)+'\n')
with zipfile.ZipFile(a.output/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name,data in sources.items():z.writestr(name,data)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(0);stream=torch.Generator(device='cuda:0').manual_seed(0)
with torch.device('cuda:0'):
 means=host.ring_means()
 prior=recipe.make_prior(init_std=.5,generator=stream)
 g=host.SimpleMLPGenerator(4,96,3,2);d=host.SimpleMLPDiscriminator(2,96,3,3)
assert sum(p.numel() for m in (g,prior) for p in m.parameters())==19346
assert sum(p.numel() for p in d.parameters())==20161
t=GANTrainer(recipe,g,d,prior=prior,seed=0,latent_generator=stream,serial_backward=True,optimizer_options={'foreach':False,'fused':False})
initial=dict(trainer=t.state_dict(),data_rng=stream.get_state());torch.save(initial,a.output/'initial-state.pt')
(a.output/'initial.json').write_text(json.dumps({'backend':'canonical CUDA shared-stream host','complete':digest(initial),'models':{k:digest(v) for k,v in initial['trainer']['models'].items()}},indent=2)+'\n')
(a.output/'runtime.json').write_text(json.dumps({'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'deterministic':True,'tf32':False,'threads':1,'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}},indent=2)+'\n')
def real_data(rng):
 with torch.device('cuda:0'):return host.sample_ring(means,128,host.SIGMA,rng)
@torch.no_grad()
def measure(ema=False):
 model,table=(t.ema_G,t.ema_prior) if ema else (t.G,t.prior)
 modes=[(m,m.training) for root in (model,table) for m in root.modules()]
 try:
  model.eval();table.eval()
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(402+t.completed_steps)
   latent=table.sample(4096,generator=torch.Generator(device='cuda:0').manual_seed(9))[0]
   fake=model(latent);sigma=output_noise_std(recipe,t.completed_steps)
   if sigma:fake=fake+sigma*torch.randn_like(fake)
   return host.diversity(fake,means)
 finally:
  for module,flag in modes:module.training=flag
started=time.monotonic();observations=[]
try:
 with (a.output/'metrics.jsonl').open('w',buffering=1) as obs,(a.output/'learning-rates.jsonl').open('w',buffering=1) as lr,(a.output/'batch-receipts.jsonl').open('w',buffering=1) as batches:
  for step in range(1,spec['steps']+1):
   real=real_data(stream)
   future=torch.Generator(device='cuda:0');future.set_state(stream.get_state())
   latent_d=prior.sample_indices(128,generator=future);latent_g=prior.sample_indices(128,generator=future)
   before_g_real=future.get_state();real_g=real_data(future);after_g_real=future.get_state()
   t.step(real,generator_real=real_g,collect_stats=step%50==0)
   assert torch.equal(stream.get_state(),before_g_real),'accepted latent stream did not consume exactly two frozen draws'
   stream.set_state(after_g_real)
   batches.write(json.dumps(dict(step=step,real_d=digest(real),latent_d=digest(latent_d),latent_g=digest(latent_g),real_g=digest(real_g),accepted_cursor=digest(stream.get_state())))+'\n')
   lr.write(json.dumps(dict(step=step,**rates(t),input_noise=input_noise_std(recipe,step-1),output_noise=output_noise_std(recipe,step-1),precision=t.precision.state,game=t.game_stats))+'\n')
   if step%50==0:
    before=digest([t.state_dict(),stream.get_state()]);row=dict(step=step,**measure(),ema=measure(True));assert digest([t.state_dict(),stream.get_state()])==before
    observations.append(row);obs.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
 verdict=test_verdict(spec,dict(live=observations[-1],observations=observations));assert verdict['convergence']['complete']
 status='PASS' if verdict['passed'] and verdict['convergence']['passing_suffix']>=5 else 'FAIL'
 metrics=dict(verdict=verdict,final=observations[-1],policy=t.precision.state,updates=t.completed_steps)
 torch.save(dict(trainer=t.state_dict(),data_rng=stream.get_state()),a.output/'final-state.pt')
except Exception as e:status='ERROR';metrics={'error':repr(e)}
row=dict(candidate='API-RP9',gate='broader_mode_hold',status=status,seconds=time.monotonic()-started,metrics=metrics,artifact=str(a.output.resolve()))
(a.output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
with (ROOT.parent/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
(a.output/'artifact-sha256.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.iterdir() if x.is_file() and x.name!='artifact-sha256.json'},indent=2)+'\n')
print(json.dumps(row),flush=True)

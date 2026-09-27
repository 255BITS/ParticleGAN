"""Independent CPU initialization proof. Stops before losses/optimizers/forward."""
from pathlib import Path
import argparse, hashlib, importlib, inspect, json, runpy, sys, zipfile
from unittest.mock import patch
E=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
B=args.bundle.resolve();ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
plan=read(B/'source-plan.json');manifest=read(B/'manifest.json')
for name,want in manifest['files'].items():assert sha((B/name).read_bytes())==want,name
runtime=Path(plan['historical_runtime'])
for name,want in plan['historical_runtime_files'].items():assert sha((runtime/name).read_bytes())==want,name
with zipfile.ZipFile(B/'historical-runtime-source.zip') as z:
 assert {n:sha(z.read(n)) for n in z.namelist()}==plan['historical_runtime_files']
assert sha((B/'historical-runtime-source.zip').read_bytes())==plan['historical_runtime_zip_sha256']
sys.path[:0]=[str(B),str(B/'candidate-source'),str(runtime)]
import torch
assert not torch.cuda.is_initialized() and str(torch.get_default_device())=='cpu'
torch.set_num_threads(1)
from benchmarks.locked_shared import mode_hold as host
assert sha(Path(host.__file__).read_bytes())==plan['frozen_host_sha256']
from particlegan.grad_regularizers import GradientPenalty
probe_path=B/'candidate-source/probe.py'
original_penalty=GradientPenalty.penalty;original_phi=GradientPenalty._phi
from initialization_bridge import load_initializer,bind_mode_hold,transform_function
public=load_initializer(B/'initializer-authority',plan['initializer_package'])
initializer=importlib.import_module(public.__name__+'.initialization')
prior_module=importlib.import_module(public.__name__+'.particle_prior')
original_function=host.train_mode_hold;original_prior=host.ParticlePrior;private_prior=prior_module.ParticlePrior
assert original_prior.__module__ == 'particlegan.particle_prior'
assert not any(n in sys.modules for n in ('mechanism','latent','response','checkpoint'))
original_source=inspect.getsource(original_function).rstrip('\n')
changed=transform_function(original_source)
assert sha(original_source.encode())==plan['train_mode_hold_sha256']
checks={};captures=[];rng=[];captured_streams=[]
def receipt(t):
 v=t.detach().cpu().contiguous();return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=sha(v.reshape(-1).view(torch.uint8).numpy().tobytes()))
def models(g,d,p):
 return {r:{k:receipt(v) for k,v in m.state_dict().items()} for r,m in [('generator',g),('critic',d),('prior',p)]}
def all_values(g,d,p):
 return {r:dict(parameters={k:receipt(v) for k,v in m.named_parameters()},buffers={k:receipt(v) for k,v in m.named_buffers()}) for r,m in [('generator',g),('critic',d),('prior',p)]}
class Captured(BaseException):pass
class Forbidden(Exception):pass
def forbidden(*args,**kwargs):raise Forbidden('Forward/backward/optimizer step is forbidden in constructor proof')
def capture(g,d,p,stream):
 assert type(p) is original_prior
 assert host.ParticlePrior is original_prior
 assert not any(n in sys.modules for n in ('mechanism','latent','response','checkpoint'))
 captured_streams.append(stream)
 captures.append(dict(material=models(g,d,p),all_values=all_values(g,d,p),cpu=receipt(torch.get_rng_state()),stream=receipt(stream.get_state())))
 raise Captured()
# Witness the exact public initializers without changing their operations.
original_initializers={n:getattr(initializer,n) for n in ['initialize_','_initialize_prior']}
shared=[None]
def witnessed(name):
 def call(*args,**kwargs):
  before=torch.get_rng_state().clone(); sb=None if shared[0] is None else shared[0].get_state().clone()
  value=original_initializers[name](*args,**kwargs)
  assert torch.equal(before,torch.get_rng_state())
  if sb is not None:assert torch.equal(sb,shared[0].get_state())
  rng.append(dict(operation=name,cpu_neutral=True,shared_stream_neutral=sb is not None))
  return value
 return call
with patch.object(torch.nn.Module,'_call_impl',forbidden),patch.object(torch.Tensor,'backward',forbidden),patch.object(torch.optim.Adam,'step',forbidden),patch.object(torch.optim.Adam,'__init__',forbidden):
 with patch.object(initializer,'initialize_',witnessed('initialize_')),patch.object(initializer,'_initialize_prior',witnessed('_initialize_prior')):
  with bind_mode_hold(host,public,plan['train_mode_hold_sha256'],capture):
   # Actual full function, arrested immediately after constructor initialization.
   argv=list(sys.argv)
   sys.argv=[str(probe_path),'--repo',str(runtime),'--config',str(B/'candidate-source/config.json'),'--task','mode_hold','--backend','cpu','--output',str(ROOT/'probe-capture')]
   try:runpy.run_path(str(probe_path),run_name='__main__')
   except Captured:pass
   finally:sys.argv=argv
   assert len(captures)==1
   first=captures[-1]
   # New constructors use the current RNG cursor, with extra draws and no reset.
   torch.rand(97);stream=captured_streams[0];torch.rand(113,generator=stream);shared[0]=stream
   p=host._newinit_make_prior(12,4,init_std=.5,generator=stream)
   g=host.SimpleMLPGenerator(4,96,3,2);d=host.SimpleMLPDiscriminator(2,96,3,3)
   try:host._newinit_models(g,d,p,stream)
   except Captured:pass
   assert len(captures)==2 and captures[1]['material']==first['material'] and captures[1]['all_values']==first['all_values']
   # A constructor exception must restore the private factory's class binding.
   def broken(*args,**kwargs):raise Captured('constructor failure')
   with patch.object(original_prior,'__init__',broken):
    try:host._newinit_make_prior(12,4,init_std=.5,generator=stream)
    except Captured:pass
   assert prior_module.ParticlePrior is private_prior and host.ParticlePrior is original_prior
   # A nested foreign BatchDistance class is refused before initialization.
   d.add_module('unreviewed_head',type('BatchDistanceDiscriminator',(torch.nn.Identity,),{})())
   try:host._newinit_models(g,d,p,stream)
   except RuntimeError as error:assert 'independent binding' in str(error)
   else:raise AssertionError('foreign BatchDistance head silently accepted')
  assert host.train_mode_hold is original_function and '_newinit_models' not in host.__dict__ and '_newinit_make_prior' not in host.__dict__
 # Deliberate scope exception must restore host function and namespace.
 try:
  with bind_mode_hold(host,public,plan['train_mode_hold_sha256'],capture):raise Captured('scope failure')
 except Captured:pass
 assert host.train_mode_hold is original_function and prior_module.ParticlePrior is private_prior and host.ParticlePrior is original_prior
 assert '_newinit_models' not in host.__dict__ and '_newinit_make_prior' not in host.__dict__
 # Compare untouched constructor RNG cursor using restored initial state in a
 # local fork only. No alternate seeds, learner execution or training occurs.
 with torch.random.fork_rng(devices=[]):
  zero=torch.Generator(device='cpu').manual_seed(0);torch.set_rng_state(zero.get_state())
  old_stream=torch.Generator(device='cpu').manual_seed(0)
  oldp=original_prior(12,4,init_std=.5,generator=old_stream)
  oldg=host.SimpleMLPGenerator(4,96,3,2);oldd=host.SimpleMLPDiscriminator(2,96,3,3)
  assert receipt(torch.get_rng_state())==first['cpu'] and receipt(old_stream.get_state())==first['stream']
 expected=read(E/'public-ka2-cpu-preflight.json')['cpu_initialization']['all_initial_material']
 for old,new in [('generator','G'),('critic','D'),('prior','prior')]:
  assert first['material'][old]==expected['state']['models'][new]
  assert first['all_values'][old]==expected['all_parameters_and_buffers'][new]
assert not torch.cuda.is_initialized()
assert not any(n in sys.modules for n in ('mechanism','latent','response','checkpoint'))
checks={key:True for key in ['all_initial_tensors_match_public_host','repeat_without_rng_reset','constructor_rng_cursor_preserved','initializer_rng_neutral','historical_prior_registration_preserved','all_bindings_restore_on_exception','batch_distance_scope_explicit']}
proof=dict(status='PASS',scope='CPU constructor-only independent proof; no forward/backward/optimizer step/training/GPU',source_plan_sha256=sha((B/'source-plan.json').read_bytes()),bridge_sha256=sha((B/'initialization_bridge.py').read_bytes()),runner_sha256=sha((B/'run_research_mode_hold.py').read_bytes()),manifest_sha256=sha((B/'manifest.json').read_bytes()),checks=checks,all_initial_material=first['material'],all_named_parameters_and_buffers=first['all_values'],constructor_end_cpu=first['cpu'],constructor_end_shared=first['stream'],initializer_rng_witnesses=rng,transform=dict(original_sha256=sha(original_source.encode()),transformed_sha256=sha(changed.encode())),historical_prior_class=str(original_prior),prior_binding_contract='Original plain ParticlePrior class and constructor preserved; no response registration or added mechanism hooks',original_probe_sha256=sha(probe_path.read_bytes()),penalty_binding=dict(penalty=GradientPenalty.penalty.__name__,phi=GradientPenalty._phi.__name__),cuda_initialized=False,learner_steps=0,limits=['Dense tiny frozen research mode_hold only','Requires own external CUDA execution; no quality result inherited','Original ordinary Adam setup and inline penalty remain; no added eager counters or six-hook mechanism stack', 'No final checkpoint in these original probe variants; no continuation claim'])
(ROOT/'cpu-constructor-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:proof[k] for k in ['status','scope','source_plan_sha256','bridge_sha256','manifest_sha256','checks','cuda_initialized','learner_steps']},indent=2))

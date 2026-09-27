"""Unchanged frozen image quality gate through API-DV6's actual public trainer."""
from worker import ROOT, digest, rates
from pathlib import Path
import argparse, hashlib, json, os, sys, time, zipfile
import torch
from particlegan import GANTrainer, get_recipe
from particlegan.training import input_noise_std, output_noise_std
from benchmarks.transfer_suite import image_tasks
from benchmarks.toy100.models import OUTPUT_NOISE_SEED_OFFSET
from benchmarks.locked_shared.observation import sustained

p=argparse.ArgumentParser();p.add_argument('--candidate',default='dv7');p.add_argument('--task',default='img_intensity2',choices=[s['name'] for s in image_tasks.TASKS[:4]])
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
plan_path=ROOT/'benchmarks/transfer_suite/plans/default_comparison.json'
spec=next(dict(job['spec']) for job in json.loads(plan_path.read_text()) if job['spec']['name']==a.task)
assert spec['runner']=='image'
recipe=get_recipe(total_steps=None,continuous_policy=a.candidate,input_noise_std=0.,output_noise_warmup=0.,
                  num_particles=spec['particles'],z_dim=spec['z_dim'],batch_size=spec['batch_size'])
sources=[*sorted((ROOT/'particlegan').glob('*.py')),Path(__file__),Path(image_tasks.__file__),ROOT/'benchmarks/locked_shared/observation.py', plan_path,
         Path(__file__).with_name('worker.py'),ROOT/'benchmarks/toy100/models.py',
         ROOT/'benchmarks/toy100/device.py',ROOT/'benchmarks/locked_shared/mode_hold.py',ROOT/'benchmarks/locked_shared/mlp.py']
ring=json.loads((ROOT/('reports/data-drift-api/candidate-packages/'+a.candidate+'.json')).read_text())
for name,expected in ring['source_sha256'].items():
 if name.startswith('particlegan/'):
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
sources += [ROOT/name for name in json.loads((ROOT/'reports/data-drift-api/serial-dependency-receipt.json').read_text())['dependencies'] if name.startswith('benchmarks/')]
sources += [ROOT/'reports/data-drift-api/frozen-image-initial.json', ROOT/('reports/data-drift-api/'+a.candidate+'.json'),ROOT/'reports/data-drift-api/remaining-images-protocol.json']
for module in list(sys.modules.values()):
 file=getattr(module,'__file__',None)
 if file and Path(file).is_file() and Path(file).suffix=='.py' and Path(file).is_relative_to(ROOT):
  sources.append(Path(file))
sources=sorted(set(sources))
manifest=dict(candidate='API-'+a.candidate.upper(),gate='broader_'+a.task,spec=spec,recipe=recipe.to_dict(),
              training='actual get_recipe and GANTrainer.step',seed=0,serial_backward=True,policy='same candidate as ring; only frozen host resources differ',
              initialization='frozen image G,D,prior construction order on CPU; then CUDA',
              streams='frozen image shared CUDA global data/latent stream; candidate private training-noise stream; isolated observation stream seed402+update+OUTPUT_NOISE_SEED_OFFSET(1901)',
              observation='enumerate all32 prior rows, original image_metrics and24-observation sustained gate; live primary, EMA diagnostic',
              deviation='Port frozen image host to same continuous public recipe. Noise comes only from API constant0/.029 policy, no external host schedule or wrapper; the raw card prior_weight is consistently replaced by recipe.prior_reg=0 as in the pinned public comparison; no task-required auxiliary objective is omitted.',
              source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
(a.output/'declaration.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(a.output/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in manifest['source_sha256']:z.write(ROOT/name,name)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(0)
centers=image_tasks.templates(spec).to('cuda:0')
g=image_tasks.Generator(spec);d=image_tasks.Discriminator(spec);prior=recipe.make_prior()
assert all(v.device.type=='cpu' for m in (g,d,prior) for v in m.parameters())
g=g.cuda();d=d.cuda();prior=prior.cuda()
shared=torch.cuda.default_generators[0]
t=GANTrainer(recipe,g,d,prior=prior,seed=0,latent_generator=shared,penalty_generator=shared,
             optimizer_options={'foreach':False,'fused':False},serial_backward=True)
manifest['runtime']={'torch':str(torch.__version__),'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'fp32':True,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'threads':torch.get_num_threads(),'serial_backward':t.serial_backward}
(a.output/'declaration.json').write_text(json.dumps(manifest,indent=2)+'\n')
torch.save(t.state_dict(),a.output/'initial-state.pt')
(a.output/'initial.json').write_text(json.dumps({'trainer':digest(t.state_dict()),'global_data_rng':digest(shared.get_state()),
  'models':{k:digest(v) for k,v in t.state_dict()['models'].items()}},indent=2)+'\n')
baseline=json.loads((ROOT/'reports/data-drift-api/frozen-image-initial.json').read_text())
assert {k:digest({n:x for n,x in v.items() if n not in ('log_width','shape_shear')}) for k,v in t.state_dict()['models'].items()}==baseline['models']
assert digest(shared.get_state())==baseline['global_data_rng']
started=time.monotonic();observations=[];expected=image_tasks.evaluation_steps(spec)
@torch.no_grad()
def measure(ema=False):
    model,prior=(t.ema_G,t.ema_prior) if ema else (t.G,t.prior)
    # Exact finite-prior enumeration, including the declared output noise.
    with torch.random.fork_rng(devices=[0]):
        stream=torch.Generator(device='cuda:0').manual_seed(402+t.completed_steps+OUTPUT_NOISE_SEED_OFFSET)
        samples=t._generate(model,prior.z,output_noise_std(recipe,t.completed_steps),stream,torch.arange(prior.num_particles,device=prior.z.device))
        return image_tasks.image_metrics(samples,centers,spec['thresholds'])
try:
 with (a.output/'metrics.jsonl').open('w',buffering=1) as obs,(a.output/'learning-rates.jsonl').open('w',buffering=1) as lr:
  for completed in range(1,spec['steps']+1):
   real=centers[torch.randint(len(centers),(spec['batch_size'],),device='cuda:0')]
   real=(real+spec['noise_std']*torch.randn_like(real)).clamp(0.,1.)
   before={name:[p.detach().clone() for p in m.parameters()] for name,m in [('G',t.G),('D',t.D),('prior',t.prior)]}
   t.step(real,generator_real=real,collect_stats=completed in expected)
   updates={}
   for name,m in [('G',t.G),('D',t.D),('prior',t.prior)]:
    delta=[p.detach()-b for p,b in zip(m.parameters(),before[name])]
    updates[name]={'l2':float(sum(x.square().sum() for x in delta).sqrt()),'max_abs':max(float(x.abs().max()) for x in delta)}
   updates['prior']['changed_rows']=int((delta[0]!=0).any(dim=1).sum())
   lr.write(json.dumps({'step':completed,**rates(t),'input_noise':input_noise_std(recipe,completed-1),
                       'output_noise':output_noise_std(recipe,completed-1),'controller':t.opt_d.record.state_dict(),'policy':t.controller.diagnostics(),'updates':updates})+'\n')
   if completed in expected:
    before=digest(t.state_dict());point={'step':completed,**measure(),'ema':measure(True)}
    assert digest(t.state_dict())==before,'evaluation mutated training state'
    observations.append(point);obs.write(json.dumps(point)+'\n');print(json.dumps(point),flush=True)
 convergence=sustained(observations,[('modes','>=',spec['thresholds']['modes']),('hq','>=',spec['thresholds']['hq_min'])],
                       expected_steps=expected,minimum=spec['thresholds']['minimum_stable_checks'])
 status='PASS' if convergence['complete'] and convergence['passing_suffix']>=spec['thresholds']['minimum_stable_checks'] else 'FAIL'
 metrics={'convergence':convergence,'final':observations[-1],'critic_memory':t.opt_d.record.state_dict(),'policy':t.controller.diagnostics(),'updates':t.completed_steps,'effective_recipe':t.recipe.to_dict(),
          'optimizer_classes':[type(t.opt_g).__name__,type(t.opt_d).__name__],
          'controller_class':type(t.controller).__name__}
 torch.save(t.state_dict(),a.output/'final-state.pt')
except Exception as e:
 status='ERROR';metrics={'error':repr(e)}
row=dict(candidate='API-'+a.candidate.upper(),gate='broader_'+a.task,status=status,seconds=time.monotonic()-started,
         metrics=metrics,artifact=str(a.output.resolve()))
(a.output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
with (ROOT.parent/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
(a.output/'artifact-sha256.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.iterdir() if x.is_file() and x.name!='artifact-sha256.json'},indent=2)+'\n')
print(json.dumps(row),flush=True)

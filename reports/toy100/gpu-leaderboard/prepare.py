"""Copy declared candidate sources and apply device/RNG plumbing only."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).parent
B=Path('/ml2/hypergan')
selected=B/'ParticleGAN-selected-h-stability-base'
column=B/'gan-attempts/tradeoff-20260924T191753Z/generator_geometry/20260924T191753Z-1689121/repo'
shared=B/'gan-attempts/eps-net-20260924T185715Z/stability_plus_mobility/20260924T185715Z-1646699/repo'
avg=B/'gan-attempts/post-convergence-20260924T194741Z/gradient_variance/20260924T194741Z-1750596/repo'
pr=B/'ParticleGAN-pr143-review'
cold='reports/toy100/h_stability/'
entries=[
 ('eps_net_1m',selected,selected/(cold+'eps-net-base/independent-cold/eps_net_1m')),
 ('H',selected,selected/'reports/toy100/critic_signal_attempt/batch-h/h_n05r06_mixup_c0p01_lr15'),
 ('shared_rms',shared,shared/(cold+'combined/batch02/eps_radial_g_shape_shared_rms')),
 ('shared_column_rms',column,column/(cold+'generator-moments/batch01/eps_radial_g_shape_shared_column_rms')),
 ('shared_rms_average2',avg,avg/(cold+'gradient-variance/batch02-ring/shared_rms_average2')),
]
rows=[]
for name,source,candidate in entries:
 cfg=json.loads((candidate/'config.json').read_text())
 opts=json.loads((candidate/'options.json').read_text())
 rows.append(dict(name=name,source=str(source),config=cfg,options=opts,kind='signal',
                  supported='three' if name=='shared_rms_average2' else 'all'))
for name,method in [('PR107','reachstall'),('PR140','delayg05'),('PR143','holdw15')]:
 cfg=json.loads((pr/'configs/toy100/constraints_simple_regularization.json').read_text())
 cfg.update(lr_floor=1.,lr_anneal_start=0.)
 for key in ('network_lr_horizon_cap','network_lr_floor'):cfg.pop(key,None)
 rows.append(dict(name=name,source=str(pr),config=cfg,options={},kind='pr',method=method,supported='two'))
for row in rows:
 source=Path(row['source']);target=ROOT/'repos'/row['name'];target.mkdir(parents=True,exist_ok=False)
 for name in ('benchmarks','particlegan','lib','configs'):
  shutil.copytree(source/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 resource=target/'reports/transfer_suite/unadjusted/leading_profile.json'
 resource.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(source/'reports/transfer_suite/unadjusted/leading_profile.json',resource)
 reports=target/'reports/toy100';reports.mkdir(parents=True)
 for path in (source/'reports/toy100').glob('*.py'):shutil.copy2(path,reports/path.name)
 stability=reports/'h_stability';stability.mkdir()
 for path in (source/'reports/toy100/h_stability').glob('*.py'):shutil.copy2(path,stability/path.name)
 # The common convergence observer is independent of candidate policy.
 shutil.copy2(selected/'reports/toy100/h_stability/convergence_gate.py',stability/'convergence_gate.py')
 before={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*.py')}
 edits={}
 for p in target.rglob('*.py'):
  # Adapt only benchmark/research plumbing. Public GANTrainer already owns CUDA RNG correctly.
  if str(p.relative_to(target)).startswith(('particlegan/','lib/')):continue
  old=p.read_text();new=old
  new=new.replace('torch.Generator()', 'torch.Generator(device=torch.get_default_device())')
  new=new.replace('torch.Generator(device="cpu")','torch.Generator(device=torch.get_default_device())')
  new=new.replace('torch.device("cpu")','torch.get_default_device()')
  new=new.replace('torch.random.fork_rng(devices=[])','torch.random.fork_rng(devices=([torch.cuda.current_device()] if torch.get_default_device().type == "cuda" else []))')
  new=new.replace('torch.random.default_generator.manual_seed(', 'torch.manual_seed(')
  new=new.replace('global_stream = torch.default_generator','global_stream = (torch.cuda.default_generators[torch.cuda.current_device()] if torch.get_default_device().type == "cuda" else torch.default_generator)')
  if p.name=='alternating_curvature_scratch.py':
   new=new.replace('return [torch.get_rng_state().clone()]+[s.get_state().clone() for s in streams]',
     'return [torch.get_rng_state().clone()]+[s.get_state().clone() for s in streams]+([torch.cuda.get_rng_state().clone()] if torch.get_default_device().type == "cuda" else [])')
   new=new.replace('for stream,state in zip(streams,states[1:]):stream.set_state(state)',
     'for stream,state in zip(streams,states[1:]):stream.set_state(state)\n        if torch.get_default_device().type == "cuda":torch.cuda.set_rng_state(states[-1].cpu())')
  if p.name=='mid_scale_identity.py':
   new=new.replace('    if any(param.is_cuda for param in student.parameters()):\n        raise RuntimeError("mid-scale identity toy is CPU only")','    if any(param.device != teacher.concept.device for param in student.parameters()):\n        raise RuntimeError("student and teacher must share a device")')
   new=new.replace('"device": "cpu"','"device": str(torch.get_default_device())')
  if p.name=='ae_gan_hold.py':
   new=new.replace('    state = torch.get_rng_state()','    state = torch.get_rng_state()\n    cuda_state = torch.cuda.get_rng_state() if torch.get_default_device().type == "cuda" else None')
   new=new.replace('        torch.set_rng_state(state)','        torch.set_rng_state(state)\n        if cuda_state is not None:\n            torch.cuda.set_rng_state(cuda_state)')
   new=new.replace('    torch.set_rng_state(stream.get_state())','    if torch.get_default_device().type == "cuda":\n        torch.cuda.set_rng_state(stream.get_state())\n    else:\n        torch.set_rng_state(stream.get_state())')
  if new!=old:
   p.write_text(new);edits[str(p.relative_to(target))]=dict(before=before[str(p.relative_to(target))],after=hashlib.sha256(p.read_bytes()).hexdigest())
 row['repo']=str(target)
 row['config']['device']='cuda:0'
 (target/'gpu-config.json').write_text(json.dumps(row['config'],indent=2)+'\n')
 (ROOT/f"sources-{row['name']}.json").write_text(json.dumps(dict(original=before,device_edits=edits),indent=2)+'\n')
(ROOT/'candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([dict(name=r['name'],supported=r['supported']) for r in rows]))

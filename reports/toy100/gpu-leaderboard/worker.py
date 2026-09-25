"""One declared CUDA toy or convergence gate. Never falls back to CPU."""
import argparse
from contextlib import ExitStack,contextmanager
from copy import deepcopy
import gzip,hashlib,inspect,json,os
from pathlib import Path
import subprocess,sys,time,traceback
from unittest.mock import patch

p=argparse.ArgumentParser()
p.add_argument('--candidate',required=True);p.add_argument('--task',required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
p.add_argument('--diagnostic-every',type=int,default=10)
a=p.parse_args();ROOT=Path(__file__).parent
row=next(r for r in json.loads((ROOT/'candidates.json').read_text()) if r['name']==a.candidate)
repo=Path(row['repo'])
sys.path[:0]=[str(repo),str(repo/'reports/toy100'),str(repo/'reports/toy100/h_stability')]
a.output.mkdir(parents=True,exist_ok=False)
import torch
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: no CPU fallback')
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('require CUBLAS_WORKSPACE_CONFIG=:4096:8')
torch.cuda.set_device(0);torch.set_default_device('cuda:0');torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
started=time.perf_counter();proof={'adam_calls':0,'optimizers':{},'cpu_parameter_calls':0,'gradient_devices':[]}

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def emit(**r):print(json.dumps(r,default=str),flush=True)
env=dict(profile='cuda_fp32_v1',torch=torch.__version__,torch_revision=torch.version.git_version,
 cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
 gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),device='cuda:0',visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
 deterministic=True,tf32=False,cudnn_benchmark=False,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
 threads=1,interop=1,adam_foreach=False,adam_fused=False)
config=deepcopy(row['config'])
dump(a.output/'declaration.json',dict(candidate=a.candidate,task=a.task,config=config,options=row['options'],
 environment=env,sources=str(ROOT/f'sources-{a.candidate}.json'),worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 scope='GPU research qualification; frozen targets/budgets/metrics, CUDA RNG streams; no CPU score reuse'))
original_adam_init=torch.optim.Adam.__init__;original_adam_step=torch.optim.Adam.step

def init(opt,*args,**kwargs):
 kwargs.setdefault('foreach',False);kwargs.setdefault('fused',False)
 original_adam_init(opt,*args,**kwargs)

def step(opt,*args,**kwargs):
 params=[p for group in opt.param_groups for p in group['params']]
 assert all(p.device.type=='cuda' for p in params),'CPU optimizer parameter: refusing score'
 assert all(p.grad is None or p.grad.device==p.device for p in params),'CPU gradient: refusing score'
 entry=proof['optimizers'].setdefault(str(id(opt)),dict(calls=0,parameters=sum(p.numel() for p in params),device='cuda:0'))
 entry['calls']+=1;proof['adam_calls']+=1
 result=original_adam_step(opt,*args,**kwargs)
 if entry['calls']==1:
  for state in opt.state.values():
   for k in ('exp_avg','exp_avg_sq'):
    if k in state:assert state[k].device.type=='cuda'
 if proof['adam_calls']%2000==0:emit(event='UPDATES',candidate=a.candidate,task=a.task,adam_calls=proof['adam_calls'])
 return result

@contextmanager
def policy(task):
 if row['kind']=='pr':
  if task not in ('trajectory','mode_hold','convergence','prefix'):
   raise NotImplementedError('Published PR adapter supports only trajectory and ring; no substitute policy scored')
  from reports.toy100 import gan_followup_probe as probe
  with probe.factory(row['method'])(task='mode_hold' if task in ('convergence','prefix') else task) as (recorder,source):
   (a.output/'generated-host.py').write_text(source)
   yield recorder
 elif row['supported']=='three' and task not in ('two_pole','mode_hold','unipolar','convergence','prefix'):
  raise NotImplementedError('Published independent-gradient-averaging adapter supports only two_pole/ring/unipolar')
 else:
  if 'adam_response' in row['options']:
   from adam_response import response_policy as impl
  else:
   from selected_h_extension import extended_signal_policy as impl
  with impl(row['options']) as receipt:yield receipt

class Finished(Exception):pass

def save_state(name):
 frame=inspect.currentframe()
 while frame and frame.f_code.co_name!='train_mode_hold':frame=frame.f_back
 if not frame:raise RuntimeError('could not locate live host state')
 v=frame.f_locals
 payload=dict(models={k:deepcopy(v[k].state_dict()) for k in ('generator','critic','prior')},
              optimizers={k:deepcopy(v[k].state_dict()) for k in ('opt_g','opt_d')},
              cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),stream_rng=v['stream'].get_state(),
              noise_policy=deepcopy(v['noise_policy'].__dict__))
 torch.save(payload,a.output/name)

def convergence():
 from benchmarks.toy100.continuous_probe import run_probe
 from convergence_gate import ConvergenceGate
 gate=ConvergenceGate();points=[]
 def observe(point):
  if point.get('event')!='checkpoint':return
  points.append(point)
  if a.limit and point['step']>=a.limit:
   save_state('final-state.pt');raise Finished()
  if point['step']==1200:save_state('cold-state.pt')
  if point['step']>1200:
   before=gate.converged_step
   done=gate.observe(point)
   if before is None and gate.converged_step is not None:
    save_state('converged-state.pt');emit(event='CONVERGED',step=point['step'])
   if done:
    save_state('final-state.pt');raise Finished()
 with ExitStack() as stack:
  if row['kind']=='pr' and row['method']!='reachstall':
   from reports.toy100.pr84_delayed_arm_g_lr import DelayedArmRecorder
   original=DelayedArmRecorder.consider_diagnostic
   def consider(self,index,modes,hq):
    # Dense observation must not give the method new control opportunities.
    if isinstance(index,int) and index%10==0:original(self,index,modes,hq)
   stack.enter_context(patch.object(DelayedArmRecorder,'consider_diagnostic',consider))
  with policy('convergence') as receipt:
   try:run_probe(config,mode='constant',steps=7200,diagnostic_every=a.diagnostic_every,dense_after=1200,log=observe)
   except Finished:pass
   dynamics=receipt.receipt() if hasattr(receipt,'receipt') else receipt
 result=dict(status='PREFIX_COMPLETE' if a.limit else gate.status,convergence=gate.summary(),diagnostic=points)
 dump(a.output/'result.json',result)
 if a.limit:
  raw=torch.load(a.output/'final-state.pt',weights_only=False,map_location='cpu')
  h=hashlib.sha256()
  for role,values in sorted(raw['models'].items()):
   for name,t in sorted(values.items()):h.update((role+'.'+name).encode()+t.contiguous().view(torch.uint8).numpy().tobytes())
  result['model_sha256']=h.hexdigest()
 return result,dynamics

result={};dynamics={};status='ERROR'
try:
 with patch.object(torch.optim.Adam,'__init__',init),patch.object(torch.optim.Adam,'step',step):
  emit(event='START',candidate=a.candidate,task=a.task,environment=env)
  if a.task in ('convergence','prefix'):
   result,dynamics=convergence();status=result['status']
  elif a.task in ('grid100','rotated100','staggered100'):
   from benchmarks.toy100.train import train
   from benchmarks.toy100.config import resolve_problem_config
   from benchmarks.toy100.gate import evaluate_suite
   from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
   cfg=resolve_problem_config(config,a.task,device='cuda:0')
   with policy(a.task) as dynamics:
    result=train(cfg,a.output/'native'/a.task)
   coverage=evaluate_suite(a.output/'native',problem=a.task)
   accuracy=accuracy_suite(a.output/'native',problem=a.task)
   result=dict(summary=result,coverage=coverage,accuracy=accuracy)
   status='PASS' if coverage['status']=='PASS' and accuracy['status']=='PASS' else 'FAIL'
  else:
   from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy,run_vector,run_image
   from benchmarks.transfer_suite.public_default_verification import load_declaration,declared_spec
   from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
   from benchmarks.transfer_suite.protocol import test_verdict
   recipe,noise,_=declared_recipe(config);jobs,profile=load_declaration()
   job=next(job for job in jobs if job['spec']['name']==a.task)
   spec,card,_=declared_spec(job,profile,recipe)
   with policy(a.task) as receipt:
    if spec['runner']=='vector':result,context=run_vector(spec,card,recipe,noise,model_policy=declared_model_policy(config))
    elif spec['runner']=='image':result,context=run_image(spec,recipe,noise,model_policy=declared_model_policy(config))
    else:result,context=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
    dynamics=receipt.receipt() if hasattr(receipt,'receipt') else receipt
   if result.get('error'):raise RuntimeError(result['error'])
   verdict=test_verdict(spec,result);status=verdict['status']
   result=dict(spec=spec,result=result,verdict=verdict)
  assert proof['adam_calls']>0,'no CUDA optimizer updates performed'
  proof['gradient_devices']=['cuda:0']
except NotImplementedError as exc:
 status='UNSUPPORTED';result=dict(reason=str(exc))
except Exception:
 result=dict(error=traceback.format_exc());emit(event='ERROR',error=result['error'])
finally:
 torch.cuda.synchronize()
 dump(a.output/'result.json',result)
 (a.output/'dynamics.json.gz').write_bytes(gzip.compress(json.dumps(dynamics,default=str).encode(),mtime=0))
 dump(a.output/'device-proof.json',proof)
 record=dict(candidate=a.candidate,task=a.task,status=status,seconds=time.perf_counter()-started,
             environment=env,device_proof=proof,artifact=str(a.output/'result.json'))
 if 'verdict' in result:record['metrics']=result['result'].get('live');record['verdict']=result['verdict']
 if 'convergence' in result:record['convergence']=result['convergence']
 if 'model_sha256' in result:record['model_sha256']=result['model_sha256']
 dump(a.output/'status.json',record);emit(event='DONE',**record)
raise SystemExit(2 if status=='ERROR' else 0)

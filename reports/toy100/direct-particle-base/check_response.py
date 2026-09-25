"""Focused CUDA analytic checks, without training or optimizer updates."""
import pathlib,sys,json,time,torch,traceback
root=pathlib.Path(__file__).resolve().parent;attempt=root.parents[2].parent
steering=(attempt/'supervisor.md').read_text() if (attempt/'supervisor.md').exists() else ''
if 'STOP' in steering:raise SystemExit('Supervisor STOP')
sys.path.insert(0,str(root/'prepared/repos/cuda'));sys.path.insert(0,str(root/'candidates/coherent_particle_response'))
import mechanism,response
from particlegan import GradientPenalty
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
start=time.perf_counter();cases=0
class D(torch.nn.Module):
 def __init__(self,w):super().__init__();self.w=torch.nn.Parameter(torch.tensor(w,device='cuda'));self.calls=0
 def forward(self,x):self.calls+=1;return self.w*x.square().flatten(1).sum(1)
try:
 for dim in (1,2,16,64):
  for w in (0.,.05,.8):
   real=torch.arange(1,2*dim+1,device='cuda',dtype=torch.float32).reshape(2,dim)/dim;fake=real+1
   d=D(w);reg=GradientPenalty('a_r1r2',coeff=1.,kappa=1.)
   actual,_=reg.penalty(d,real,fake,collect_stats=False)
   real_sq=(2*d.w*real).square().sum(1);fake_sq=(2*d.w*fake).square().sum(1)
   expected=.5*((real_sq/dim).mean()+((fake_sq+1e-12).sqrt()/dim**.5-1).relu().square().mean())
   ga=torch.autograd.grad(actual,d.w,retain_graph=True)[0];ge=torch.autograd.grad(expected,d.w)[0]
   assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6) and torch.allclose(ga,ge,atol=2e-5,rtol=2e-6)
   assert d.calls==2 and torch.isfinite(ga) and real.grad is None and fake.grad is None;cases+=1
 x=torch.tensor([-1.,1.],device='cuda');z=torch.zeros_like(x)
 for old,expected in ((None,1.),(x,2.),(-x,1.),(z,1.)):
  gain,_=response.gain_from(x,old);assert abs(gain-expected)<1e-6;cases+=1
 class O:pass
 o=O();p=torch.nn.Parameter(torch.zeros(2,1,device='cuda'));p.grad=x[:,None].clone()
 o.param_groups=[dict(params=[p],lr=.0085,_comparison_prior=True)]
 first=response.begin(o);response.end(first)
 second=response.begin(o);assert abs(o.param_groups[0]['lr']-.017)<1e-8
 assert torch.equal(p.grad,x[:,None]) and torch.count_nonzero(p)==0
 response.end(second);assert o.param_groups[0]['lr']==.0085
 assert response.receipt['history_devices']==['cuda:0'];cases+=1
 row=dict(candidate='regression',gate='dimension_rms_and_coherent_response',status='PASS',seconds=time.perf_counter()-start,metrics=dict(cases=cases,optimizer_updates=0,critic_formula=True,gradient_and_parameters_unchanged_by_gain=True,scheduled_lr_restored=True,response_history_cuda=True),artifact=str(root/'mechanism-check.json'))
except Exception:row=dict(candidate='regression',gate='dimension_rms_and_coherent_response',status='ERROR',seconds=time.perf_counter()-start,metrics=dict(cases=cases,error=traceback.format_exc()),artifact=str(root/'mechanism-check.json'))
pathlib.Path(row['artifact']).write_text(json.dumps(row,indent=2)+'\n')
with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row));raise SystemExit(2 if row['status']=='ERROR' else 0)

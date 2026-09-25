"""Closed-form penalty/parameter-gradient checks; no toy training."""
import argparse,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('candidate');a=p.parse_args()
root=Path(__file__).resolve().parent;attempt=root.parents[3]
torch.set_num_threads(1);torch.set_num_interop_threads(1)
sys.path.insert(0,str(root/'prepared/repos/cuda'))
c=root/'candidates'/a.candidate
config=json.loads((c/'config.json').read_text())
spec=importlib.util.spec_from_file_location('mechanism',c/'mechanism.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from particlegan.grad_regularizers import GradientPenalty
class Critic(torch.nn.Module):
 def __init__(self,w):
  super().__init__();self.w=torch.nn.Parameter(torch.tensor(w,dtype=torch.float64));self.calls=0
 def forward(self,x):
  self.calls+=1;return self.w*x.square().flatten(1).sum(1)
start=time.perf_counter();cases=0;status='PASS';details={}
try:
 for shape in ((2,),(16,),(1,8,8)):
  dim=1
  for n in shape:dim*=n
  real=torch.arange(1,2*dim+1,dtype=torch.float64).reshape(2,*shape)/dim;fake=real+1
  if a.candidate=='detached_fake_scale_cap':fake=real.flip(0)*.25
  for weight in (0.,.05,.8):
   d=Critic(weight);reg=GradientPenalty(config['reg_arm'],coeff=config['reg_coeff'],kappa=config['reg_kappa'])
   actual,_=reg.penalty(d,real,fake,collect_stats=False)
   real_sq=(2*d.w*real).square().flatten(1).sum(1)
   fake_sq=(2*d.w*fake).square().flatten(1).sum(1)
   if a.candidate=='dimension_rms_hybrid':
    expected=.5*((real_sq/dim).mean()+((fake_sq+1e-12).sqrt()/dim**.5-1).relu().square().mean())
   elif a.candidate=='dimension_rms_bcap':
    expected=.5*((((real_sq+1e-12)/dim).sqrt()-.5).relu().square().mean()+(((fake_sq+1e-12)/dim).sqrt()-.5).relu().square().mean())
   else:
    cap=((fake_sq.detach()+1e-12).mean()/dim).sqrt()
    expected=.5*((((real_sq+1e-12)/dim).sqrt()-cap).relu().square().mean()+(((fake_sq+1e-12)/dim).sqrt()-1).relu().square().mean())
   ga=torch.autograd.grad(actual,d.w,retain_graph=True)[0];ge=torch.autograd.grad(expected,d.w)[0]
   assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
   assert torch.allclose(ga,ge,atol=1e-12,rtol=1e-12)
   assert torch.isfinite(ga) and d.calls==2
   assert real.grad is None and fake.grad is None
   cases+=1
 # Lazy skip must return zero and do no forwards.
 d=Critic(.8);reg=GradientPenalty(config['reg_arm'],lazy_k=2)
 skip,_=reg.penalty(d,real,fake,step=1);assert skip.item()==0 and d.calls==0
 details={'analytic_cases':cases,'zero_gradient_finite':True,'two_critic_forwards':True,'no_input_gradients':True,'lazy_skip':True}
except Exception:
 status='ERROR';details={'error':traceback.format_exc(),'cases_before_error':cases}
artifact=root/(a.candidate+'-mechanism-check.json')
row={'candidate':'regression','gate':a.candidate+'_analytic_penalty','status':status,'seconds':time.perf_counter()-start,'metrics':details,'artifact':str(artifact),'mechanism_sha256':hashlib.sha256((c/'mechanism.py').read_bytes()).hexdigest()}
artifact.write_text(json.dumps(row,indent=2)+'\n')
with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row),flush=True)
raise SystemExit(2 if status=='ERROR' else 0)

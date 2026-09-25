import hashlib,json,runpy,sys,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'prepared/repos/cuda'));torch.set_num_threads(1)
started=time.perf_counter();mechanism=root/'r1_only_install.py';runpy.run_path(str(mechanism),init_globals={'torch':torch})
from particlegan.grad_regularizers import GradientPenalty
reg=GradientPenalty('a_r1r2',coeff=1.)
class Critic(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.3,.4],dtype=torch.float64));self.calls=0
 def forward(self,x):
  self.calls+=1;return (x.square()*self.weight).sum(1)
d=Critic();real=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64);fake=real+4
pen,_=reg.penalty(d,real,fake,collect_stats=False);gradient=torch.autograd.grad(pen,d.weight)[0]
# D=sum(w*x^2): R1/2 = 2 sum(w^2 E[x^2]); derivative=4*w*E[x^2].
expected=4*d.weight*real.square().mean(0)
assert torch.allclose(gradient,expected,atol=1e-12,rtol=1e-12);assert d.calls==2
other,_=reg.penalty(d,real,-fake*3,collect_stats=False);other_grad=torch.autograd.grad(other,d.weight)[0]
assert torch.equal(other,pen) and torch.equal(other_grad,gradient)
# Full zero-weight branch still participates in autograd, preserving declared work.
result=dict(checks=3,real_gradient_correct=True,fake_samples_have_zero_penalty_effect=True,critic_forwards_per_penalty=2,mechanism_sha256=hashlib.sha256(mechanism.read_bytes()).hexdigest())
artifact=root/'r1-only-check.json';artifact.write_text(json.dumps(result,indent=2)+'\n')
row=dict(candidate='regression',gate='r1_only_gradient',status='PASS',seconds=time.perf_counter()-started,metrics=result,artifact=str(artifact))
with (root.parents[3]/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row))

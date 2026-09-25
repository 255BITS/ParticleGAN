import hashlib,json,math,runpy,sys,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'prepared/repos/cuda'));torch.set_num_threads(1)
started=time.perf_counter();mechanism=root/'smooth_cap_install.py';runpy.run_path(str(mechanism),init_globals={'torch':torch})
from particlegan.grad_regularizers import GradientPenalty
reg=GradientPenalty('b_cap',coeff=1.,kappa=1.)
d=torch.nn.Linear(2,1,bias=False).double();real=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64);fake=-real
with torch.no_grad():d.weight.copy_(torch.tensor([[.3,.4]],dtype=torch.float64))
pen,_=reg.penalty(d,real,fake,collect_stats=False);gradient=torch.autograd.grad(pen,d.weight)[0];expected=2*d.weight*(1-1/math.sqrt(1.25))
assert torch.allclose(gradient,expected,atol=1e-10,rtol=1e-8)
errors=[]
for i in range(2):
 with torch.no_grad():d.weight[0,i]+=1e-5
 plus=reg.penalty(d,real,fake,collect_stats=False)[0].item()
 with torch.no_grad():d.weight[0,i]-=2e-5
 minus=reg.penalty(d,real,fake,collect_stats=False)[0].item()
 with torch.no_grad():d.weight[0,i]+=1e-5
 errors.append(abs((plus-minus)/2e-5-gradient[0,i].item()))
assert max(errors)<1e-8
with torch.no_grad():d.weight.zero_()
pen,_=reg.penalty(d,real,fake,collect_stats=False);zero_grad=torch.autograd.grad(pen,d.weight)[0]
assert torch.isfinite(zero_grad).all() and zero_grad.abs().max()==0
result=dict(checks=3,analytic_gradient=True,finite_difference_max_error=max(errors),zero_slope_finite=True,mechanism_sha256=hashlib.sha256(mechanism.read_bytes()).hexdigest())
artifact=root/'smooth-cap-check.json';artifact.write_text(json.dumps(result,indent=2)+'\n')
row=dict(candidate='regression',gate='smooth_cap_gradient',status='PASS',seconds=time.perf_counter()-started,metrics=result,artifact=str(artifact))
with (root.parents[3]/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row))

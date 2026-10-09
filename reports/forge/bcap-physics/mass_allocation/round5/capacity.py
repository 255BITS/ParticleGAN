import time,json,torch,sys
from pathlib import Path
sys.path.insert(0,'/home/martyn/dev/ParticleGAN-bcap-r5-mass_allocation')
from particlegan.kinetic_transport import balanced_assignment_loss,kinetic_transport_local_loss,kinetic_transport_loss
from experiments.forge.contracts import atomic_json
start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
x=torch.arange(4096,device='cuda:0',dtype=torch.float32).reshape(2048,2);real=torch.sin(x/19);fake=(real*.8+.1).detach().requires_grad_();rows=[]
for mode,fn in [('sliced',kinetic_transport_loss),('balanced_assignment',balanced_assignment_loss)]:
 torch.cuda.synchronize();t=time.monotonic();loss=fn(fake,real)+kinetic_transport_local_loss(fake,real);grad=torch.autograd.grad(loss,fake)[0];torch.cuda.synchronize();rows.append(dict(mode=mode,seconds=time.monotonic()-t,loss=float(loss),gradient_finite=bool(torch.isfinite(grad).all()),projected_7000_loss_only_seconds=(time.monotonic()-t)*7000))
r=dict(schema_version=1,qualification_input=False,scope='synthetic2048-row sample-space forward/backward only; no trainer, optimizer updates, learned draws or quality claims',reservation_seconds=120,elapsed_seconds=time.monotonic()-start,rows=rows,optimizer_updates_added=0);atomic_json(Path('/home/martyn/dev/ParticleGAN-bcap-r5-mass_allocation/reports/forge/bcap-physics/mass_allocation/round5/capacity.json'),r);print(json.dumps(r))

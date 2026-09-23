from pathlib import Path
from copy import deepcopy
import sys,json
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-recipe')
import torch
from particlegan import GradientPenalty
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES,constructor
from benchmarks.transfer_suite.skip_critic_research import ARCHITECTURES as OLD,constructor as old_constructor

torch.set_num_threads(1);old_card=next(c for c in OLD if c['name']=='fourier_softplus5_linear_skip');control=deepcopy(ARCHITECTURES[0]);control['beta']=5.
torch.manual_seed(0);a=old_constructor(old_card)(2,64,2,2)
torch.manual_seed(0);b=constructor(control)(2,64,2,2)
assert list(a.state_dict())==list(b.state_dict()) and all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
rng=torch.Generator().manual_seed(0);real=torch.randn(16,2,generator=rng);fake=torch.randn(16,2,generator=rng)
assert torch.equal(a(fake),b(fake));gradients=[];caps=[]
for model in (a,b):
 f=fake.clone().requires_grad_(True);cap=GradientPenalty('b_cap',coeff=3.,kappa=.0001)(model,real,f.detach(),step=1,generator=torch.Generator().manual_seed(0));assert cap>0
 loss=model(f).mean()+cap;loss.backward();gradients.append([f.grad.clone()]+[p.grad.clone() for p in model.parameters()]);caps.append(cap.detach())
assert torch.equal(caps[0],caps[1]) and all(torch.equal(x,y) for x,y in zip(*gradients))
rows=[]
for card in ARCHITECTURES:
 torch.manual_seed(0);d=constructor(card)(2,card['hidden'],2,2)
 assert torch.equal(d(fake),d.main(fake))
 assert torch.allclose(d(fake),torch.cat([d(x[None]) for x in fake]),atol=1e-7,rtol=1e-5)
 f=fake.clone().requires_grad_(True);penalty=GradientPenalty('b_cap',coeff=3.,kappa=.0001)(d,real,f.detach(),step=1,generator=torch.Generator().manual_seed(0));assert penalty>0
 (d(f).mean()+penalty).backward();assert torch.isfinite(f.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())
 rows.append({'name':card['name'],'parameters':sum(p.numel() for p in d.parameters()),'pointwise':True,'zero_initial_skip':True,'active_cap_backward_finite':True})
result={'control_is_not_a_training_episode':True,'old_d64_beta5_exact_state_output_inputgrad_parametergrad_and_active_cap_parity':True,'cards':rows}
Path('/tmp/pr36-valid-linear-final/architecture_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

from pathlib import Path
import sys,json
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-recipe')
import torch
from benchmarks.transfer_suite.skip_critic_research import ARCHITECTURES,constructor,quadratic
from benchmarks.transfer_suite.smooth_critic_research import SmoothFourierCritic
from particlegan import GradientPenalty

torch.set_num_threads(1);rows=[]
for c in ARCHITECTURES:
 torch.manual_seed(0);d=constructor(c)(2,64,2,2);rng=torch.Generator().manual_seed(0);real=torch.randn(16,2,generator=rng)
 assert torch.equal(d(real),d.base_score(real))
 assert torch.allclose(d(real),torch.cat([d(x[None]) for x in real]),atol=1e-7,rtol=1e-5)
 if c['features']=='axis_fourier' and not c['residual_blocks']:
  torch.manual_seed(0);a=dict(features='axis',activation='softplus' if c['activation']=='softplus5' else 'silu',beta=5.)
  baseline=SmoothFourierCritic(2,64,2,2,architecture=a)
  assert torch.equal(d(real),baseline(real))
  assert all(torch.equal(x,y) for x,y in zip(d.main.parameters(),baseline.parameters()))
 with torch.no_grad():
  for p in d.parameters():p.mul_(2.)
 fake=torch.randn(16,2,generator=rng,requires_grad=True)
 penalty=GradientPenalty('b_cap',coeff=3.,kappa=.0001)(d,real,fake.detach(),step=1,generator=rng)
 assert penalty>0 and torch.isfinite(penalty)
 (penalty+d(fake).mean()).backward()
 assert torch.isfinite(fake.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())
 rows.append({'name':c['name'],'parameters':sum(p.numel() for p in d.parameters()),'pointwise':True,'zero_skip_output_initialization':True,'active_cap_backward_finite':True,'zero_skip_base_output_unchanged':True,'matches_existing_axis_base_at_init':c['features']=='axis_fourier' and not c['residual_blocks']})
assert torch.equal(quadratic(torch.tensor([[2.,3.]])),torch.tensor([[2.,3.,4.,6.,9.]]))
Path('/tmp/pr36-valid-skip-d/architecture_checks.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))

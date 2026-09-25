from pathlib import Path
import sys,json
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-recipe')
import torch
from benchmarks.transfer_suite.local_critic_research import ARCHITECTURES,constructor
from particlegan import GradientPenalty

torch.set_num_threads(1);rows=[]
for card in ARCHITECTURES:
 torch.manual_seed(0);critic=constructor(card)(2,64,2,2)
 rng=torch.Generator().manual_seed(0);points=torch.randn(16,2,generator=rng)
 # Pointwise scores must not depend on companion batch samples.
 together=critic(points);separate=torch.cat([critic(p[None]) for p in points])
 assert torch.allclose(together,separate,atol=1e-7,rtol=1e-5)
 original_state={k:v.clone() for k,v in critic.state_dict().items()}
 torch.manual_seed(0);replica=constructor(card)(2,64,2,2)
 assert all(torch.equal(v,replica.state_dict()[k]) for k,v in original_state.items())
 with torch.no_grad():
  output=critic.output if card['residual_blocks'] else critic.net[-1];output.weight.mul_(100.)
 fake=torch.randn(16,2,generator=rng,requires_grad=True)
 penalty=GradientPenalty('b_cap',coeff=3.,kappa=.001)(critic,points,fake.detach(),step=1,generator=rng)
 assert torch.isfinite(penalty) and penalty>0
 (critic(fake).mean()+penalty).backward()
 assert torch.isfinite(fake.grad).all()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
 if card['train_centers']:assert critic.centers.grad.norm()>0
 if card['train_widths']:assert critic.log_width.grad.norm()>0
 rows.append({'name':card['name'],'parameters':sum(p.numel() for p in critic.parameters()),'feature_dimensions':2+card['count'],'pointwise':True,'active_penalty_backward_finite':True,'fixed_init_reproduced':True})
path=Path('/tmp/pr36-valid-local-d/architecture_checks.json');path.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))

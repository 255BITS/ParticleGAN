"""Static architecture/derivative checks, not training experiments."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-recipe')
import torch
from benchmarks.transfer_suite.smooth_critic_research import ARCHITECTURES,constructor
from lib.toy_models import SimpleMLPDiscriminator
from particlegan import GradientPenalty

torch.set_num_threads(1)
rows=[]
for card in ARCHITECTURES:
 torch.manual_seed(0);model=constructor(card)(2,64,2,2)
 torch.manual_seed(0);replica=constructor(card)(2,64,2,2)
 assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),replica.state_dict().values()))
 generator=torch.Generator().manual_seed(0);real=torch.randn(16,2,generator=generator);fake=torch.randn(16,2,generator=generator,requires_grad=True)
 output=model(fake);assert output.shape==(16,) and torch.isfinite(output).all()
 loss=output.mean()+GradientPenalty('b_cap',coeff=3.,kappa=1.25)(model,real,fake.detach(),step=1,generator=generator)
 loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 assert fake.grad is not None and torch.isfinite(fake.grad).all()
 if card['features']=='axis':
  original=SimpleMLPDiscriminator(2,64,2,2);phase=(real.unsqueeze(-1)*original.freqs).flatten(1)
  assert torch.equal(model.encode(real),torch.cat([real,phase.sin(),phase.cos()],1))
 else:
  actual=model.projection.norm(dim=1)/torch.pi
  assert torch.allclose(actual,torch.tensor(card['radial_bands']))
 rows.append({'name':card['name'],'parameters':sum(p.numel() for p in model.parameters()),'input_features':model.encode(real).shape[1],'finite_backward_and_penalty':True,'reproducible_fixed_initialization':True})
path=Path('/tmp/pr36-valid-smooth-d/architecture_checks.json');path.write_text(json.dumps(rows,indent=2)+'\n');print(path);print(json.dumps(rows))

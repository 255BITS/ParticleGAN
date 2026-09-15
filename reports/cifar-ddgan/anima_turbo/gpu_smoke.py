import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,yaml
from experiments.train_cifar_ddgan import DEFAULTS,validate
from lib.image_anima import AnimaTransplantGenerator
from lib.image_ddgan import ImageGenerator,update_ema
name=sys.argv[1]
torch.set_num_threads(2)
cfg={**DEFAULTS,**yaml.safe_load(Path(f'configs/cifar_ddgan/anima_turbo_10k/{name}.yaml').read_text())}
validate(cfg)
torch.manual_seed(cfg['seed']); baseline=ImageGenerator(cfg).cuda()
torch.manual_seed(cfg['seed']); g=AnimaTransplantGenerator(cfg).cuda().train()
g.requires_grad_(False).requires_grad_(True)
assert all(not p.requires_grad for p in g.donor.parameters())
original=torch.load(cfg['anima_weights'],weights_only=True)['tensors']
for k,p in g.donor.named_parameters():
 assert torch.equal(p.cpu(),original[k].to(p.dtype)), k
versions={k:p._version for k,p in g.donor.named_parameters()}
z=torch.randn(2,cfg['z_dim'],device='cuda',requires_grad=True)
x=torch.randn(2,3,32,32,device='cuda',requires_grad=True)
c=torch.tensor([2,3],device='cuda'); t=torch.tensor([1,4],device='cuda')
torch.testing.assert_close(g(z,c,x,t),baseline(z,c,x,t),rtol=0,atol=0)
optimizer=torch.optim.Adam((p for p in g.parameters() if p.requires_grad),lr=cfg['lr'])
target=torch.randn_like(x)
for step in range(2):
 optimizer.zero_grad(set_to_none=True)
 (g(z,c,x,t)-target).square().mean().backward()
 for p in [x,z,g.transplant_out.weight]+([g.transplant_in.weight,g.context_z.weight,g.context_c.weight] if step else []):
  assert p.grad.isfinite().all() and p.grad.abs().sum()>0
 assert all(p.grad is None for p in g.donor.parameters())
 optimizer.step()
assert versions=={k:p._version for k,p in g.donor.named_parameters()}
result={'arm':name,'device':torch.cuda.get_device_name(),'initial_identity':True,'source_weights_match':True,'image_particle_adapter_gradients':True,'donor_frozen':True,'metadata':g.pretrained_metadata}
Path(f'results/cifar_ddgan/anima_turbo_validation/{name}.json').write_text(json.dumps(result,indent=2)+'\n')
print(name,'PASS')

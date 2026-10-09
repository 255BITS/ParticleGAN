import itertools
import pytest
import torch
from particlegan import Recipe,get_recipe,GANTrainer
from particlegan.kinetic_transport import balanced_assignment_loss,kinetic_transport_loss

def test_assignment_exact_optimum_gradient_detached_real_and_conservation():
 x=torch.tensor([[-2.,0.],[1.,.5],[2.,-.3]],dtype=torch.float64,requires_grad=True)
 y=torch.tensor([[-1.,.3],[.2,.8],[3.,-.4]],dtype=torch.float64,requires_grad=True)
 scale=(y.detach()-y.detach().mean(0)).square().mean()
 exact=min((x-y[list(p)]).square().mean().item()/scale.item() for p in itertools.permutations(range(3)))
 assert balanced_assignment_loss(x,y).item()==pytest.approx(exact)
 gx,gy=torch.autograd.grad(balanced_assignment_loss(x,y),(x,y),allow_unused=True);assert gy is None
 assert torch.autograd.gradcheck(lambda f:balanced_assignment_loss(f,y),(x,))
 null=y.detach().clone().requires_grad_();v=balanced_assignment_loss(null,y);assert v.item()==0;assert torch.count_nonzero(torch.autograd.grad(v,null)[0])==0

def test_assignment_units_rotation_permutations_blocks_tail_and_no_rng():
 x=torch.tensor([[-2.,0.],[1.,.5],[2.,-.3],[.7,2.],[.1,-.2]],dtype=torch.float64)
 y=x+torch.tensor([.3,-.6]);rng=torch.get_rng_state().clone();v=balanced_assignment_loss(x,y)
 assert balanced_assignment_loss(7*x+11,7*y+11).item()==pytest.approx(v.item())
 assert balanced_assignment_loss(x.flip(0),y.roll(2,0)).item()==pytest.approx(v.item())
 rot=torch.tensor([[0.,-1.],[1.,0.]],dtype=x.dtype);assert balanced_assignment_loss(x@rot,y@rot).item()==pytest.approx(v.item())
 scale=(y-y.mean(0)).square().mean();expected=sum(min((x[s:s+2]-y[s:s+2][list(p)]).square().sum().item() for p in itertools.permutations(range(len(x[s:s+2])))) for s in range(0,len(x),2))/(x.numel()*scale.item())
 assert balanced_assignment_loss(x,y,block_size=2).item()==pytest.approx(expected)
 assert torch.equal(rng,torch.get_rng_state())

def test_assignment_scalar_equals_original_and_validation():
 x=torch.tensor([[-2.],[-1.],[0.],[2.]],dtype=torch.float64);y=torch.tensor([[-2.],[0.],[1.],[2.]],dtype=torch.float64)
 assert balanced_assignment_loss(x,y).item()==kinetic_transport_loss(x,y).item()
 for kw in [{'kinetic_transport_mode':'unknown'},{'kinetic_transport_block_size':1},{'kinetic_transport_block_size':True}]:
  with pytest.raises(ValueError):Recipe(**kw)
 with pytest.raises(ValueError):balanced_assignment_loss(x,y.float())
 from particlegan.training import _normalized_recipe
 assert _normalized_recipe({'kinetic_transport_mode':'sliced','kinetic_transport_block_size':128})==_normalized_recipe({})

def test_assignment_missing_solver_blocks_before_reservation(monkeypatch):
 from experiments.forge.api import task_policy_blockers
 import importlib.util
 monkeypatch.setattr(importlib.util,'find_spec',lambda name:None)
 blockers=task_policy_blockers({'id':'fixture','execution':{}},{'recipe_overrides':{'kinetic_transport_weight':1.,'kinetic_transport_mode':'balanced_assignment'}})
 assert any('SciPy' in b for b in blockers)

def test_public_trainer_dispatch_preserves_critic_and_streams():
 def build(mode):
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(0)
   recipe=get_recipe('bcap',num_particles=12,z_dim=2,batch_size=6,total_steps=1,input_noise_std=0.,output_noise_std=0.,kinetic_transport_weight=1.,kinetic_transport_local_weight=1.,kinetic_transport_mode=mode)
   return GANTrainer(recipe,torch.nn.Linear(2,2),torch.nn.Linear(2,1))
 a,b=build('sliced'),build('balanced_assignment');real=torch.tensor([[-1.,-.2],[-.7,.2],[.6,.7],[1.1,-.5],[.3,.15],[1.7,.1]])
 ca,cb=a.step(real),b.step(real)
 assert cb['loss_g'].item()==pytest.approx((cb['loss_gan']+cb['kinetic_transport']+cb['kinetic_transport_local']).item())
 assert cb['kinetic_transport'].item()!=ca['kinetic_transport'].item()
 assert all(torch.equal(v,b.D.state_dict()[k]) for k,v in a.D.state_dict().items())
 assert any(not torch.equal(x,y) for x,y in zip(a.G.parameters(),b.G.parameters()))
 for k,v in a.state_dict()['streams'].items():assert torch.equal(v,b.state_dict()['streams'][k])
 assert b.state_dict()['recipe']['kinetic_transport_mode']=='balanced_assignment'

"""Finite-displacement contracts; these are software tests, not quality screens."""
from copy import deepcopy
import pytest
import torch

from particlegan import get_recipe
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.optim.critic_cap import CriticCapOptimizer


def setup(mode='finite_cap', weight=.9, lr=.2):
    recipe=get_recipe('bcap').replace(optimizer_family='dualnorm', critic_step_mode=mode,
        lr=lr,d_lr_mult=1.,reg_every=1,input_noise_std=0.,output_noise_std=0.)
    critic=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad(): critic.weight.fill_(weight)
    optimizer=recipe.make_critic_optimizer(critic)
    return recipe,critic,optimizer,recipe.make_critic_penalty(optimizer)


def step(critic,optimizer,penalty,sign=-1.):
    optimizer.zero_grad()
    x=torch.tensor([[-1.],[1.]])
    value=sign*critic.weight.sum()+penalty(critic,x,x/2)
    value.backward()
    optimizer.step()


def test_actual_finite_step_halves_to_cap_and_records_one_clock():
    _,critic,opt,penalty=setup()
    step(critic,opt,penalty)
    assert isinstance(opt,CriticCapOptimizer)
    assert float(critic.weight.detach())<=1.
    assert .9<float(critic.weight.detach())<1.
    assert opt.record.observed_steps==opt.state[critic.weight]['step']==1
    assert opt.record.calls==1
    assert opt.critic_cap_stats['attempted_scales']>1
    assert opt.critic_cap_stats['damped_steps']==1
    assert opt.record.lr_last<.2


def test_rejected_step_keeps_actual_parameters_but_advances_clock():
    _,critic,opt,penalty=setup(weight=1.2)
    old=critic.weight.detach().clone()
    step(critic,opt,penalty,sign=-100.)
    assert torch.equal(old,critic.weight)
    assert opt.record.observed_steps==1 and opt.record.lr_last==0.
    assert opt.critic_cap_stats['rejected_steps']==1
    assert opt.critic_cap_stats['minimum_scale']==0.


def test_boundary_checkpoint_resumes_actual_step_and_missing_active_state_fails_closed():
    _,critic,opt,penalty=setup()
    step(critic,opt,penalty)
    model,state=deepcopy(critic.state_dict()),deepcopy(opt.state_dict())
    _,other,resumed,bound=setup()
    other.load_state_dict(model)
    resumed.load_state_dict(state)
    step(critic,opt,penalty,sign=1.)
    step(other,resumed,bound,sign=1.)
    assert torch.equal(critic.weight,other.weight)
    assert opt.record.state_dict()==resumed.record.state_dict()
    assert opt.critic_cap_stats==resumed.critic_cap_stats
    bad=deepcopy(state);bad.pop('critic_cap')
    before=deepcopy(resumed.state_dict())
    with pytest.raises(ValueError): resumed.load_state_dict(bad)
    assert resumed.critic_cap_stats==before['critic_cap']['stats']


def test_disabled_factory_retains_original_type_and_no_optional_checkpoint_fields():
    recipe,critic,opt,penalty=setup('none')
    assert type(opt) is NormalizedOptimizer
    assert 'critic_step_mode' not in recipe.to_dict()
    assert not hasattr(opt,'_cap_panels')
    step(critic,opt,penalty)
    assert 'critic_cap' not in opt.state_dict()


def test_enabled_requires_current_penalty_and_rejects_rng_consuming_probe():
    _,critic,opt,_=setup()
    with pytest.raises(ValueError): opt.step()
    cpu=torch.get_rng_state().clone()
    def draw(): return torch.rand(())
    opt.bind_cap_panel(draw,1.)
    with pytest.raises(ValueError,match='RNG'): opt.step()
    assert torch.equal(cpu,torch.get_rng_state())
    assert opt.record.observed_steps==0

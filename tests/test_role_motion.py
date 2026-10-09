from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer,get_recipe,init
from particlegan.role_motion import RoleMotionBalance
from experiments.forge.api import task_formulation_context,CapabilityError
from experiments.forge.contracts import read_json
from experiments.forge.state import state_digest


def trainer(enabled=False):
    recipe=get_recipe('bcap',num_particles=16,z_dim=2,batch_size=8,total_steps=4,
                      prior_kind='mog',standardize=False)
    with torch.random.fork_rng(devices=[]):
        g=nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,2))
        d=nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,1))
    init.deterministic_orthogonal_(g);init.deterministic_orthogonal_(d)
    prior=recipe.make_prior(sigma=.025,generator=torch.Generator().manual_seed(0))
    init.deterministic_orthogonal_(prior)
    return GANTrainer(recipe,g,d,prior=prior,role_motion_balance=enabled)


def test_inactive_constructor_checkpoint_and_streams_are_exact():
    a=trainer();b=trainer(False)
    real=torch.arange(16,dtype=torch.float32).reshape(8,2)/8
    for _ in range(2):a.step(real);b.step(real)
    assert state_digest(a.state_dict())==state_digest(b.state_dict())
    assert 'role_motion' not in a.state_dict()


def test_public_control_bounds_motion_retains_prior_and_critic_first_step():
    a=trainer();b=trainer(True)
    real=torch.arange(16,dtype=torch.float32).reshape(8,2)/8
    a.step(real);b.step(real)
    assert state_digest(a.prior.state_dict())==state_digest(b.prior.state_dict())
    assert state_digest(a.D.state_dict())==state_digest(b.D.state_dict())
    assert state_digest(a.opt_g.state_dict())==state_digest(b.opt_g.state_dict())
    assert state_digest(a.state_dict()['streams'])==state_digest(b.state_dict()['streams'])
    row=b.role_motion.summary
    assert row['updates']==1 and row['max_network_prior_ratio']<=1
    assert row['network_accepted_sum']<=row['prior_sum']
    assert b.prior.z.requires_grad and all(p.requires_grad for p in b.G.parameters())


def test_exact_checkpoint_continuation_and_fail_closed_counters():
    a=trainer(True);real=torch.ones(8,2)
    a.step(real);saved=a.state_dict();a.step(real);expected=state_digest(a.state_dict())
    b=trainer(True);b.load_state_dict(saved);b.step(real)
    assert state_digest(b.state_dict())==expected
    invalid=deepcopy(b.state_dict());invalid['role_motion']['summary']['max_network_prior_ratio']=2
    before=state_digest(b.state_dict())
    with pytest.raises(ValueError,match='role-motion'):b.load_state_dict(invalid)
    assert state_digest(b.state_dict())==before


def test_zero_response_restores_generator_exactly_without_prior_freeze():
    g=nn.Linear(1,1,bias=False)
    with torch.no_grad():g.weight.fill_(1)
    class Prior:pass
    prior=Prior();prior.z=nn.Parameter(torch.ones(2,1))
    class Step:
        def step(self):
            with torch.no_grad():g.weight.add_(1)
    before=g.weight.detach().clone();g.weight.grad=torch.ones_like(g.weight)
    control=RoleMotionBalance();control.step(Step(),g,prior,prior.z.detach(),torch.arange(2))
    assert torch.equal(g.weight,before) and prior.z.requires_grad
    assert control.summary['rejected']==1


def test_scale1_no_float_recomposition_and_units_equivariance():
    scales=[]
    for unit in (1.,10.):
        g=nn.Linear(1,1,bias=False)
        with torch.no_grad():g.weight.fill_(unit)
        class Prior:pass
        prior=Prior();prior.z=nn.Parameter(torch.ones(2,1))
        class Step:
            def step(self):
                with torch.no_grad():g.weight.add_(.01*unit);prior.z.add_(.2)
        g.weight.grad=torch.ones_like(g.weight);control=RoleMotionBalance()
        control.step(Step(),g,prior,prior.z.detach().clone(),torch.arange(2))
        expected=torch.tensor([[unit]],dtype=g.weight.dtype)+.01*unit
        assert torch.equal(g.weight,expected)
        scales.append(control.summary['scale_sum'])
    assert scales==[1.,1.]


def test_component_host_is_explicitly_blocked():
    task=read_json('configs/forge/tasks/two_pole.json')
    candidate=read_json('configs/forge/ideas/bcap-role-motion-balance-round4-v1.json')
    with pytest.raises(CapabilityError,match='role-motion consumer'):
        task_formulation_context(candidate,task)

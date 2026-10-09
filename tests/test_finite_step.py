"""Finite acceptance, replay and public trainer checkpoint software checks."""
from copy import deepcopy
import pytest
import torch
from torch import nn
from particlegan import GANTrainer, get_recipe, init
from particlegan.finite_step import FiniteStep
from experiments.forge.state import state_digest


def test_downhill_proposal_can_overshoot_and_is_shrunk():
    model=nn.Linear(1,1,bias=False).double()
    with torch.no_grad():model.weight.fill_(1.)
    optimizer=torch.optim.SGD(model.parameters(),lr=2.)
    loss=model.weight.square().sum()
    replay=FiniteStep.capture((model,))
    loss.backward()
    controller=FiniteStep()
    result=controller.apply(optimizer,loss,lambda:model.weight.square().sum(),replay,(model,))
    assert result['slope']<0 and result['full_loss']>result['before_loss']
    assert result['accepted'] and result['scale']==.25
    assert model.weight.item()==0
    assert controller.stats['downhill_loss_increases']==1


def test_replayed_randomness_and_buffers_advance_once():
    model=nn.Sequential(nn.Linear(2,2,bias=False),nn.BatchNorm1d(2),nn.Dropout(.5))
    torch.manual_seed(0)
    x=torch.tensor([[1.,0.],[0.,1.],[2.,1.],[1.,2.]])
    optimizer=torch.optim.SGD(model.parameters(),lr=10.)
    replay=FiniteStep.capture((model,))
    objective=lambda:model(x).square().mean()
    loss=objective();loss.backward()
    after=FiniteStep.capture((model,))
    result=FiniteStep().apply(optimizer,loss,objective,replay,(model,))
    assert result['accepted']
    assert torch.equal(torch.get_rng_state(),after['cpu'])
    assert all(torch.equal(b,v) for b,v in after['buffers'])
    assert model[1].num_batches_tracked==1


def test_exception_rolls_back_parameters_history_and_replay():
    model=nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters(),lr=1.)
    replay=FiniteStep.capture((model,));loss=model(torch.ones(1,1)).square().sum();loss.backward()
    before=state_digest(dict(model=model.state_dict(),opt=optimizer.state_dict()))
    rng=torch.get_rng_state().clone()
    def error():torch.rand(1);raise RuntimeError('probe')
    with pytest.raises(RuntimeError,match='probe'):
        FiniteStep().apply(optimizer,loss,error,replay,(model,))
    assert before==state_digest(dict(model=model.state_dict(),opt=optimizer.state_dict()))
    assert torch.equal(rng,torch.get_rng_state())


def trainer(mode):
    recipe=get_recipe('bcap',num_particles=16,batch_size=8,z_dim=2,total_steps=5,
                      prior_kind='mog',standardize=False,finite_step_mode=mode)
    g=nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,1))
    d=nn.Sequential(nn.Linear(1,8),nn.Tanh(),nn.Linear(8,1))
    init.deterministic_orthogonal_(g);init.deterministic_orthogonal_(d)
    prior=recipe.make_prior(sigma=.1)
    init.deterministic_orthogonal_(prior)
    return GANTrainer(recipe,g,d,prior=prior,seed=0)


def test_public_trainer_resume_exact_and_no_additional_stream_draws():
    torch.set_num_threads(1)
    active,inactive=trainer('armijo'),trainer('none')
    batches=[torch.linspace(1.,3.,8)[:,None],torch.linspace(1.5,2.5,8)[:,None]]
    active.step(batches[0]);inactive.step(batches[0])
    assert all(torch.equal(active.state_dict()['streams'][k],inactive.state_dict()['streams'][k])
               for k in active.state_dict()['streams'])
    saved=active.state_dict()
    active.step(batches[1]);expected=state_digest(active.state_dict())
    active.load_state_dict(saved);active.step(batches[1])
    assert expected==state_digest(active.state_dict())
    assert active.finite_step.stats['proposals']==2
    assert active.finite_step.stats['armijo_violations']==0


def test_checkpoint_rejects_corruption_before_model_mutation():
    active=trainer('armijo');active.step(torch.ones(8,1))
    saved=active.state_dict();bad=deepcopy(saved)
    bad['finite_step']['stats']['accepted']+=100
    before=state_digest(active.state_dict())
    with pytest.raises(ValueError,match='counters'):active.load_state_dict(bad)
    assert before==state_digest(active.state_dict())


def test_inactive_default_has_no_recipe_or_checkpoint_addition():
    inactive=trainer('none')
    assert 'finite_step_mode' not in inactive.recipe.to_dict()
    assert 'finite_step' not in inactive.state_dict()
    with pytest.raises(ValueError,match='requires'):
        get_recipe('bcap',finite_step_mode='armijo',prior_reg=1.)

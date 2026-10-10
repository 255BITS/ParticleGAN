from copy import deepcopy
import pytest
import torch
from particlegan import Recipe, GANTrainer, get_recipe, MoGParticlePrior
from particlegan.kinetic_backtrack import kinetic_backtrack
from particlegan.training import _normalized_recipe


def test_finite_backtrack_controls_overshoot_keeps_full_bits_and_exact_rejection():
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    before=[p.detach().clone()];gradient=[2*before[0]]
    with torch.no_grad():p.fill_(-2.)
    audit=kinetic_backtrack([p],before,gradient,lambda:p.square().sum(),1.)
    assert audit['scale']==.5 and audit['trials']==2
    assert p.item()==-.5 and audit['final']<=audit['bound']
    before=[p.detach().clone()]
    with torch.no_grad():p.fill_(-.25)
    proposed=p.detach().clone()
    audit=kinetic_backtrack([p],before,[2*before[0]],lambda:p.square().sum(),.25)
    assert audit['scale']==1 and torch.equal(p,proposed)
    before=[p.detach().clone()]
    with torch.no_grad():p.fill_(3.)
    audit=kinetic_backtrack([p],before,[torch.ones_like(p)],lambda:float('nan'),.0625)
    assert audit['rejected'] and audit['scale']==0 and torch.equal(p,before[0])


@pytest.mark.parametrize('standardize',[False,True])
@pytest.mark.parametrize('sigma',[0.,.025])
def test_mog_noise_payload_replays_consumed_draws_without_advancing_streams(standardize,sigma):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        prior=MoGParticlePrior(12,2,sigma=0.,standardize=standardize)
    prior.set_sigma(sigma)
    def streams():return torch.Generator().manual_seed(1),torch.Generator().manual_seed(2)
    a,b=streams();c,d=streams()
    latent,indices=prior.sample(6,generator=a,noise_generator=b)
    replay,rows,jitter=prior.sample(6,generator=c,noise_generator=d,return_noise=True)
    assert torch.equal(indices,rows) and torch.equal(latent,replay)
    assert torch.equal(a.get_state(),c.get_state()) and torch.equal(b.get_state(),d.get_state())
    assert torch.equal(prior.means()[rows]+jitter,replay)
    assert not jitter.requires_grad


def test_public_trainer_backtrack_joint_objective_streams_buffers_and_checkpoint():
    def build(enabled):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            r=get_recipe('bcap',num_particles=12,z_dim=2,batch_size=6,total_steps=2,
                input_noise_std=0.,output_noise_std=0.,kinetic_transport_weight=1.,
                kinetic_transport_local_weight=1.,kinetic_transport_backtrack=enabled)
            return GANTrainer(r,torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.BatchNorm1d(8),torch.nn.Tanh(),torch.nn.Linear(8,2)),torch.nn.Linear(2,1))
    a,b=build(False),build(True)
    real=torch.tensor([[-1.,-.2],[-.7,.2],[.6,.7],[1.1,-.5],[.3,.15],[1.7,.1]])
    x,y=a.step(real),b.step(real)
    audit=y['kinetic_backtrack'];assert audit['final']<=audit['bound']
    assert 'kinetic_backtrack' not in x
    assert audit['trials']<=8
    for key in a.D.state_dict():assert torch.equal(a.D.state_dict()[key],b.D.state_dict()[key])
    for key in a.state_dict()['streams']:assert torch.equal(a.state_dict()['streams'][key],b.state_dict()['streams'][key])
    from experiments.forge.state import state_digest
    assert state_digest(a.state_dict()['optimizers'])==state_digest(b.state_dict()['optimizers'])
    assert b.G[1].num_batches_tracked==a.G[1].num_batches_tracked==1
    checkpoint=deepcopy(b.state_dict());expected=b.step(real);saved=deepcopy(b.state_dict())
    restored=build(True);restored.load_state_dict(checkpoint);actual=restored.step(real)
    assert expected['kinetic_backtrack']==actual['kinetic_backtrack']
    for name,model in saved['models'].items():
        for key,value in model.items():
            if isinstance(value,torch.Tensor):assert torch.equal(value,restored.state_dict()['models'][name][key])
    assert _normalized_recipe({'kinetic_transport_backtrack':False})==_normalized_recipe({})
    with pytest.raises(ValueError):Recipe(kinetic_transport_backtrack=1)
    with pytest.raises(ValueError,match='zero additive'):Recipe(kinetic_transport_backtrack=True)

import pytest
import torch
from particlegan import Recipe, GANTrainer, get_recipe
from particlegan.init import initialize_
from particlegan.kinetic_transport import kinetic_transport_tail_loss
from particlegan.training import _normalized_recipe
from experiments.forge.api import FormulationContext, CapabilityError, task_policy_blockers


def test_tail_moment_null_units_permutation_rng_and_pullback():
    y=torch.tensor([[-1.3,.2],[-.7,-.4],[.6,.8],[1.1,-.5],[.3,.15],[1.7,.1]],dtype=torch.float64,requires_grad=True)
    x=(y.detach()+torch.tensor([.2,-.1])).requires_grad_(True)
    before=torch.get_rng_state().clone()
    v=kinetic_transport_tail_loss(x,y)
    assert v>0 and torch.equal(before,torch.get_rng_state())
    assert kinetic_transport_tail_loss(x.flip(0),y.roll(2,0)).item()==pytest.approx(v.item())
    assert kinetic_transport_tail_loss(7*x+11,7*y+11).item()==pytest.approx(v.item())
    null=y.detach().clone().requires_grad_(True)
    loss=kinetic_transport_tail_loss(null,y)
    assert loss.item()==0 and torch.equal(torch.autograd.grad(loss,null)[0],torch.zeros_like(null))
    assert y.grad is None
    assert torch.autograd.gradcheck(lambda f:kinetic_transport_tail_loss(f,y.detach()),(x,))


def test_tail_contamination_unbounded_loss_and_inward_force():
    real=torch.tensor([[-1.],[-.7],[-.2],[.3],[.8],[1.]],dtype=torch.float64)
    values=[]
    for radius in (10.,20.,40.):
        fake=real.clone();fake[-1]=radius;fake.requires_grad_(True)
        loss=kinetic_transport_tail_loss(fake,real)
        force=-torch.autograd.grad(loss,fake)[0][-1].item()
        assert force<0
        values.append(loss.item())
    assert values[2]>8*values[1]>64*values[0]
    duplicate=torch.zeros(6,2,dtype=torch.float64,requires_grad=True)
    value=kinetic_transport_tail_loss(duplicate,duplicate.detach())
    assert value.item()==0 and torch.isfinite(torch.autograd.grad(value,duplicate)[0]).all()


def test_tail_recipe_validation_compatibility_and_component_refusal():
    assert _normalized_recipe({'kinetic_transport_tail_weight':0.})==_normalized_recipe({})
    for weight in (-1.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):Recipe(kinetic_transport_tail_weight=weight)
    with pytest.raises(ValueError,match='matching'):
        kinetic_transport_tail_loss(torch.ones(2,1),torch.ones(3,1))
    task={'id':'two_pole','execution':{'execution_path':'public_components'}}
    assert task_policy_blockers(task,{'recipe_overrides':{'kinetic_transport_tail_weight':1.}})
    with pytest.raises(CapabilityError,match='consume'):
        FormulationContext(recipe_overrides={'kinetic_transport_tail_weight':1.},execution_path='public_components')


def build(weight):
    with torch.random.fork_rng(devices=[]):
        recipe=get_recipe('bcap',num_particles=12,z_dim=2,batch_size=6,total_steps=2,
            input_noise_std=0.,output_noise_std=0.,kinetic_transport_weight=1.,
            kinetic_transport_local_weight=1.,kinetic_transport_tail_weight=weight)
        g,d=torch.nn.Linear(2,2),torch.nn.Linear(2,1)
        initialize_(g,method='xavier_uniform_zero_bias_v1',parameter_generators={'weight':torch.Generator().manual_seed(0)})
        initialize_(d,method='xavier_uniform_zero_bias_v1',parameter_generators={'weight':torch.Generator().manual_seed(1)})
        return GANTrainer(recipe,g,d,seed=0)


def test_public_trainer_consumes_tail_preserves_d_rng_and_exact_continuation():
    baseline,candidate=build(0.),build(1.)
    real=torch.tensor([[-1.,-.2],[-.7,.2],[.6,.7],[1.1,-.5],[.3,.15],[1.7,.1]])
    b,c=baseline.step(real),candidate.step(real)
    assert 'kinetic_transport_tail' not in b and c['kinetic_transport_tail']>0
    assert c['loss_g'].item()==pytest.approx((c['loss_gan']+c['kinetic_transport']+c['kinetic_transport_local']+c['kinetic_transport_tail']).item())
    assert all(torch.equal(v,candidate.D.state_dict()[k]) for k,v in baseline.D.state_dict().items())
    assert any(not torch.equal(a,z) for a,z in zip(baseline.G.parameters(),candidate.G.parameters()))
    a,z=baseline.state_dict(),candidate.state_dict()
    assert all(torch.equal(a['streams'][k],z['streams'][k]) for k in a['streams'])
    restored=build(1.);restored.load_state_dict(z)
    candidate.step(real);restored.step(real)
    from experiments.forge.state import state_digest
    assert state_digest(candidate.state_dict())==state_digest(restored.state_dict())


def test_round4_study_single_global_delta_and_genuine_blocker():
    from pathlib import Path
    from experiments.forge.planning import resolve_idea
    request=resolve_idea(Path(__file__).resolve().parents[1],'transport_tail_moments_r4',study='transport_tails_candidate_round4')
    assert request['study_review']['status']=='READY'
    binding=request['study_review']['actual_bindings']
    for arm in ('candidate','control'):assert binding[arm]['recipe']['two_pole']['status']=='BLOCKED'
    for task in ('gaussian1d_smoke','gaussian1d_stability','vector_unequal_mass','vector_unequal_width','vector_two_broad'):
        a,b=binding['candidate']['recipe'][task],binding['control']['recipe'][task]
        assert {k:(b[k],a[k]) for k in a if a[k]!=b[k]}=={'kinetic_transport_tail_weight':(0.,1.)}

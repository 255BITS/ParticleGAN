import pytest
import torch

from particlegan import Recipe
from particlegan.kinetic_transport import kinetic_transport_loss
from particlegan.training import _normalized_recipe
from experiments.forge.api import CapabilityError, FormulationContext, task_policy_blockers


def test_kinetic_transport_exact_1d_quantiles_and_force():
    # Too much left-hand mass: quantile transport pulls one left atom right.
    fake = torch.tensor([[-2.], [-1.], [0.], [2.]], dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[-2.], [0.], [1.], [2.]], dtype=torch.float64, requires_grad=True)
    value = kinetic_transport_loss(fake, real)
    assert value.item() == pytest.approx(.5 / 2.1875)
    gradient, = torch.autograd.grad(value, fake)
    assert gradient[1].item() < 0 and gradient[2].item() < 0
    assert gradient[0].item() == gradient[3].item() == 0
    assert real.grad is None


def test_kinetic_transport_permutation_scale_and_no_rng_draws():
    x = torch.tensor([[0., 1.], [2., -1.], [3., 4.], [-2., 2.]], dtype=torch.float64)
    y = torch.tensor([[1., -1.], [0., 2.], [4., 3.], [-1., 0.]], dtype=torch.float64)
    before = torch.get_rng_state().clone()
    original = kinetic_transport_loss(x, y)
    assert torch.equal(torch.get_rng_state(), before)
    assert kinetic_transport_loss(x[[2, 0, 3, 1]], y.flip(0)).item() == pytest.approx(original.item())
    assert kinetic_transport_loss(7*x+11, 7*y+11).item() == pytest.approx(original.item())
    assert kinetic_transport_loss(x, x).item() == 0
    # Avoid projected ties: sorting is differentiable only away from them.
    x = x + torch.tensor([[.013, .027], [.041, -.031], [-.019, .053], [.037, -.023]], dtype=x.dtype)
    assert torch.autograd.gradcheck(kinetic_transport_loss, (x.requires_grad_(), y))


def test_kinetic_transport_validation_default_checkpoint_and_component_refusal():
    for options in ({"kinetic_transport_weight": -1}, {"kinetic_transport_weight": float('nan')},
                    {"kinetic_transport_projections": 0}):
        with pytest.raises(ValueError):
            Recipe(**options)
    assert _normalized_recipe({"kinetic_transport_weight": 0., "kinetic_transport_projections": 32}) == _normalized_recipe({})
    task = {"id": "two_pole", "execution": {"execution_path": "public_components"}}
    assert task_policy_blockers(task, {"recipe_overrides": {"kinetic_transport_weight": 1.}})
    with pytest.raises(CapabilityError, match="consume"):
        FormulationContext(recipe_overrides={"kinetic_transport_weight": 1.}, execution_path="public_components")
    with pytest.raises(ValueError, match="matching"):
        kinetic_transport_loss(torch.ones(2, 1), torch.ones(3, 1))


def test_kinetic_transport_local_empirical_null_permutation_units_and_rng():
    from particlegan.kinetic_transport import kinetic_transport_local_loss
    y=torch.tensor([[-1.3,.2],[-.7,-.4],[.6,.8],[1.1,-.5],[.3,.15],[1.7,.1]],dtype=torch.float64,requires_grad=True)
    x=(y.detach()+torch.tensor([.2,-.1])).requires_grad_(True)
    before=torch.get_rng_state().clone()
    value=kinetic_transport_local_loss(x,y)
    assert value>0
    assert torch.equal(before,torch.get_rng_state())
    assert kinetic_transport_local_loss(x.flip(0),y.roll(2,0)).item()==pytest.approx(value.item())
    assert kinetic_transport_local_loss(7*x+11,7*y+11).item()==pytest.approx(value.item())
    null_x=y.detach().clone().requires_grad_(True)
    null=kinetic_transport_local_loss(null_x,y)
    assert null.item()==0
    assert torch.equal(torch.autograd.grad(null,null_x)[0],torch.zeros_like(null_x))
    assert y.grad is None
    # Check fake pullback only: real anchors are deliberately detached.
    assert torch.autograd.gradcheck(lambda f:kinetic_transport_local_loss(f,y.detach()),(x,))


def test_kinetic_transport_local_contracted_cloud_force_and_duplicate_floor():
    from particlegan.kinetic_transport import kinetic_transport_local_loss
    real=torch.tensor([[-1.],[-.6],[.6],[1.]],dtype=torch.float64)
    fake=torch.tensor([[-.06],[-.02],[.02],[.06]],dtype=torch.float64,requires_grad=True)
    grad=torch.autograd.grad(kinetic_transport_local_loss(fake,real),fake)[0]
    assert (grad*fake<0).all()  # Negative gradient expands this contracted symmetric cloud.
    repeated=torch.zeros(6,2,dtype=torch.float64)
    fake=repeated.clone().requires_grad_(True)
    value=kinetic_transport_local_loss(fake,repeated)
    assert value.item()==0
    assert torch.isfinite(torch.autograd.grad(value,fake)[0]).all()
    assert _normalized_recipe({'kinetic_transport_local_weight':0.})==_normalized_recipe({})
    for weight in (-1.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):Recipe(kinetic_transport_local_weight=weight)
    task={'id':'two_pole','execution':{'execution_path':'public_components'}}
    assert task_policy_blockers(task,{'recipe_overrides':{'kinetic_transport_local_weight':1.}})
    with pytest.raises(CapabilityError,match='consume'):
        FormulationContext(recipe_overrides={'kinetic_transport_local_weight':1.},execution_path='public_components')


def test_kinetic_transport_local_public_trainer_consumes_signal_preserves_streams_and_d():
    from particlegan import GANTrainer,get_recipe
    def build(weight):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            recipe=get_recipe('bcap',num_particles=12,z_dim=2,batch_size=6,total_steps=2,
                input_noise_std=0.,output_noise_std=0.,kinetic_transport_weight=1.,
                kinetic_transport_local_weight=weight)
            return GANTrainer(recipe,torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    baseline,candidate=build(0.),build(1.)
    real=torch.tensor([[-1.,-.2],[-.7,.2],[.6,.7],[1.1,-.5],[.3,.15],[1.7,.1]])
    b=baseline.step(real);c=candidate.step(real)
    assert 'kinetic_transport_local' not in b
    assert c['kinetic_transport_local'].item()>0
    assert c['loss_g'].item()==pytest.approx((c['loss_gan']+c['kinetic_transport']+c['kinetic_transport_local']).item())
    for key in baseline.D.state_dict():assert torch.equal(baseline.D.state_dict()[key],candidate.D.state_dict()[key])
    assert any(not torch.equal(a,z) for a,z in zip(baseline.G.parameters(),candidate.G.parameters()))
    a,z=baseline.state_dict(),candidate.state_dict()
    for key in a['streams']:assert torch.equal(a['streams'][key],z['streams'][key])


def test_kinetic_transport_round2_admits_supported_peers_with_blocked_primary_control():
    from pathlib import Path
    from experiments.forge.planning import resolve_idea
    root=Path(__file__).resolve().parents[1]
    request=resolve_idea(root,'kinetic_transport_local_v2',study='kinetic_transport_candidate_round2')
    assert request['study_review']['status']=='READY'
    assert request['tasks']['two_pole']['preflight_blockers']
    before=request['study_review']['actual_bindings']['control']
    after=request['study_review']['actual_bindings']['candidate']
    assert before['recipe']['two_pole']['status']==after['recipe']['two_pole']['status']=='BLOCKED'
    assert before['recipe']['vector_unequal_width']['kinetic_transport_local_weight']==0
    assert after['recipe']['vector_unequal_width']['kinetic_transport_local_weight']==1

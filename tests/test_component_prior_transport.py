"""Tiny software fixtures, distinct from full-budget quality comparisons."""
from pathlib import Path
import pytest
import torch

from particlegan import Recipe
from particlegan.training import _normalized_recipe
from experiments.forge.api import FormulationContext, CapabilityError
from experiments.forge.planning import resolve_idea
from experiments.forge.state import state_digest


def build(routing, weight=1.):
    context = FormulationContext(recipe_preset='bcap',seed=0,
        prior={'kind':'mog','sigma':.025,'standardize':False,'learnable':True},
        recipe_overrides={'num_particles':12,'z_dim':2,'batch_size':6,'total_steps':3,
            'input_noise_std':0.,'output_noise_std':0.,'ema_decay':0.,
            'optimizer_family':'dualnorm','optimizer_smoothing':.001,
            'kinetic_transport_weight':weight,'kinetic_transport_local_weight':weight,
            'kinetic_transport_prior_only':routing})
    g = context.construct(lambda:torch.nn.Linear(2,2),component='generator')
    d = context.construct(lambda:torch.nn.Linear(2,1),component='discriminator')
    return context.build_trainer(g,d,max_steps=3)


def test_transport_routing_exact_network_gradient_and_unchanged_prior_signal_rng():
    full, routed, adversarial = build(False), build(True), build(False,0.)
    real = torch.tensor([[-1.,-.2],[-.7,.2],[.6,.7],[1.1,-.5],[.3,.15],[1.7,.1]])
    gradients = []
    for trainer in (full,routed,adversarial):
        original = trainer.opt_g.step
        def capture(*args,trainer=trainer,original=original,**kwargs):
            gradients.append({'g':[p.grad.clone() for p in trainer.G.parameters()],
                              'prior':trainer.prior.z.grad.clone()})
            return original(*args,**kwargs)
        trainer.opt_g.step = capture
    values = [trainer.step(real) for trainer in (full,routed,adversarial)]
    # Independent gradient oracle: the public adversarial-only trainer computes
    # precisely the G gradient that routing must preserve at the matched state.
    assert all(torch.equal(a,b) for a,b in zip(gradients[1]['g'],gradients[2]['g']))
    assert any(not torch.equal(a,b) for a,b in zip(gradients[0]['g'],gradients[1]['g']))
    torch.testing.assert_close(gradients[0]['prior'],gradients[1]['prior'],atol=2e-7,rtol=2e-6)
    assert not torch.equal(gradients[1]['prior'],gradients[2]['prior'])
    assert values[0]['loss_g'] == values[1]['loss_g']
    assert values[1]['kinetic_transport_prior_only'] is True
    assert all(torch.equal(v,routed.D.state_dict()[k]) for k,v in full.D.state_dict().items())
    assert all(torch.equal(v,adversarial.G.state_dict()[k]) for k,v in routed.G.state_dict().items())
    a,b=full.state_dict(),routed.state_dict()
    assert all(torch.equal(a['streams'][k],b['streams'][k]) for k in a['streams'])
    restored=build(True);restored.load_state_dict(b)
    routed.step(real);restored.step(real)
    assert state_digest(routed.state_dict()) == state_digest(restored.state_dict())


@pytest.mark.parametrize('overrides',[
    {'kinetic_transport_prior_only':1,'kinetic_transport_weight':1.},
    {'kinetic_transport_prior_only':True},
    {'kinetic_transport_prior_only':True,'kinetic_transport_weight':1.,'kinetic_transport_tail_weight':1.},
    {'kinetic_transport_prior_only':True,'kinetic_transport_weight':1.,'kinetic_transport_backtrack':True},
])
def test_invalid_routing_recipes_refuse(overrides):
    with pytest.raises(ValueError):
        Recipe(**overrides)


def test_routing_preserves_archived_defaults_and_frozen_component_refusal():
    assert _normalized_recipe({'kinetic_transport_prior_only':False}) == _normalized_recipe({})
    with pytest.raises(CapabilityError,match='consume'):
        FormulationContext(recipe_overrides={'kinetic_transport_weight':1.,'kinetic_transport_prior_only':True},
                           execution_path='public_components')


def test_round5_ready_one_global_delta_and_original_gates_budgets():
    root=Path(__file__).resolve().parents[1]
    request=resolve_idea(root,'component_prior_transport_r5',study='component_tails_candidate_round5')
    assert request['study_review']['status']=='READY'
    bindings=request['study_review']['actual_bindings']
    for task in request['tasks']:
        a,b=bindings['candidate']['recipe'][task],bindings['control']['recipe'][task]
        assert {k:(b[k],a[k]) for k in a if a[k]!=b[k]} == {'kinetic_transport_prior_only':(False,True)}
    assert sum(t['resources']['timeout_seconds'] for t in request['tasks'].values()) == 9720
    assert request['tasks']['grid100']['execution']['steps']==7000
    assert request['tasks']['gaussian1d_stability']['dependencies']

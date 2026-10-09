from pathlib import Path
import json
import pytest
import torch
from particlegan import Recipe
from particlegan.conditional_transport import OutputMarginalTransport
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from experiments.forge.api import task_policy_blockers, FormulationContext, CapabilityError
ROOT=Path(__file__).resolve().parents[1]


def test_public_marginal_consumer_retains_inactive_tensor_and_rng():
    total=torch.tensor(2.,requires_grad=True)
    real=torch.arange(24.).reshape(6,4)
    fake=(real+.3).requires_grad_()
    conditioning=torch.arange(6.).unsqueeze(1).requires_grad_()
    consumer=OutputMarginalTransport(Recipe())
    before=torch.get_rng_state().clone()
    assert consumer.add(total,fake,real,conditioning=conditioning) is total
    assert torch.equal(before,torch.get_rng_state())
    assert consumer.state_dict()['active_calls']==0


def test_marginal_consumer_is_not_paired_identity_supervision():
    real=torch.tensor([[0.,0.],[1.,1.],[-1.,2.],[2.,-1.]],requires_grad=True)
    fake=real.detach().roll(1,0).requires_grad_()
    conditioning=torch.arange(4.).unsqueeze(1).requires_grad_()
    recipe=Recipe(kinetic_transport_weight=1.,kinetic_transport_local_weight=1.)
    consumer=OutputMarginalTransport(recipe)
    zero=fake.new_zeros(())
    value=consumer.add(zero,fake,real,conditioning=conditioning)
    assert value.item()==pytest.approx(0.,abs=1e-12)
    assert (fake-real).square().mean()>.02
    grads=torch.autograd.grad(value,(fake,real,conditioning),allow_unused=True)
    assert grads[1] is grads[2] is None
    assert torch.isfinite(grads[0]).all()
    shifted=(real.detach()+.1).requires_grad_()
    loss=consumer.add(zero,shifted,real,conditioning=conditioning)
    assert loss>0 and torch.autograd.grad(loss,shifted)[0].norm()>0
    assert consumer.state_dict()['active_calls']==2


def test_original_contracts_block_and_explicit_variants_admit():
    candidate={'recipe_overrides':{'kinetic_transport_weight':1.,'kinetic_transport_local_weight':1.}}
    for host in ('trajectory','residual_student','mid_scale_identity'):
        original=json.loads((ROOT/f'configs/forge/tasks/{host}.json').read_text())
        variant=json.loads((ROOT/f'configs/forge/tasks/{host}_transport_round5_v1.json').read_text())
        assert task_policy_blockers(original,candidate)
        assert not task_policy_blockers(variant,candidate)
        bad={**variant,'execution':{**variant['execution'],'host':'two_pole'}}
        assert task_policy_blockers(bad,candidate)
        assert original['evaluation']['thresholds']==variant['evaluation']['thresholds']
        assert original['resources']==variant['resources']
    with pytest.raises(CapabilityError,match='consume'):
        FormulationContext(recipe_overrides=candidate['recipe_overrides'],execution_path='public_components')
    context=FormulationContext(recipe_overrides=candidate['recipe_overrides'],execution_path='public_components',component_transport='output_marginal_v1')
    assert context.component_transport=='output_marginal_v1'


def test_exact_direction_blend_uses_no_finite_evaluator_and_restores_stats():
    p=torch.nn.Parameter(torch.tensor([1.,.5]))
    opt=DirectionBlendOptimizer([p],family='dualnorm',smoothing=.001,lr=.1,momentum=0.)
    original=p.detach().clone()
    protected=p[0]
    total=-p[0]+p[1]
    def forbidden():raise AssertionError('direction-only called finite evaluator')
    constraint_geometry_backward(total,opt,(protected,),protected_evaluator=forbidden)
    opt.step()
    assert p[0]<=original[0]
    assert opt.direction_blend_stats['conflict_steps']==1
    state=opt.state_dict()
    q=torch.nn.Parameter(p.detach().clone())
    restored=DirectionBlendOptimizer([q],family='dualnorm',smoothing=.001,lr=.1,momentum=0.)
    restored.load_state_dict(state)
    assert restored.direction_blend_stats==opt.direction_blend_stats

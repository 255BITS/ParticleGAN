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

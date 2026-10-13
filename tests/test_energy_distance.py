"""Software-only raw energy-U and signed consumer contracts; no qualification."""
from copy import deepcopy

import pytest
import torch

from particlegan import Recipe
from particlegan.energy_distance import OutputEnergyU, energy_distance_u


def test_raw_signed_u_excludes_indices_not_duplicate_values_and_detaches_target():
    fake = torch.tensor([[-1.], [1.]], dtype=torch.float64, requires_grad=True)
    real = fake.detach().clone().requires_grad_(True)
    value = energy_distance_u(fake, real)
    assert value.item() == -2.
    gradient, target = torch.autograd.grad(value, (fake, real), allow_unused=True)
    assert target is None
    torch.testing.assert_close(gradient, fake.new_tensor([[.5], [-.5]]))
    duplicate = fake.detach().repeat_interleave(2, dim=0)
    assert energy_distance_u(duplicate, duplicate).item() == pytest.approx(-2 / 3)
    assert energy_distance_u(3 * duplicate, 3 * duplicate).item() == pytest.approx(-2.)


def test_first_derivative_unequal_panels_permutation_and_no_rng():
    fake = torch.tensor([[-.73, .21], [.19, -.61], [.82, .48], [-.14, .92]],
                        dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[-.42, -.83], [.64, -.19], [.08, .57]], dtype=torch.float64)
    before = torch.get_rng_state().clone()
    assert torch.autograd.gradcheck(lambda x: energy_distance_u(x, real), (fake,))
    gradient, = torch.autograd.grad(energy_distance_u(fake, real), fake)
    order = torch.tensor([2, 0, 3, 1])
    reordered = fake.detach()[order].requires_grad_(True)
    value = energy_distance_u(reordered, real.flip(0))
    torch.testing.assert_close(value, energy_distance_u(fake, real))
    torch.testing.assert_close(torch.autograd.grad(value, reordered)[0], gradient[order])
    assert torch.equal(before, torch.get_rng_state())


def test_collision_subgradient_zero_and_symmetric_pole_force_zero():
    fake = torch.zeros(12, 1, dtype=torch.float64, requires_grad=True)
    collapsed = energy_distance_u(fake, fake.detach())
    assert collapsed.item() == 0
    assert torch.equal(torch.autograd.grad(collapsed, fake)[0], torch.zeros_like(fake))
    offsets = torch.linspace(-.05, .05, 6, dtype=torch.float64)
    target = torch.cat((-1 + offsets, 1 + offsets)).unsqueeze(1)
    gradient, = torch.autograd.grad(energy_distance_u(fake, target), fake)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-17)


@pytest.mark.parametrize("weight", [True, -1., float("nan"), float("inf")])
def test_invalid_recipe_weights_rejected(weight):
    with pytest.raises(ValueError, match="energy_distance_weight"):
        Recipe(energy_distance_weight=weight)


@pytest.mark.parametrize("bad", [torch.zeros(1, 2), torch.zeros(3, 0),
                                    torch.full((3, 2), float("nan")), torch.zeros(3, 2, dtype=torch.int64)])
def test_invalid_output_panels_rejected(bad):
    with pytest.raises(ValueError, match="energy U"):
        energy_distance_u(bad, bad)


def test_signed_consumer_resume_and_condition_exclusion():
    recipe = Recipe(energy_distance_weight=1.)
    consumer = OutputEnergyU(recipe)
    fake = torch.tensor([[-1.], [1.]], requires_grad=True)
    value = consumer.add(fake.new_zeros(()), fake, fake.detach(), conditioning=torch.tensor([[100.], [-200.]]))
    assert value.item() == consumer.loss_sum == consumer.maximum_loss == consumer.minimum_loss == -2.
    value.backward()
    assert torch.isfinite(fake.grad).all()
    restored = OutputEnergyU(recipe)
    restored.load_state_dict(consumer.state_dict())
    assert restored.state_dict() == consumer.state_dict()
    restored.add(value.detach(), fake, fake.detach())
    assert restored.calls == restored.active_calls == 2 and restored.loss_sum == -4.


@pytest.mark.parametrize("field,value", [("weight", .5), ("consumer", "output_marginal_v1"),
    ("schema_version", True), ("loss_sum", float("nan")), ("loss_sum", 9.),
    ("calls", -1), ("active_calls", 0), ("minimum_loss", 4.)])
def test_corrupt_signed_state_rejected_atomically(field, value):
    consumer = OutputEnergyU(Recipe(energy_distance_weight=1.))
    panel = torch.tensor([[-1.], [1.]])
    consumer.add(torch.tensor(0.), panel, panel)
    original = consumer.state_dict()
    bad = deepcopy(original)
    bad[field] = value
    with pytest.raises(ValueError, match="checkpoint"):
        consumer.load_state_dict(bad)
    assert consumer.state_dict() == original


def test_disabled_consumer_and_recipe_packet_keep_archived_meaning(monkeypatch):
    def forbidden(*args):
        raise AssertionError("disabled energy helper was consumed")
    monkeypatch.setattr(Recipe, "energy_distance_loss", forbidden)
    recipe = Recipe()
    assert "energy_distance_weight" not in recipe.to_dict()
    total = torch.tensor(3.)
    consumer = OutputEnergyU(recipe)
    assert consumer.add(total, None, None) is total
    assert consumer.calls == 0
    assert Recipe(energy_distance_weight=1.).to_dict()["energy_distance_weight"] == 1.

"""Analytical checks for the opt-in output energy objective and public consumer."""
import copy

import pytest
import torch
from torch import nn

from particlegan.allocation_repair import energy_distance


def test_energy_uses_cross_and_ordered_off_diagonal_self_pairs():
    fake = torch.tensor([[0.], [2.]], dtype=torch.float64)
    real = torch.tensor([[1.], [3.], [5.]], dtype=torch.float64)
    cross = 14. / 6.
    fake_self = 2.
    real_self = 16. / 6.
    real_rms = (48. / 6.) ** .5
    expected = (cross - .5 * fake_self - .5 * real_self) / real_rms
    assert energy_distance(fake, real).item() == pytest.approx(expected, abs=1e-14)


def test_energy_is_permutation_rotation_translation_and_positive_scale_invariant():
    fake = torch.tensor([[0., 1.], [2., 0.], [3., 4.]], dtype=torch.float64)
    real = torch.tensor([[2., 3.], [-1., 2.], [4., 0.]], dtype=torch.float64)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    before = energy_distance(fake, real)
    after = energy_distance(7 * fake.flip(0) @ rotation + 19,
                            7 * real.roll(1, 0) @ rotation + 19)
    assert torch.allclose(before, after, atol=1e-14, rtol=1e-14)
    assert torch.equal(before, energy_distance(fake.reshape(3, 1, 2), real.reshape(3, 1, 2)))


def test_constant_target_has_unit_scale_no_epsilon_amplification_and_no_target_gradient():
    fake = torch.tensor([[1.], [1.]], dtype=torch.float64, requires_grad=True)
    real = torch.zeros(2, 1, dtype=torch.float64, requires_grad=True)
    loss = energy_distance(fake, real)
    grad_fake, grad_real = torch.autograd.grad(loss, (fake, real), allow_unused=True)
    assert loss.item() == 1.
    assert torch.equal(grad_fake, torch.full_like(fake, .5))
    assert grad_real is None


def test_energy_supplies_long_range_force_toward_missing_population():
    # Incorrect 50:50 generated mass versus a 75:25 target. Ordered distinct
    # generated samples prevent a coincident-particle subgradient from hiding
    # the transport force. The lowest ranks in the surplus right population
    # must move left; the objective sees no component labels.
    left = torch.linspace(-1.02, -.98, 50, dtype=torch.float64)
    right = torch.linspace(.98, 1.02, 50, dtype=torch.float64)
    fake = torch.cat((left, right))[:, None].requires_grad_()
    real = torch.cat((torch.linspace(-1.02, -.98, 75, dtype=torch.float64),
                      torch.linspace(.98, 1.02, 25, dtype=torch.float64)))[:, None]
    gradient, = torch.autograd.grad(energy_distance(fake, real), fake)
    assert bool((gradient[50:70] > 0).all())
    moved = fake.detach() - .01 * gradient
    assert energy_distance(moved, real) < energy_distance(fake, real)


def test_coincident_samples_have_finite_gradient_and_do_not_consume_rng():
    fake = torch.zeros(8, 2, dtype=torch.float64, requires_grad=True)
    real = torch.zeros(8, 2, dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    loss = energy_distance(fake, real)
    grad, = torch.autograd.grad(loss, fake)
    assert loss == 0 and bool(torch.isfinite(grad).all())
    assert torch.equal(grad, torch.zeros_like(grad))
    assert torch.equal(before, torch.random.get_rng_state())


def test_energy_finite_difference_gradient():
    fake = torch.tensor([[-1.2, .1], [.2, 1.1], [1.3, -.3]], dtype=torch.float64, requires_grad=True)
    real = torch.tensor([[-.9, .2], [.5, .9], [1.5, -.2]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: energy_distance(x, real), (fake,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("fake,real", [
    (torch.zeros(1, 2), torch.zeros(2, 2)),
    (torch.zeros(2, 2), torch.zeros(2, 3)),
    (torch.zeros(2), torch.zeros(2)),
    (torch.zeros(2, 0), torch.zeros(2, 0)),
    (torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.float64)),
    (torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long)),
])
def test_energy_rejects_incompatible_batches(fake, real):
    with pytest.raises(ValueError, match="matching floating sample shapes"):
        energy_distance(fake, real)


def _public_trainer(weight):
    from particlegan import GANTrainer, get_recipe, init
    recipe = get_recipe("bispf", num_particles=16, z_dim=2, batch_size=8,
                        total_steps=4, output_energy_weight=weight)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        generator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 2))
        critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1))
        prior = recipe.make_prior()
        init.deterministic_orthogonal_(generator, seed=0)
        init.deterministic_orthogonal_(critic, seed=1)
        init.deterministic_orthogonal_(prior, seed=2)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


def test_public_trainer_energy_is_opt_in_uses_existing_batches_and_resumes_exactly():
    real = torch.linspace(-1., 1., 16).reshape(8, 2)
    baseline, candidate = _public_trainer(0.), _public_trainer(1.)
    before = copy.deepcopy(candidate.state_dict())
    plain = baseline.step(real, generator_real=real.flip(0))
    active = candidate.step(real, generator_real=real.flip(0))
    assert "output_energy" not in plain
    assert torch.isfinite(active["output_energy"])
    for key, value in baseline.state_dict()["streams"].items():
        assert torch.equal(value, candidate.state_dict()["streams"][key]), key
    assert any(not torch.equal(value, candidate.state_dict()["models"]["G"][key])
               for key, value in baseline.state_dict()["models"]["G"].items())
    checkpoint = copy.deepcopy(candidate.state_dict())
    uninterrupted = candidate.step(real)
    resumed = _public_trainer(1.)
    resumed.load_state_dict(checkpoint)
    repeated = resumed.step(real)
    assert torch.equal(uninterrupted["loss_g"], repeated["loss_g"])
    assert torch.equal(uninterrupted["output_energy"], repeated["output_energy"])
    assert any(not torch.equal(value, candidate.state_dict()["models"]["G"][key])
               for key, value in before["models"]["G"].items())


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_public_recipe_rejects_invalid_energy_weight(weight):
    from particlegan import get_recipe
    with pytest.raises(ValueError, match="output_energy_weight"):
        get_recipe("bispf", output_energy_weight=weight)

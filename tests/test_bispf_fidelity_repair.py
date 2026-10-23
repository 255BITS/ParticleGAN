"""Numerical controls for the proposed same-batch quartic output mechanism."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan.fidelity_repair import sliced_quartic_transport


def test_scalar_empirical_w4_and_real_detach():
    real = torch.tensor([[-1.], [1.]], dtype=torch.float64, requires_grad=True)
    fake = torch.tensor([[2.], [-1.]], dtype=torch.float64, requires_grad=True)
    loss = sliced_quartic_transport(fake, real)
    assert loss.item() == pytest.approx(.5, abs=1e-12)
    loss.backward()
    torch.testing.assert_close(fake.grad, torch.tensor([[2.], [0.]], dtype=torch.float64))
    assert real.grad is None


@pytest.mark.parametrize("shape", [(9, 1), (9, 2), (9, 1, 3, 3)])
def test_permuted_identical_batch_has_zero_loss_and_finite_zero_force(shape):
    real = torch.arange(torch.tensor(shape).prod(), dtype=torch.float64).reshape(shape) / 10
    fake = real.flip(0).clone().requires_grad_()
    loss = sliced_quartic_transport(fake, real)
    loss.backward()
    assert loss.item() < 1e-50
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().max() < 1e-40


@pytest.mark.parametrize("shape", [(12, 1), (12, 2), (12, 1, 3, 3)])
def test_translation_positive_scale_and_permutation_invariance(shape):
    real = torch.arange(torch.tensor(shape).prod(), dtype=torch.float64).reshape(shape) / 9
    fake = real.square() / 3 + .4
    expected = sliced_quartic_transport(fake, real)
    actual = sliced_quartic_transport((fake.flip(0) + 7) * 2.5, (real.roll(2, 0) + 7) * 2.5)
    torch.testing.assert_close(expected, actual, atol=1e-10, rtol=1e-11)


def test_remote_contamination_has_quartic_cost_and_cubic_restoring_force():
    real = torch.tensor([[-1.], [0.], [0.], [1.]], dtype=torch.float64)
    losses, forces = [], []
    for displacement in (1., 2.):
        fake = real.clone()
        fake[-1] += displacement
        fake.requires_grad_()
        loss = sliced_quartic_transport(fake, real)
        loss.backward()
        losses.append(loss.item())
        forces.append(fake.grad[-1].item())
    assert losses[1] / losses[0] == pytest.approx(16)
    assert forces[1] / forces[0] == pytest.approx(8)
    assert forces[0] > 0  # Gradient descent moves the remote sample back.


def test_constant_target_uses_unit_scale_and_consumes_no_rng():
    before = torch.random.get_rng_state().clone()
    real = torch.zeros(8, 1, 2, 2, requires_grad=True)
    fake = torch.ones_like(real, requires_grad=True)
    loss = sliced_quartic_transport(fake, real)
    loss.backward()
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert 0 < loss.item() <= 16  # No epsilon-denominator force amplification.
    assert torch.isfinite(fake.grad).all() and fake.grad.sum() > 0
    assert real.grad is None


def test_coincident_constant_batches_have_finite_zero_gradient():
    real = torch.zeros(6, 2)
    fake = real.clone().requires_grad_()
    loss = sliced_quartic_transport(fake, real)
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(fake.grad).all()
    assert torch.count_nonzero(fake.grad) == 0


def test_sorting_gradient_matches_finite_difference_away_from_ties():
    real = torch.tensor([[-2., 1.], [-.4, -.2], [.5, 1.2], [2., -1.]], dtype=torch.float64)
    fake = torch.tensor([[-1.413, .723], [-.817, -.319], [.637, .811], [2.107, -.751]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: sliced_quartic_transport(x, real), (fake,))


@pytest.mark.parametrize("fake,real", [
    (torch.zeros(2), torch.zeros(2)),
    (torch.zeros(1, 2), torch.zeros(1, 2)),
    (torch.zeros(3, 2), torch.zeros(3, 3)),
    (torch.zeros(3, 0), torch.zeros(3, 0)),
    (torch.zeros(3, 2), torch.zeros(3, 2, dtype=torch.float64)),
    (torch.zeros(3, 2, dtype=torch.long), torch.zeros(3, 2, dtype=torch.long)),
])
def test_invalid_input_is_explicit(fake, real):
    with pytest.raises(ValueError, match="quartic transport"):
        sliced_quartic_transport(fake, real)


def _fidelity_public_trainer(weight, *, image=False):
    """Small explicit software fixture, not a substituted quality-test host."""
    from particlegan import GANTrainer, get_recipe, init
    recipe = get_recipe("bispf", num_particles=16, z_dim=2, batch_size=8,
                        total_steps=4, fidelity_transport_weight=weight,
                        prior_kind="mog", sigma_rel=.025, standardize=False)
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(torch.Generator().manual_seed(0).get_state())
        width = 8 if image else 1
        generator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, width),
                                  nn.Unflatten(1, (1, 2, 4)) if image else nn.Identity())
        critic = nn.Sequential(nn.Flatten(), nn.Linear(width, 8), nn.LeakyReLU(.2), nn.Linear(8, 1))
        prior = recipe.make_prior()
        for index, module in enumerate((generator, critic, prior)):
            init.deterministic_orthogonal_(module, seed=index)
        streams = {name: torch.Generator().manual_seed(index + 10) for index, name in enumerate((
            "latent_generator", "penalty_generator", "noise_generator", "input_noise_generator",
            "prior_noise_generator", "eval_generator", "model_generator"))}
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0, **streams)


def _assert_fidelity_state_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_fidelity_state_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_fidelity_state_exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("image", [False, True])
def test_public_fidelity_consumer_uses_existing_g_batches_and_adds_weight_once(monkeypatch, image):
    import particlegan.fidelity_repair as mechanism
    trainer = _fidelity_public_trainer(.25, image=image)
    shape = (8, 1, 2, 4) if image else (8, 1)
    real = torch.linspace(-1., 1., 64 if image else 8).reshape(shape)
    generator_real = real.flip(0) + .3
    observations, generated, targets = [], [], []
    handle = trainer.G.register_forward_hook(lambda module, args, output: generated.append(output))

    def observe(fake, target):
        value = sliced_quartic_transport(fake, target)
        observations.append((fake, target, value.detach()))
        return value

    monkeypatch.setattr(mechanism, "sliced_quartic_transport", observe)
    result = trainer.step(real, generator_real=lambda: targets.append(1) or generator_real)
    handle.remove()
    assert targets == [1] and len(observations) == 1 and len(generated) == 2
    assert observations[0][0] is generated[-1]
    assert torch.equal(observations[0][1], generator_real)
    assert observations[0][1].data_ptr() == generator_real.data_ptr()
    assert torch.equal(result["fidelity_transport"], .25 * observations[0][2])
    assert torch.equal(result["loss_g"], result["loss_gan"]
                       + trainer.recipe.prior_reg * result["prior_regularization"]
                       + result["fidelity_transport"])
    assert not result["fidelity_transport"].requires_grad


@pytest.mark.parametrize("image", [False, True])
def test_public_fidelity_disabled_path_skips_mechanism_and_keeps_consumed_streams(monkeypatch, image):
    import particlegan.fidelity_repair as mechanism
    baseline, candidate = (_fidelity_public_trainer(weight, image=image) for weight in (0., 1.))
    shape = (8, 1, 2, 4) if image else (8, 1)
    real = torch.linspace(-1., 1., 64 if image else 8).reshape(shape)
    initial = baseline.state_dict()
    _assert_fidelity_state_exact(initial["models"], candidate.state_dict()["models"])
    with monkeypatch.context() as patch:
        def forbidden(*args):
            raise AssertionError("zero weight must not evaluate the auxiliary objective")
        patch.setattr(mechanism, "sliced_quartic_transport", forbidden)
        plain = baseline.step(real)
    active = candidate.step(real)
    assert "fidelity_transport" not in plain and torch.isfinite(active["fidelity_transport"])
    _assert_fidelity_state_exact(baseline.state_dict()["streams"], candidate.state_dict()["streams"])
    assert any(not torch.equal(value, candidate.G.state_dict()[key])
               for key, value in baseline.G.state_dict().items())


@pytest.mark.parametrize("image", [False, True])
def test_public_fidelity_checkpoint_restores_complete_state_for_exact_continuation(image):
    trainer = _fidelity_public_trainer(1., image=image)
    shape = (8, 1, 2, 4) if image else (8, 1)
    real = torch.linspace(-1., 1., 64 if image else 8).reshape(shape)
    trainer.step(real)
    checkpoint = deepcopy(trainer.state_dict())
    expected_result = trainer.step(real, generator_real=real.flip(0) + .2)
    expected_state = deepcopy(trainer.state_dict())
    resumed = _fidelity_public_trainer(1., image=image)
    resumed.load_state_dict(checkpoint)
    actual_result = resumed.step(real, generator_real=real.flip(0) + .2)
    _assert_fidelity_state_exact(expected_result, actual_result)
    _assert_fidelity_state_exact(expected_state, resumed.state_dict())


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf"), True])
def test_public_recipe_rejects_invalid_fidelity_weight(weight):
    from particlegan import get_recipe
    with pytest.raises(ValueError, match="fidelity_transport_weight"):
        get_recipe("bispf", fidelity_transport_weight=weight)

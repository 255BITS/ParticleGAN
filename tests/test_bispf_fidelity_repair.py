"""Numerical controls for the proposed same-batch quartic output mechanism."""
import pytest
import torch

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

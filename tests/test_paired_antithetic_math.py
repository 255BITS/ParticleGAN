"""Pure RpGAN estimator identities; these do not qualify a trained model."""

import torch

from particlegan import get_recipe


def objective(error, noise):
    loss = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=16).make_loss()
    # A quadratic critic isolates the stochastic force. Its sign is chosen for
    # this mathematical probe, not imposed on the toy's learned critic.
    real = -noise.square().sum(-1)
    fake = -(noise + error).square().sum(-1)
    return loss.g_loss(fake, real.detach())


def gaussian():
    return 1.3 * torch.randn(16, 4, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(1729))


def test_quadratic_noise_force_cancels_at_exact_paired_fit():
    noise = gaussian()
    error = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    ordinary = torch.autograd.grad(objective(error, noise), error)[0]
    paired = torch.autograd.grad(
        (objective(error, noise) + objective(error, -noise)) / 2, error)[0]
    torch.testing.assert_close(ordinary, noise.mean(0), rtol=1e-12, atol=1e-12)
    assert ordinary.norm() > .1
    torch.testing.assert_close(paired, torch.zeros_like(paired), rtol=0, atol=1e-12)


def test_antithetic_retains_restoring_force_and_expected_gan_objective():
    noise = gaussian()
    error = torch.full((4,), .04, dtype=torch.float64, requires_grad=True)
    paired_loss = (objective(error, noise) + objective(error, -noise)) / 2
    marginal_loss = objective(error, torch.cat((noise, -noise)))
    paired_gradient = torch.autograd.grad(paired_loss, error)[0]
    marginal_gradient = torch.autograd.grad(marginal_loss, error)[0]
    torch.testing.assert_close(paired_loss, marginal_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(paired_gradient, marginal_gradient, rtol=1e-12, atol=1e-12)
    assert paired_gradient.dot(error) > 0


def test_local_gan_curvature_is_preserved_by_antithetic_pairing():
    noise = gaussian()
    error = torch.zeros(4, dtype=torch.float64)
    ordinary = torch.autograd.functional.hessian(lambda e: objective(e, noise), error)
    paired = torch.autograd.functional.hessian(
        lambda e: (objective(e, noise) + objective(e, -noise)) / 2, error)
    expected = torch.eye(4, dtype=torch.float64) + noise.T @ noise / len(noise)
    torch.testing.assert_close(ordinary, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(paired, expected, rtol=1e-12, atol=1e-12)

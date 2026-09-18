"""Validate the added DDGAN path and its reconstruction gradient."""
import torch
from experiments.train_cifar_particle_ddgan import DEFAULTS, validate, state_hash
from lib.image_particle_ddgan import ParticleDDGenerator, ParticleDDDiscriminator
from lib.image_particle_autoencoder import ImageRoutingEncoder
from particlegan import MoGParticlePrior


def test_encoder_reconstruction_gradients_and_fixed_sigma():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    g = ParticleDDGenerator(16, 8, DEFAULTS['alpha_bar'])
    e = ImageRoutingEncoder(16, 8)
    p = MoGParticlePrior(num_particles=32, z_dim=16, sigma_rel=.025)
    sigma = p.sigma.clone()
    x = torch.randn(4, 3, 32, 32).tanh()
    t = torch.arange(1, 5)
    _, xt = g.schedule.forward_pair(x, t)
    z, _, _, _ = e(x, p.means(), p.sigma, .125)
    y = g(z, xt, t)
    (y-x).square().mean().backward()
    for value in [e.query.weight, e.offset.weight, p.z, g.net.embed[0].weight]:
        assert value.grad is not None and value.grad.isfinite().all() and value.grad.abs().sum() > 0
    assert torch.equal(p.sigma, sigma) and not p.sigma.requires_grad
    z = z.detach()
    torch.testing.assert_close(y[:1], g(z[:1], xt[:1], t[:1]), rtol=2e-5, atol=2e-6)
    out = g.sample(p, 3, torch.Generator().manual_seed(17))
    assert out.shape == (3, 3, 32, 32) and out.isfinite().all()


def test_ddgan_frozen_critic_condition_and_second_derivative():
    torch.set_num_threads(2)
    d = ParticleDDDiscriminator(8, 16, DEFAULTS['alpha_bar']).train().requires_grad_(True)
    before = state_hash([d.critic.features])
    xt = torch.randn(4, 3, 32, 32)
    t = torch.arange(1, 5)
    x = torch.randn_like(xt, requires_grad=True)
    conditioned = d.conditioned(xt, t)
    score = conditioned(x)
    torch.testing.assert_close(score, d.critic(x, torch.zeros_like(t), xt, t)[0])
    gradient = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
    (score.mean() + gradient.square().mean()).backward()
    assert gradient.abs().sum() > 0
    assert all(p.grad is None for p in d.critic.features.parameters())
    assert state_hash([d.critic.features]) == before
    for model in ['direct', 'ddgan']:
        for arm in ['gan', 'bounded']:
            validate({**DEFAULTS, 'model': model, 'arm': arm})

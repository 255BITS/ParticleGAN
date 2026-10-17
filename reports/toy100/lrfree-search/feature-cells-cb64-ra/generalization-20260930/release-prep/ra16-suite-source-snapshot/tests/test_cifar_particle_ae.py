"""Scientific invariants of the direct particle image experiment."""
import torch

from experiments.train_cifar_particle_ae import DEFAULTS, state_hash, validate
from lib.image_particle_autoencoder import route, DirectGenerator, DirectDiscriminator, ImageRoutingEncoder


def test_hard_forward_and_separate_query_particle_gradients():
    q = torch.tensor([[.9, .1]], requires_grad=True)
    means = torch.tensor([[0., 0.], [1., 0.], [0., 2.]], requires_grad=True)
    offset = torch.zeros_like(q, requires_grad=True)
    z, ids, u, soft = route(q, means, .2, offset, .125)
    assert ids.tolist() == [1]
    assert torch.equal(z, means[[1]])
    z.sum().backward()
    assert q.grad.abs().sum() > 0
    assert torch.equal(means.grad, torch.tensor([[0., 0.], [1., 1.], [0., 0.]]))
    torch.testing.assert_close(offset.grad, torch.full_like(offset, .2))
    torch.testing.assert_close(soft.sum(1), torch.ones(1))


def test_distance_selection_matches_explicit_distances_and_bounds():
    torch.manual_seed(4)
    q, means = torch.randn(7, 16), torch.randn(33, 16)
    offset = torch.randn_like(q) * 20
    z, ids, u, _ = route(q, means, .1, offset, .125)
    assert torch.equal(ids, (q[:, None] - means[None]).square().sum(-1).argmin(1))
    assert u.abs().max() <= 3
    torch.testing.assert_close(z, means[ids] + .1 * u)


def test_direct_generator_and_encoder_have_no_batch_information_path():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    g, e = DirectGenerator(16, 8), ImageRoutingEncoder(16, 8)
    z = torch.randn(3, 16)
    x = g(z)
    assert x.shape == (3, 3, 32, 32)
    torch.testing.assert_close(x[:1], g(z[:1]), atol=2e-6, rtol=2e-5)
    means = torch.randn(20, 16)
    code, ids, u, _ = e(x, means, .1, .125)
    torch.testing.assert_close(code[:1], e(x[:1], means, .1, .125)[0], atol=2e-6, rtol=2e-5)
    assert torch.equal(u, torch.zeros_like(u))
    assert torch.equal(code, means[ids])


def test_feature_critic_freezes_weights_and_bn_but_keeps_input_gradient():
    torch.set_num_threads(2)
    d = DirectDiscriminator(8).train().requires_grad_(True)
    before = state_hash([d.critic.features])
    assert not any(p.requires_grad for p in d.critic.features.parameters())
    assert not any(m.training for m in d.critic.features.modules())
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    score = d(x)
    grad = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
    (grad.square().sum() + score.mean()).backward()
    assert grad.abs().sum() > 0
    assert all(p.grad is None for p in d.critic.features.parameters())
    assert d.critic.pixel.output.weight.grad is not None
    assert state_hash([d.critic.features]) == before
    d.requires_grad_(False).requires_grad_(True)
    assert not any(p.requires_grad for p in d.critic.features.parameters())


def test_full_and_pilot_configs_validate():
    validate(DEFAULTS)
    validate({**DEFAULTS, 'arm': 'bounded', 'steps': 200, 'final_samples': 0, 'eval_samples': 0})

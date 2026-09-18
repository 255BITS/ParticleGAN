"""Protect the hard routing and fixed-sigma contracts used by the scout."""
import torch

from experiments.train_mog_autoencoder import RoutingEncoder, routing_probabilities, usage_balance
from particlegan import MoGParticlePrior


def test_hard_forward_and_reconstruction_gradient_paths():
    torch.manual_seed(3)
    encoder = RoutingEncoder(width=16)
    means = torch.tensor([[-1., -1.], [1., 1.], [-1., 1.]], requires_grad=True)
    sigma = torch.tensor(.02)
    x = torch.tensor([[-2., -1.], [2., 1.]])
    z, ids, offset = encoder(x, means, sigma, "route_offset", torch.zeros_like(x))
    torch.testing.assert_close(z, means[ids] + sigma * offset)
    (z - x).square().mean().backward()
    assert means.grad[ids.unique()].abs().sum() > 0
    assert torch.equal(means.grad[2], torch.zeros(2))
    final_grad = encoder.net.net[-1].weight.grad
    assert final_grad[:2].abs().sum() > 0, "choice surrogate must reach encoder"
    assert final_grad[2:].abs().sum() > 0, "offset must receive reconstruction gradients"


def test_random_offset_is_input_noise_and_bounded_offset_stays_local():
    encoder = RoutingEncoder(width=16)
    means = torch.randn(8, 2)
    x, noise = torch.randn(12, 2), torch.randn(12, 2)
    _, _, u = encoder(x, means, torch.tensor(.02), "route_noise", noise)
    torch.testing.assert_close(u, noise)
    with torch.no_grad():
        encoder.net.net[-1].bias[2:] = torch.tensor([100., -100.])
    _, _, u = encoder(x, means, torch.tensor(.02), "route_bounded", noise)
    assert u.abs().max() <= 3


def test_reconstruction_step_moves_particles_but_keeps_sigma_fixed():
    torch.manual_seed(4)
    prior = MoGParticlePrior(num_particles=16, z_dim=2)
    encoder = RoutingEncoder(width=16)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(prior.parameters()), lr=.01)
    initial_sigma, initial_particles = prior.sigma.clone(), prior.z.detach().clone()
    x = torch.randn(32, 2)
    z, _, _ = encoder(x, prior.means(), prior.sigma, "route_offset", torch.randn_like(x))
    (z - x).square().mean().backward()
    opt.step()
    assert torch.equal(prior.sigma, initial_sigma)
    assert not torch.equal(prior.z, initial_particles)
    assert "sigma" not in dict(prior.named_parameters())


def test_zero_offset_ignores_noise_and_offset_head_but_keeps_routing_gradients():
    encoder = RoutingEncoder(width=16)
    with torch.no_grad():
        encoder.net.net[-1].bias[2:] = 100.
    means = torch.tensor([[-1., -1.], [1., 1.], [-1., 1.]], requires_grad=True)
    x = torch.tensor([[-2., -1.], [2., 1.]])
    z, ids, u = encoder(x, means, torch.tensor(.02), "route_zero", torch.randn_like(x))
    assert torch.equal(z, means[ids])
    assert torch.count_nonzero(u) == 0
    (z - x).square().mean().backward()
    assert means.grad[ids.unique()].abs().sum() > 0
    grad = encoder.net.net[-1].weight.grad
    assert grad[:2].abs().sum() > 0
    assert torch.count_nonzero(grad[2:]) == 0


def test_gradient_amplification_preserves_forward_and_other_direct_gradients():
    torch.manual_seed(7)
    encoder = RoutingEncoder(width=16)
    with torch.no_grad():
        encoder.net.net[-1].weight.normal_(std=.1)
    means = torch.tensor([[-1., -1.], [1., 1.], [-1., 1.]], requires_grad=True)
    decoder = torch.nn.Linear(2, 2)
    x = torch.tensor([[-2., -1.], [2., 1.]])
    sigma, noise = torch.tensor(.02), torch.randn_like(x)
    outputs, gradients = [], []
    params = (encoder.net.net[-1].weight, means, decoder.weight)
    for arm in ("route_offset", "route_grad100"):
        z, ids, u = encoder(x, means, sigma, arm, noise)
        outputs.append((z, ids, u))
        gradients.append(torch.autograd.grad((decoder(z) - x).square().mean(), params))
    for original, amplified in zip(*outputs):
        assert torch.equal(original, amplified), "forward values must be identical"
    base, amplified = gradients
    torch.testing.assert_close(amplified[0][:2], base[0][:2])
    assert base[0][2:].abs().sum() > 0
    torch.testing.assert_close(amplified[0][2:], 100 * base[0][2:])
    torch.testing.assert_close(amplified[1], base[1])
    torch.testing.assert_close(amplified[2], base[2])


def test_usage_balance_counts_hard_choices_and_pushes_toward_unused_particle():
    logits = torch.zeros(4, 2, requires_grad=True)
    loss = usage_balance(torch.zeros(4, dtype=torch.long), logits.softmax(1))
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert (logits.grad[:, 0] > 0).all()
    assert (logits.grad[:, 1] < 0).all()
    balanced = usage_balance(torch.tensor([0, 1, 0, 1]), logits.softmax(1))
    assert balanced.item() == 0


def test_balancing_preserves_bounded_forward_and_only_directly_trains_query():
    encoder = RoutingEncoder(width=16)
    means = torch.tensor([[-1., -1.], [1., 1.]], requires_grad=True)
    x = torch.tensor([[-2., -1.], [-1., -2.]])
    noise, sigma = torch.randn_like(x), torch.tensor(.02)
    original = encoder(x, means, sigma, "route_bounded", noise)
    z, ids, u, soft = encoder(x, means, sigma, "route_balanced", noise, return_routing=True)
    for a, b in zip(original, (z, ids, u)):
        assert torch.equal(a, b)
    usage_balance(ids, soft).backward()
    assert means.grad is None
    grad = encoder.net.net[-1].weight.grad
    assert grad[:2].abs().sum() > 0
    assert torch.count_nonzero(grad[2:]) == 0


def test_local_surrogate_preserves_hard_forward_bounds_and_gradient_paths():
    torch.manual_seed(19)
    encoder = RoutingEncoder(width=16)
    with torch.no_grad():
        encoder.net.net[-1].weight.normal_(std=.1)
        encoder.net.net[-1].bias[2:] = 10.
    means = torch.randn(16, 2, requires_grad=True)
    x, noise, sigma = torch.randn(32, 2), torch.randn(32, 2), torch.tensor(.02)
    original = encoder(x, means, sigma, "route_bounded", noise)
    for arm in ("route_local", "route_local_balanced"):
        z, ids, u, soft = encoder(x, means, sigma, arm, noise, return_routing=True)
        for a, b in zip(original, (z, ids, u)):
            assert torch.equal(a, b)
        assert u.abs().max() <= 3
        assert ((soft > 0).sum(1) == 8).all()
        torch.testing.assert_close(soft.sum(1), torch.ones(x.shape[0]))
        gradients = torch.autograd.grad((z - x).square().mean(),
                                        (encoder.net.net[-1].weight, means), retain_graph=True)
        assert gradients[0][:2].abs().sum() > 0
        assert gradients[0][2:].abs().sum() > 0
        assert gradients[1].abs().sum() > 0
        balance_grads = torch.autograd.grad(usage_balance(ids, soft),
                                            (encoder.net.net[-1].weight, means), allow_unused=True)
        assert balance_grads[0][:2].abs().sum() > 0
        assert torch.count_nonzero(balance_grads[0][2:]) == 0
        assert balance_grads[1] is None


def test_local_surrogate_excludes_distant_particles_and_handles_ties():
    distances = torch.arange(12, dtype=torch.float32)[None].requires_grad_()
    soft = routing_probabilities(distances, .25, local=True)
    assert torch.count_nonzero(soft[:, 8:]) == 0
    grad, = torch.autograd.grad(soft[:, 0].sum(), (distances,))
    assert torch.count_nonzero(grad[:, 8:]) == 0
    assert grad[:, :8].abs().sum() > 0
    # The detached bandwidth preserves invariance to a common distance shift.
    torch.testing.assert_close(routing_probabilities(distances + 100, .25, local=True), soft)
    tied = torch.zeros(2, 12, requires_grad=True)
    weights = routing_probabilities(tied, .25, local=True)
    gradients, = torch.autograd.grad((weights * torch.arange(12)).sum(), (tied,))
    assert torch.isfinite(weights).all() and torch.isfinite(gradients).all()

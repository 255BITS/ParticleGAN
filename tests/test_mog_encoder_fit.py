import torch

from experiments.train_mog_autoencoder import RoutingEncoder
from experiments.train_mog_encoder_fit import fit_loss, query, state_hash
from particlegan import MoGParticlePrior


def test_frozen_hash_covers_prior_buffers_and_extra_state():
    prior = MoGParticlePrior(num_particles=8, z_dim=2)
    original = state_hash(prior)
    prior.standardize = not prior.standardize
    assert state_hash(prior) != original
    prior.standardize = not prior.standardize
    assert state_hash(prior) == original
    prior.sigma.add_(.001)
    assert state_hash(prior) != original


def test_query_reproduces_hard_routing_and_oracle_loss_targets_decoded_best():
    encoder = RoutingEncoder(width=8)
    x = torch.tensor([[2., 0.]])
    means = torch.tensor([[-1., 0.], [1., 0.]])
    # Reverse data/latent ordering: the best decoded center is NOT nearest to X.
    decoder = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    with torch.no_grad():
        decoder.weight.copy_(-2*torch.eye(2))
    decoded = decoder(means)
    _, ids, _ = encoder(x, means, torch.tensor(.02), "route_zero", torch.zeros_like(x))
    assert ids.item() == ((query(encoder, x)[:, None]-means).square().sum(-1)).argmin(1).item() == 1
    loss = fit_loss("oracle_query", encoder, decoder, means, torch.tensor(.02), decoded, x, .25)
    torch.testing.assert_close(loss, (query(encoder, x)-means[0]).square().mean())
    loss.backward()
    assert encoder.net.net[-1].bias.grad[0] > 0  # descent moves query toward -1


def test_both_losses_update_encoder_without_updating_decoder_or_offset_head():
    for arm in ("oracle_query", "recon_st"):
        encoder = RoutingEncoder(width=8)
        decoder = torch.nn.Linear(2, 2).requires_grad_(False)
        means = torch.tensor([[-1., 0.], [1., 0.]])
        x = torch.tensor([[2., .5], [-2., -.5]])
        decoded = decoder(means).detach()
        decoder_hash, encoder_hash = state_hash(decoder), state_hash(encoder)
        opt = torch.optim.Adam(encoder.parameters(), lr=.0006, betas=(0., .999))
        fit_loss(arm, encoder, decoder, means, torch.tensor(.02), decoded, x, .25).backward()
        grad = encoder.net.net[-1].weight.grad
        assert grad[:2].abs().sum() > 0 and torch.count_nonzero(grad[2:]) == 0
        assert all(p.grad is None for p in decoder.parameters())
        opt.step()
        assert state_hash(decoder) == decoder_hash
        assert state_hash(encoder) != encoder_hash

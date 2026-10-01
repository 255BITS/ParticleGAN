"""Public encoding math, gradient estimators, recipe integration and contracts."""
import itertools
import math

import pytest
import torch
from torch import nn

from particlegan import MoGParticlePrior, ParticleEncoding, Recipe, get_recipe, particle_ae, particle_vae


def prior():
    p = MoGParticlePrior(sigma=.025, num_particles=3, z_dim=2, standardize=False, dtype=torch.float64)
    with torch.no_grad():
        p.z.copy_(torch.tensor([[-1., 0.], [0., 1.], [1., 0.]]))
        p.sigma.fill_(.1)
    return p


def test_ae_forward_and_selected_center_gradient():
    p = prior()
    q = torch.tensor([[.8, .1]], dtype=torch.float64, requires_grad=True)
    offset = torch.ones_like(q, requires_grad=True)
    encoded = particle_ae(q, offset, p)
    assert encoded.indices.tolist() == [[2]]
    torch.testing.assert_close(encoded.codes[:, 0], p.z[[2]] + .3 * torch.tanh(offset / 3))
    encoded.codes.sum().backward()
    assert q.grad.abs().sum() > 0 and offset.grad.abs().sum() > 0
    torch.testing.assert_close(p.z.grad, torch.tensor([[0., 0.], [0., 0.], [1., 1.]], dtype=p.z.dtype))
    assert encoded.log_probs is None
    with pytest.raises(ValueError, match='no variational'):
        encoded.negative_elbo(encoded.codes, q)


@pytest.mark.parametrize('hard', [False, True])
def test_posterior_rng_kl_and_local_noise(hard):
    p = prior()
    q = torch.zeros(256, 2, dtype=torch.float64)
    state = torch.random.get_rng_state()
    a = particle_vae(q, p, temperature=.7, draws=16, hard=hard, generator=torch.Generator().manual_seed(9))
    b = particle_vae(q, p, temperature=.7, draws=16, hard=hard, generator=torch.Generator().manual_seed(9))
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(a.codes, b.codes)
    eps = (a.codes - p.means()[a.indices]) / p.sigma
    assert abs(float(eps.detach().mean())) < .05
    assert abs(float(eps.detach().std()) - 1) < .05
    if hard:
        torch.testing.assert_close(a.kl, q.new_full((256,), math.log(3)))
        assert (a.log_probs.exp().sum(1) == 1).all()
        assert (a.log_probs.exp().max(1).values == 1).all()
        assert (a.indices == a.indices[:, :1]).all()
    else:
        torch.testing.assert_close(a.kl, torch.zeros_like(a.kl), atol=1e-12, rtol=0)


def test_score_gradient_matches_exact_enumeration():
    # Sum every possible independent two-draw outcome; includes pathwise cost
    # gradients and the leave-one-out score gradient, compared with exact E[c].
    logits = torch.tensor([[.3, -.2, .8]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([.2, .8, -1.2], dtype=torch.float64, requires_grad=True)
    logq = logits.log_softmax(1)
    probabilities = logq.exp()
    exact = (probabilities * values.square()).sum()
    expected_gradient = torch.autograd.grad(exact, (logits, values), retain_graph=True)
    estimate = 0
    for i, j in itertools.product(range(3), repeat=2):
        ids = torch.tensor([[i, j]])
        result = ParticleEncoding(torch.zeros(1, 2, 1), ids, torch.zeros(1), logq, 'categorical')
        prediction = values[ids][..., None]
        estimate = estimate + (probabilities[0, i] * probabilities[0, j]).detach() * result.reconstruction_loss(prediction, torch.zeros(1, 1))
    actual = torch.autograd.grad(estimate, (logits, values))
    for a, b in zip(actual, expected_gradient):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(estimate, exact)


def test_gaussian_elbo_dimensions_and_constant_kl():
    p = prior()
    result = particle_vae(torch.ones(2, 2, dtype=torch.float64), p, hard=True)
    target = torch.zeros(2, 3, 2)
    prediction = torch.ones(2, 2, 3, 2)
    tau = .2
    torch.testing.assert_close(result.negative_elbo(prediction, target, observation_sigma=tau),
                               result.kl.mean() + 6 / (2 * tau**2) + 3 * math.log(2 * math.pi * tau**2))
    torch.testing.assert_close(result.reconstruction_loss(prediction, target),
                               prediction.new_tensor(1.))


@pytest.mark.parametrize('name', ['ae_gan', 'vae_gan', 'ae_ddgan'])
def test_recipe_roundtrip_and_optimizer_encoder(name):
    recipe = get_recipe(name, num_particles=3, z_dim=2)
    assert Recipe(**recipe.to_dict()) == recipe
    p = recipe.make_prior()
    e, g, d = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 1)
    # A shared module appears only once even if included in G and encoder.
    opt, _ = recipe.make_optimizers(nn.ModuleList([g, e]), d, p, encoder=e)
    ids = [id(v) for group in opt.param_groups for v in group['params']]
    assert len(ids) == len(set(ids)) == 5
    q = e(torch.ones(4, 2))
    encoded = recipe.encode(q, p, offset=q if recipe.encoder_mode == 'ae' else None)
    out = g(encoded.codes)
    loss = encoded.reconstruction_loss(out, torch.zeros(4, 2))
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in e.parameters())
    assert p.z.grad is not None
    assert p.sigma.grad is None


def test_validation_and_explicit_components():
    p = prior()
    q = torch.zeros(2, 2, dtype=torch.float64)
    for kwargs in [{'draws': 0}, {'temperature': 0}, {'distance_reduction': 'bad'}]:
        with pytest.raises(ValueError):
            particle_vae(q, p, **kwargs)
    with pytest.raises(ValueError, match='at least two'):
        particle_vae(q, p, draws=1, hard=False).reconstruction_loss(torch.zeros(2, 1, 2), q)
    with pytest.raises(ValueError, match='offset'):
        get_recipe(prior_kind='mog', sigma_rel=0.025, encoder_mode='hard').encode(q, p, offset=q)
    with pytest.raises(ValueError, match='requires offset'):
        get_recipe(prior_kind='mog', sigma_rel=0.025, encoder_mode='ae').encode(q, p)
    with pytest.raises(ValueError):
        get_recipe(encoder_mode='ae', prior_kind='particles', sigma_rel=0)
    assert get_recipe().total_steps == 7000
    assert get_recipe(model='ddgan', prior_kind='mog', sigma_rel=.025).total_steps == 7000


def test_default_vae_has_no_kl_training_penalty():
    recipe = get_recipe(prior_kind='mog', sigma_rel=0.025, encoder_mode='hard')
    assert recipe.encoder_mode == 'hard'
    p = prior()
    encoded = recipe.encode(torch.ones(2, 2, dtype=torch.float64), p)
    prediction = encoded.codes.square()
    target = torch.zeros(2, 2, dtype=torch.float64)
    expected = (prediction - target[:, None]).square().mean()
    torch.testing.assert_close(encoded.reconstruction_loss(prediction, target), expected)
    assert encoded.kl.tolist() == [math.log(3)] * 2

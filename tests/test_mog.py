import copy
import numpy as np
import torch
from particlegan.particle_prior import ParticlePrior, MoGParticlePrior
from lib.mog_metrics import allocation_null, component_metrics, sample_metrics


def test_zero_noise_is_exact_and_rng_neutral():
    torch.manual_seed(13)
    old = ParticlePrior(100, 4)
    state = torch.get_rng_state().clone()
    torch.manual_seed(13)
    new = MoGParticlePrior(100, 4, standardize=False)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(old.z, new.z)
    for fixed in (False, True):
        a = torch.Generator().manual_seed(77)
        b = torch.Generator().manual_seed(77)
        x, i = old.sample(80, a, fixed_first_n=fixed)
        y, j = new.sample(80, b, fixed_first_n=fixed)
        assert torch.equal(x, y) and torch.equal(i, j)
        assert torch.equal(a.get_state(), b.get_state())


def test_noise_standardization_gradients_and_ema():
    prior = MoGParticlePrior(100, 4, sigma_rel=.125)
    eps = torch.randn(20, 4)
    z, ids = prior.sample(20, fixed_first_n=True, eps=eps)
    assert torch.equal(z, prior.means()[:20] + prior.sigma*eps)
    z.square().sum().backward()
    assert torch.isfinite(prior.z.grad).all()
    assert set(dict(prior.named_parameters())) == {'z'}
    sigma = prior.sigma.clone()
    before = prior.means().detach().clone()
    with torch.no_grad():
        prior.z.mul_(3).add_(7)
    assert torch.allclose(prior.means(), before, atol=5e-6)
    assert torch.equal(prior.sigma, sigma)
    ema = copy.deepcopy(prior)
    with torch.no_grad():
        ema.z[:10].add_(1)
    assert not torch.equal(prior.means(), ema.means())
    assert torch.allclose(ema.means().mean(0), torch.zeros(4), atol=1e-6)


def test_real_width_and_atom_width():
    from lib.toy_models import sample_100gaussians
    x = sample_100gaussians(200000, torch.device('cpu'), generator=torch.Generator().manual_seed(8))
    m, _, _ = sample_metrics(x)
    assert .985 < m['hq'] < .992 and .98 < m['width'] < 1.02
    assert m['modes'] == 100 and m['kl_balance'] < .001
    atoms = torch.cartesian_prod(torch.arange(10)-4.5, torch.arange(10)-4.5).repeat_interleave(100,0)
    a, nearest, good = sample_metrics(atoms)
    assert a['width'] == 0 and a['kl_balance'] == 0 and a['modes'] == 100
    c, details = component_metrics(nearest, nearest, good, 100, detail=True)
    assert c['purity_mean'] == 1 and c['alloc_empty'] == 0
    assert details['alloc'] == [1]*100
    null = allocation_null(100)
    assert abs(null['empty_modes'] - 100*.99**100) < 1


def test_baseline_pass_keeps_coverage_width_and_balance_guards():
    from lib.mog_metrics import pass_metrics
    criteria = dict(hq_ratio_min=.998, width_ratio_min=.79, width_ratio_max=1.21, kl_balance_max=.038)
    baseline = dict(modes=100, hq_ratio=1., width_ratio=.85, kl_balance=.035)
    assert pass_metrics(baseline, criteria) == dict(passed=True, passed_strict=False, passed_baseline=True)
    for changes in ({'modes':99}, {'hq_ratio':.99}, {'width_ratio':.78}, {'width_ratio':1.22}, {'kl_balance':.04}, {'kl_balance':None}):
        assert not pass_metrics({**baseline, **changes}, criteria)['passed']
    assert not pass_metrics(baseline)['passed']

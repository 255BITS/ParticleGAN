"""Architecture, isolation, initialization and exact resume for wider deconv G with hidden GroupNorm."""
import copy
import hashlib
import json
import os

import pytest
import torch

from experiments import train_cifar_ae_scaling as old
from experiments import train_cifar_ae_deconv_wide_norm as new


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_generator_structure_output_and_latent_gradients():
    torch.manual_seed(12)
    g = new.DeconvWideNormGenerator()
    assert sum(p.numel() for p in g.parameters()) == 925763
    layers = list(g.modules())
    norms = [m for m in layers if isinstance(m, torch.nn.GroupNorm)]
    assert [(m.num_groups, m.num_channels) for m in norms] == [(8, 256), (8, 128), (8, 64)]
    assert not any(isinstance(m, (torch.nn.modules.batchnorm._NormBase,
                                  torch.nn.LayerNorm)) for m in layers)
    assert isinstance(g.output[-2], torch.nn.ConvTranspose2d)
    assert isinstance(g.output[-1], torch.nn.Tanh)
    assert sum(isinstance(m, torch.nn.Linear) for m in layers) == 1
    convs = [m for m in layers if isinstance(m, torch.nn.ConvTranspose2d)]
    assert [(m.in_channels, m.out_channels) for m in convs] == [(256, 128), (128, 64), (64, 3)]
    assert all(m.kernel_size == (4, 4) and m.stride == (2, 2) and m.padding == (1, 1) for m in convs)
    z = torch.randn(3, 64, requires_grad=True)
    y = g(z)
    assert y.shape == (3, 3, 32, 32) and torch.isfinite(y).all()
    assert y.min() >= -1 and y.max() <= 1
    y.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and (z.grad.abs().sum(1) > 0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())
    metadata = new.generator_metadata(new.DEFAULTS)
    assert metadata['normalization'] == 'GroupNorm'
    assert metadata['normalized_channels'] == [256, 128, 64]
    assert metadata['output_normalization'] is None
    assert not any('transformer' in key for key in metadata)


def test_encoder_only_reconstruction_updates_only_encoder():
    cfg = {**new.DEFAULTS, 'z_dim': 8, 'width': 8, 'recon_grad': 'encoder_only'}
    g = new.DeconvWideNormGenerator(8)
    e = new.build_encoder(cfg)
    prior = new.MoGParticlePrior(16, 8)
    before = new.state_hash([g, prior])
    encoder_before = new.state_hash([e])
    optimizer = torch.optim.Adam([*g.parameters(), *e.parameters(), *prior.parameters()], lr=.001)
    loss = new.reconstruction_loss(g, e, prior, torch.randn(2, 3, 32, 32), cfg)
    loss.backward()
    assert all(p.grad is None for p in g.parameters()) and prior.z.grad is None
    assert all(p.requires_grad for p in g.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters())
    optimizer.step()
    assert before == new.state_hash([g, prior])
    assert encoder_before != new.state_hash([e])


def test_discriminator_encoder_and_global_rng_match_original():
    cfg = {**new.DEFAULTS, 'z_dim': 8, 'width': 8}
    torch.manual_seed(24002)
    _, old_d, old_e = old.build_models({**cfg, 'generator_arch': 'cnn'})
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(24002)
    g, d, e = new.build_models(cfg)
    assert isinstance(g, new.DeconvWideNormGenerator)
    assert new.state_hash([d, e]) == new.state_hash([old_d, old_e])
    assert torch.equal(torch.get_rng_state(), old_rng)


@pytest.mark.parametrize('value', [-.1, float('inf'), float('nan')])
def test_invalid_fixed_sigma(value):
    with pytest.raises(ValueError, match='fixed_sigma'):
        new.validate({**new.DEFAULTS, 'fixed_sigma': value})


@pytest.mark.parametrize('overrides', [{'g_width': 64}, {'g_depth': 2}])
def test_reject_ignored_generator_overrides(overrides):
    with pytest.raises(ValueError, match='overrides'):
        new.validate({**new.DEFAULTS, **overrides})


def test_sigma_override_metadata_and_default():
    prior = new.MoGParticlePrior(16, 8)
    calibrated = prior.sigma.clone()
    before = new.state_hash([prior])
    default = new.configure_sigma(prior, new.DEFAULTS)
    assert default['mode'] == 'calibrated' and default['actual_sigma'] == float(calibrated)
    assert before == new.state_hash([prior])
    metadata = new.configure_sigma(prior, {**new.DEFAULTS, 'fixed_sigma': .212616428732872})
    assert metadata['mode'] == 'fixed_override' and metadata['calibrated_sigma'] == float(calibrated)
    assert metadata['actual_sigma'] == float(torch.tensor(.212616428732872))
    assert before != new.state_hash([prior])
    assert torch.equal(copy.deepcopy(prior).sigma, prior.sigma)
    noiseless = new.MoGParticlePrior(16, 8, sigma_rel=0.)
    assert not noiseless._noise_enabled
    new.configure_sigma(noiseless, {**new.DEFAULTS, 'fixed_sigma': .2})
    assert noiseless._noise_enabled


def compare_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_state(x, y)
    else:
        assert a == b


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA test')
def test_full_state_split_resume_and_immutable_sigma(tmp_path, monkeypatch):
    factory = new.build_models
    def deterministic(cfg):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        g, d, e = factory(cfg)
        # Deterministic replacement avoids CUDA adaptive pooling backward variance.
        for head, kernel in zip(d.critic.project, (4, 2, 1)):
            head[5] = torch.nn.AvgPool2d(kernel)
        return g, d, e
    monkeypatch.setattr(new, 'build_models', deterministic)
    cfg = {**new.DEFAULTS, 'recon_grad': 'encoder_only', 'arm': 'bounded',
           'z_dim': 8, 'width': 8, 'num_particles': 16, 'fixed_sigma': .212616428732872,
           'batch_size': 2, 'steps': 4, 'reg_every': 2, 'eval_interval': 2,
           'log_interval': 2, 'eval_samples': 0, 'final_samples': 0,
           'recon_samples': 4, 'eval_batch_size': 4}
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        new.train({**cfg, 'out_dir': str(tmp_path / 'whole')})
        new.train({**cfg, 'steps': 2, 'out_dir': str(tmp_path / 'part')})
        parent = tmp_path / 'part/checkpoint.pt'
        resumed = {**cfg, 'out_dir': str(tmp_path / 'resumed'), 'resume_checkpoint': str(parent),
                   'resume_sha256': hashlib.sha256(parent.read_bytes()).hexdigest()}
        with pytest.raises(ValueError, match='resume cannot change'):
            new.load_resume({**resumed, 'fixed_sigma': .3})
        with pytest.raises(ValueError, match='resume cannot change'):
            new.load_resume({**resumed, 'generator_arch': 'deconv'})
        new.train(resumed)
    finally:
        torch.use_deterministic_algorithms(previous)
    a, b = [torch.load(tmp_path / name / 'checkpoint.pt', map_location='cpu', weights_only=False)
            for name in ('whole', 'resumed')]
    for key in ('G', 'D', 'E', 'prior', 'ema_G', 'ema_E', 'ema_prior',
                'optimizer_g', 'optimizer_d', 'rng', 'torch_rng', 'cuda_rng'):
        compare_state(a[key], b[key])
    assert a['initialization_sha256'] == b['initialization_sha256']
    assert float(a['prior']['sigma']) == float(torch.tensor(cfg['fixed_sigma']))
    summary = json.loads((tmp_path / 'resumed/summary.json').read_text())
    assert summary['sigma_unchanged'] and summary['frozen_features_unchanged']
    assert summary['metadata']['sigma_configuration']['actual_sigma'] == float(a['prior']['sigma'])
    assert summary['metadata']['resume']['interventions'] == {}


def test_historical_deconv_construction_unchanged():
    from experiments import train_cifar_ae_deconv as original
    cfg = {**new.DEFAULTS, 'generator_arch': 'deconv', 'width': 8, 'z_dim': 8}
    torch.manual_seed(24002)
    old_modules = original.build_models(cfg)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(24002)
    new_modules = new.build_models(cfg)
    assert new.state_hash(old_modules) == new.state_hash(new_modules)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert original.generator_metadata(cfg) == new.generator_metadata(cfg)

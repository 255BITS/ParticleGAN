"""Active spatial attention, immutable baseline initialization and real CUDA resume."""
import copy
import hashlib
import json
import os

import pytest
import torch

from experiments import train_cifar_ae_deconv_wide_norm as baseline
from experiments import train_cifar_ae_sagan as new


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_attention_is_active_at_initialization_and_has_all_gradients():
    torch.manual_seed(12)
    attention = new.SAGANAttention(64)
    assert sum(p.numel() for p in attention.parameters()) == 5120
    assert set(dict(attention.named_parameters())) == {
        'query.weight', 'key.weight', 'value.weight', 'project.weight'}
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    y = attention(x)
    assert y.shape == x.shape and not torch.equal(x, y)
    y.square().mean().backward()
    for parameter in attention.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_base_parameters_rng_and_metadata_are_preserved():
    cfg = {**new.DEFAULTS, 'z_dim': 8, 'width': 8}
    torch.manual_seed(24002)
    original = baseline.build_models({**cfg, 'generator_arch': 'deconv_wide_norm'})
    previous_rng = torch.get_rng_state().clone()
    torch.manual_seed(24002)
    models = new.build_models(cfg)
    assert torch.equal(previous_rng, torch.get_rng_state())
    for before, after in zip(original, models):
        old_state, new_state = before.state_dict(), after.state_dict()
        assert old_state.keys() <= new_state.keys()
        for key, tensor in old_state.items():
            assert torch.equal(tensor, new_state[key]), key
        assert all('attention.' in key for key in new_state.keys() - old_state.keys())
    g, d, e = models
    assert isinstance(g, new.SAGANGenerator)
    assert isinstance(d.critic.pixel, new.SAGANPixelDiscriminator)
    assert new.state_hash([original[2]]) == new.state_hash([e])
    metadata = new.generator_metadata(cfg)
    assert metadata['architecture'] == 'sagan_gd'
    assert metadata['attention']['residual_coefficient'] == 1.
    assert metadata['attention']['phase_in'] is False
    assert metadata['attention']['spectral_normalization'] is False
    assert metadata['attention']['resolution'] == 16
    new.validate(cfg)


def test_generator_shape_and_attention_location_and_gradient():
    cfg = {**new.DEFAULTS, 'width': 8}
    g, d, e = new.build_models(cfg)
    assert sum(p.numel() for p in g.parameters()) == 930883
    observed = []
    hook = g.attention.register_forward_pre_hook(lambda module, args: observed.append(args[0].shape))
    z = torch.randn(2, 64, requires_grad=True)
    output = g(z)
    hook.remove()
    assert observed == [torch.Size([2, 64, 16, 16])]
    assert output.shape == (2, 3, 32, 32)
    assert output.min() >= -1 and output.max() <= 1
    output.square().mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in g.attention.parameters())


def test_encoder_only_reconstruction_isolation():
    cfg = {**new.DEFAULTS, 'z_dim': 8, 'width': 8, 'recon_grad': 'encoder_only'}
    g, d, e = new.build_models(cfg)
    prior = new.MoGParticlePrior(16, 8)
    before = new.state_hash([g, d, prior])
    encoder_before = new.state_hash([e])
    optimizer = torch.optim.Adam([*g.parameters(), *e.parameters(), *prior.parameters()], lr=.001)
    new.reconstruction_loss(g, e, prior, torch.randn(2, 3, 32, 32), cfg).backward()
    assert all(p.grad is None for p in g.parameters()) and prior.z.grad is None
    assert all(p.requires_grad for p in g.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters())
    optimizer.step()
    assert before == new.state_hash([g, d, prior])
    assert encoder_before != new.state_hash([e])


def test_discriminator_grad_toggle_preserves_frozen_features_and_registration():
    _, d, _ = new.build_models({**new.DEFAULTS, 'width': 8, 'z_dim': 8})
    original = new.state_hash([d.critic.features])
    assert any(key.startswith('critic.pixel.attention.') for key in d.state_dict())
    for state in (False, True, False, True):
        d.train().requires_grad_(state)
        assert all(p.requires_grad == state for p in d.critic.pixel.attention.parameters())
        assert all(not p.requires_grad for p in d.critic.features.parameters())
        assert all(not module.training for module in d.critic.features.modules())
    assert original == new.state_hash([d.critic.features])


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CUDA double backward')
def test_cuda_active_bcap_double_backward_and_discriminator_attention():
    torch.manual_seed(24002)
    _, d, _ = new.build_models({**new.DEFAULTS, 'width': 8, 'z_dim': 8})
    d = d.cuda().train().requires_grad_(True)
    original = new.state_hash([d.critic.features])
    # Force a steep critic so default kappa=1 bcap is active, not a zero-penalty pass.
    with torch.no_grad():
        d.critic.pixel.output.weight.mul_(100.)
    real, fake = [torch.randn(2, 3, 32, 32, device='cuda') for _ in range(2)]
    seen = []
    hook = d.critic.pixel.attention.register_forward_pre_hook(
        lambda module, args: seen.append(tuple(args[0].shape)))
    regularizer = new.GradientPenalty(coeff=1., lazy_k=8)
    assert regularizer(d, real, fake, step=7) == 0
    penalty = regularizer(d, real, fake, step=8)
    assert torch.isfinite(penalty) and penalty > 0
    penalty.backward()
    hook.remove()
    assert seen and all(shape == (2, 16, 16, 16) for shape in seen)
    for name, parameter in d.critic.pixel.attention.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in d.critic.features.parameters())
    optimizer = torch.optim.Adam((p for p in d.parameters() if p.requires_grad), lr=.00045)
    optimizer.step()
    assert original == new.state_hash([d.critic.features])


@pytest.mark.parametrize('overrides', [{'g_width': 64}, {'g_depth': 2}])
def test_reject_ignored_generator_overrides(overrides):
    with pytest.raises(ValueError, match='overrides'):
        new.validate({**new.DEFAULTS, **overrides})


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

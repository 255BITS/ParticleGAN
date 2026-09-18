"""Generator-family initialization, true E-only routing and full-state replay."""
import hashlib
import os
import pytest
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from experiments import train_cifar_ae_transgan as trainer


def small_config(**updates):
    return {**trainer.DEFAULTS, 'arm': 'bounded', 'z_dim': 8, 'width': 8,
            'num_particles': 16, 'transgan_dim': 64, 'transgan_depths': [1, 1, 1],
            'transgan_heads': 2, **updates}


def test_upstream_linear_initialization_is_not_overwritten(monkeypatch):
    # Upstream's --init_type=xavier_uniform only changes Conv2d, not Linear.
    # Preserve every Linear's actual constructor state to catch blanket Xavier
    # overrides, which saturated the full-depth generator in the first smoke.
    original = torch.nn.Linear.reset_parameters
    initial = {}
    def capture(module):
        original(module)
        initial[module] = {key: value.clone() for key, value in module.state_dict().items()}
    monkeypatch.setattr(torch.nn.Linear, 'reset_parameters', capture)
    g = trainer.TransGANGenerator(8, 64, [1, 1, 1], 2)
    linears = [m for m in g.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == len(initial) and linears
    for module in linears:
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, initial[module][key], rtol=0, atol=0)


def test_transformer_output_latent_gradient_and_relative_attention():
    torch.set_num_threads(2)
    g = trainer.TransGANGenerator(8, 64, [1, 1, 1], 2)
    z = torch.randn(2, 8, requires_grad=True)
    output = g(z)
    assert output.shape == (2, 3, 32, 32)
    assert torch.isfinite(output).all() and output.abs().max() <= 1
    gradients = torch.autograd.grad(output.square().mean(), [z, *g.parameters()])
    assert all(torch.isfinite(v).all() for v in gradients)
    assert gradients[0].abs().sum() > 0
    assert all(v.abs().sum() > 0 for v in gradients[1:])
    # SDPA agrees with explicit scaled dot product including learned bias.
    attention = g.stages[0][0].attention
    x = torch.randn(2, 64, 64)
    q, k, v = attention.qkv(x).reshape(2, 64, 3, 2, 32).permute(2, 0, 3, 1, 4).unbind(0)
    bias = attention.relative_bias[attention.relative_index].permute(2, 0, 1)
    weights = (q @ k.transpose(-2, -1) / 32 ** .5 + bias).softmax(-1)
    expected = attention.project((weights @ v).transpose(1, 2).reshape(2, 64, 64))
    torch.testing.assert_close(attention(x), expected)


def test_bounded_rgb_head_resists_residual_stream_growth():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    g = trainer.TransGANGenerator(8, 64, [1, 1, 1], 2)
    z = torch.randn(2, 8, requires_grad=True)
    reference = g(z)
    hook = g.stages[-1].register_forward_hook(lambda module, inputs, output: output * 100)
    try:
        enlarged = g(z)
        gradient, = torch.autograd.grad(enlarged.square().mean(), z)
    finally:
        hook.remove()
    assert enlarged.abs().max() < .99
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 1e-6
    torch.testing.assert_close(enlarged, reference, rtol=.001, atol=.0001)


@pytest.mark.parametrize('architecture', ['cnn', 'transgan'])
@pytest.mark.parametrize('mode', ['all', 'encoder_only'])
@sdpa_kernel(SDPBackend.MATH)
def test_reconstruction_routing_keeps_encoder_and_adversarial_gradients(architecture, mode):
    from lib.image_particle_autoencoder import ImageRoutingEncoder
    from particlegan import MoGParticlePrior
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = small_config(generator_arch=architecture, recon_grad=mode)
    g = (trainer.TransGANGenerator(8, 64, [1, 1, 1], 2) if architecture == 'transgan'
         else trainer.DirectGenerator(8, 8))
    e, prior = ImageRoutingEncoder(8, 8), MoGParticlePrior(16, 8)
    real = torch.randn(2, 3, 32, 32)
    gparams, eparams = list(g.parameters()), list(e.parameters())
    allparams = gparams + eparams + [prior.z]
    reference = (g(e(real, prior.means(), prior.sigma, cfg['temperature'])[0]) - real).square().mean()
    ref_grad = torch.autograd.grad(reference, allparams)
    z, _ = prior.sample(2)
    adversarial = g(z).square().mean()  # Built before the temporary G freeze.
    rec = trainer.reconstruction_loss(g, e, prior, real, cfg)
    torch.testing.assert_close(rec, reference, rtol=0, atol=0)
    actual = torch.autograd.grad(rec, allparams, allow_unused=True, retain_graph=True)
    for a, b in zip(actual[len(gparams):-1], ref_grad[len(gparams):-1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(v.abs().sum() > 0 for v in actual[len(gparams):-1])
    if mode == 'encoder_only':
        assert all(v is None for v in actual[:len(gparams)]) and actual[-1] is None
        expected = torch.autograd.grad(adversarial, gparams + [prior.z], retain_graph=True)
        combined = torch.autograd.grad(adversarial + rec, gparams + [prior.z])
        for a, b in zip(expected, combined):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert expected[-1].abs().sum() > 0
    else:
        for a, b in zip(actual, ref_grad):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert actual[-1].abs().sum() > 0
    assert all(p.requires_grad for p in gparams)


def test_shared_initialization_and_original_cnn_are_preserved():
    from experiments import train_cifar_ae_routing as historical
    torch.set_num_threads(2)
    cfg = small_config(generator_arch='cnn')
    torch.manual_seed(cfg['seed'])
    original = historical.build_models(cfg)
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(cfg['seed'])
    actual = trainer.build_models(cfg)
    assert trainer.state_hash(original) == trainer.state_hash(actual)
    assert torch.equal(original_rng, torch.get_rng_state())
    torch.manual_seed(cfg['seed'])
    alternative = trainer.build_models({**cfg, 'generator_arch': 'transgan'})
    assert trainer.state_hash(original[1:]) == trainer.state_hash(alternative[1:])
    assert torch.equal(original_rng, torch.get_rng_state())


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA replay')
@pytest.mark.parametrize('mode', ['all', 'encoder_only'])
def test_full_state_continuation_matches_uninterrupted(tmp_path, monkeypatch, mode):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    factory = trainer.build_models
    def deterministic_models(cfg):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        g, d, e = factory(cfg)
        # Adaptive pooling CUDA backward is nondeterministic; equivalent fixed
        # pooling permits testing complete optimizer/RNG replay deterministically.
        for head, kernel in zip(d.critic.project, (4, 2, 1)):
            head[5] = torch.nn.AvgPool2d(kernel)
        return g, d, e
    monkeypatch.setattr(trainer, 'build_models', deterministic_models)
    cfg = small_config(recon_grad=mode, batch_size=2, steps=8, reg_every=4,
                       eval_interval=4, log_interval=4, eval_samples=0, final_samples=0,
                       recon_samples=4, eval_batch_size=4, out_dir=str(tmp_path / 'full'))
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            trainer.train(cfg)
            partial = tmp_path / 'partial'
            trainer.train({**cfg, 'steps': 4, 'out_dir': str(partial)})
            path = partial / 'checkpoint.pt'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            resumed = {**cfg, 'out_dir': str(tmp_path / 'resumed'),
                       'resume_checkpoint': str(path), 'resume_sha256': digest}
            for invalid in ({'lr': .0001}, {'generator_arch': 'cnn'}, {'transgan_dim': 128},
                            {'transgan_depths': [2, 1, 1]}, {'resume_sha256': '0' * 64}, {'steps': 4}):
                with pytest.raises(ValueError):
                    trainer.load_resume({**resumed, **invalid})
            trainer.train(resumed)
        a = torch.load(tmp_path / 'full/checkpoint.pt', map_location='cpu', weights_only=False)
        b = torch.load(tmp_path / 'resumed/checkpoint.pt', map_location='cpu', weights_only=False)
        for name in ('G', 'D', 'E', 'prior', 'ema_G', 'ema_E', 'ema_prior'):
            for key, value in a[name].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, b[name][key], rtol=1e-5, atol=1e-6)
                else:
                    assert value == b[name][key]
        for name in a['rng']:
            assert torch.equal(a['rng'][name], b['rng'][name])
        assert torch.equal(a['torch_rng'], b['torch_rng'])
        assert all(torch.equal(x, y) for x, y in zip(a['cuda_rng'], b['cuda_rng']))
        for name in ('optimizer_g', 'optimizer_d'):
            assert a[name]['param_groups'] == b[name]['param_groups']
            for key, state in a[name]['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, b[name]['state'][key][field], rtol=1e-5, atol=1e-6)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    finally:
        torch.use_deterministic_algorithms(deterministic)

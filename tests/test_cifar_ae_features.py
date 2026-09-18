"""Backbone replacement, selective gradients and resumable combined intervention."""
import copy
import hashlib
import os
import pytest
import torch
from experiments import train_cifar_ae_features as trainer


@pytest.mark.parametrize('allow_reconstruction', [False, True])
def test_only_new_G_reconstruction_gradients_change(allow_reconstruction):
    from lib.image_particle_autoencoder import DirectGenerator, ImageRoutingEncoder
    from particlegan import MoGParticlePrior
    torch.set_num_threads(2)
    cfg = {**trainer.DEFAULTS, 'grow_g': True, 'recon_growth_grad': allow_reconstruction,
           'z_dim': 8, 'width': 8, 'num_particles': 16}
    torch.manual_seed(42)
    g = DirectGenerator(8, 8)
    eg = copy.deepcopy(g).eval().requires_grad_(False)
    trainer.install_growth(cfg, g, None, eg)
    # Nonzero refinements ensure this also tests gradient transmission through
    # learned branches, rather than just through an initial identity function.
    with torch.no_grad():
        for block in g.growth:
            block.c2.weight.normal_(std=.01)
    e, prior = ImageRoutingEncoder(8, 8), MoGParticlePrior(16, 8)
    real, z = torch.randn(4, 3, 32, 32), torch.randn(4, 8)
    old = [p for name, p in g.named_parameters() if not name.startswith('growth.')]
    new = list(g.growth.parameters())
    parameters = old + list(e.parameters()) + [prior.z] + new
    reference = (g(e(real, prior.means(), prior.sigma, cfg['temperature'])[0]) - real).square().mean()
    reference_grad = torch.autograd.grad(reference, parameters)
    # Adversarial graph is built before the temporary reconstruction freeze.
    adversarial = g(z).square().mean()
    rec = trainer.reconstruction_loss(g, e, prior, real, cfg)
    torch.testing.assert_close(rec, reference, rtol=0, atol=0)
    actual = torch.autograd.grad(rec, parameters, allow_unused=True, retain_graph=True)
    for a, b in zip(actual[:-len(new)], reference_grad[:-len(new)]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(v.abs().sum() > 0 for v in actual[len(old):-len(new)])
    if allow_reconstruction:
        for a, b in zip(actual[-len(new):], reference_grad[-len(new):]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        assert all(v is None for v in actual[-len(new):])
        expected = torch.autograd.grad(adversarial, new, retain_graph=True)
        combined = torch.autograd.grad(adversarial + rec, new)
        for a, b in zip(expected, combined):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(p.requires_grad for p in g.parameters())


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real-parent CUDA backbone swap')
def test_backbone_replacement_preserves_other_state_and_supports_bcap():
    torch.set_num_threads(2)
    ck = torch.load(trainer.ROOT / 'runs/cifar_particle_ae/duration_100k/n08/checkpoint_050000.pt',
                    map_location='cpu', weights_only=False)
    cfg = {**trainer.DEFAULTS, **ck['config'], 'd_backbone': 'pretrained_resnet34'}
    g, d, _ = [m.cuda() for m in trainer.build_models(cfg)]
    g.load_state_dict(ck['G']); d.load_state_dict(ck['D'])
    eg = copy.deepcopy(g).eval().requires_grad_(False); eg.load_state_dict(ck['ema_G'])
    params = [p for p in d.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg['d_lr'], betas=(0., .999), fused=True)
    opt.load_state_dict(ck['optimizer_d'])
    saved = copy.deepcopy(opt.state_dict())
    model_hash = trainer.state_hash([g, eg, d.critic.pixel, d.critic.project])
    feature_hash = trainer.state_hash([d.critic.features])
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    audit = trainer.replace_backbone_with_audit(cfg, g, d, eg)
    assert audit['changed'] and audit['maximum_absolute_change']['D'] > 0
    assert trainer.state_hash([g, eg, d.critic.pixel, d.critic.project]) == model_hash
    assert trainer.state_hash([d.critic.features]) != feature_hash
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert all(a is b for a, b in zip(params, [p for p in d.parameters() if p.requires_grad]))
    for index, state in saved['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, opt.state_dict()['state'][index][key], rtol=0, atol=0)
    assert saved['param_groups'] == opt.state_dict()['param_groups']
    assert d._context_features is None
    assert trainer.expand_with_audit(cfg, g, d, eg)['double_backward_finite']
    d.train().requires_grad_(True)
    assert not any(p.requires_grad for p in d.critic.features.parameters())
    assert not any(m.training for m in d.critic.features.modules())
    assert not trainer.install_backbone(d, 'pretrained_resnet34')

@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA replay')
def test_full_state_continuation_matches_uninterrupted(tmp_path, monkeypatch):
    factory = trainer.build_models
    def deterministic_models(cfg):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        g, d, e = factory(cfg)
        for head, kernel in zip(d.critic.project, (4, 2, 1)):
            head[5] = torch.nn.AvgPool2d(kernel)
        return g, d, e
    monkeypatch.setattr(trainer, 'build_models', deterministic_models)
    cfg = {**trainer.DEFAULTS, 'arm': 'bounded', 'z_dim': 8, 'width': 8,
           'grow_g': True, 'grow_d_heads': False, 'd_backbone': 'pretrained_resnet34', 'recon_growth_grad': False, 'num_particles': 32, 'batch_size': 4, 'steps': 8, 'reg_every': 4,
           'eval_interval': 4, 'log_interval': 4, 'eval_samples': 0, 'final_samples': 0,
           'recon_samples': 16, 'eval_batch_size': 16, 'out_dir': str(tmp_path / 'full')}
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        trainer.train(cfg)
        partial = tmp_path / 'partial'
        trainer.train({**cfg, 'steps': 4, 'out_dir': str(partial)})
        path = partial / 'checkpoint.pt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        resumed = {**cfg, 'out_dir': str(tmp_path / 'resumed'),
                   'resume_checkpoint': str(path), 'resume_sha256': digest}
        for invalid in ({'lr': .0001}, {'g_depth': 2}, {'resume_sha256': '0' * 64}, {'steps': 4},
                        {'grow_g': False}):
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

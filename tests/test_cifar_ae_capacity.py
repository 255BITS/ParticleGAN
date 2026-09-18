import hashlib
import os

import pytest
import torch

from experiments import train_cifar_ae_capacity as trainer
from lib.image_particle_autoencoder import DirectGenerator


def test_generator_variants_preserve_discriminator_encoder_initialization():
    torch.set_num_threads(2)
    cfg = {**trainer.DEFAULTS, 'z_dim': 16, 'width': 8}
    reference = None
    for width, depth in [(8, 1), (16, 1), (16, 2)]:
        torch.manual_seed(24002)
        g, d, e = trainer.build_models({**cfg, 'g_width': width, 'g_depth': depth})
        digest = trainer.state_hash([d, e])
        if reference is None:
            reference = digest
            torch.manual_seed(24002)
            assert trainer.state_hash([g]) == trainer.state_hash([DirectGenerator(16, 8)])
        assert digest == reference
        z = torch.randn(3, 16)
        y = g(z)
        assert y.shape == (3, 3, 32, 32)
        torch.testing.assert_close(y[:1], g(z[:1]), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('change', [{'g_width': 7}, {'g_depth': 0}, {'g_width': True},
                                   {'resume_checkpoint': 'file.pt'}])
def test_invalid_capacity_config(change):
    with pytest.raises(ValueError):
        trainer.validate({**trainer.DEFAULTS, **change})


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
           'num_particles': 32, 'batch_size': 4, 'steps': 8, 'reg_every': 4,
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
        for invalid in ({'lr': .0001}, {'g_depth': 2}, {'resume_sha256': '0' * 64}, {'steps': 4}):
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

import hashlib
import os

import pytest
import torch

from experiments import train_cifar_ae_plateau as trainer
from lib.image_particle_autoencoder import DirectGenerator


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


def test_schedule_is_global_and_restored_optimizer_rates_are_overridden():
    cfg = {**trainer.DEFAULTS, 'lr_scale': .25, 'lr_final_ratio': .2,
           'lr_decay_start': 60000, 'lr_decay_end': 200000}
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(4)]
    og = torch.optim.Adam([{'params': [p], 'lr': 99.} for p in parameters[:3]])
    od = torch.optim.Adam([parameters[3]], lr=99.)
    for step, expected in [(50001, .25), (60000, .25), (130000, .15), (200000, .05)]:
        assert trainer.set_learning_rates(cfg, step, og, od) == pytest.approx(expected)
        assert [g['lr'] for g in og.param_groups] == pytest.approx(
            [cfg['lr'] * expected, cfg['lr'] * expected, cfg['prior_lr'] * expected])
        assert od.param_groups[0]['lr'] == pytest.approx(cfg['d_lr'] * expected)


@pytest.mark.parametrize('change', [{'lr_scale': 0}, {'recon_weight': -1},
                                   {'d_updates': 0}, {'lr_decay_start': 200000}])
def test_reject_invalid_interventions(change):
    with pytest.raises(ValueError):
        trainer.validate({**trainer.DEFAULTS, **change})

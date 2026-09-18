"""Unchanged continuation replays the original trainer's complete state."""
import hashlib
import os
import pytest
import torch
from experiments import train_cifar_ae_transgan as old
from experiments import train_cifar_ae_balance as new


@pytest.fixture
def deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA test')
def test_control_preserves_original_full_state(tmp_path, monkeypatch, deterministic_algorithms):
    for trainer in (old, new):
        factory = trainer.build_models
        def deterministic(cfg, factory=factory):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            g, d, e = factory(cfg)
            for head, kernel in zip(d.critic.project, (4, 2, 1)):
                head[5] = torch.nn.AvgPool2d(kernel)
            return g, d, e
        monkeypatch.setattr(trainer, 'build_models', deterministic)
    cfg = {**old.DEFAULTS, 'generator_arch': 'cnn', 'recon_grad': 'encoder_only',
           'arm': 'bounded', 'z_dim': 8, 'width': 8, 'num_particles': 16,
           'batch_size': 2, 'steps': 4, 'reg_every': 4, 'eval_interval': 4,
           'log_interval': 4, 'eval_samples': 0, 'final_samples': 0,
           'recon_samples': 4, 'eval_batch_size': 4, 'out_dir': str(tmp_path / 'parent')}
    old.train(cfg)
    parent = tmp_path / 'parent/checkpoint.pt'
    cfg.update(steps=8, resume_checkpoint=str(parent), resume_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    old.train({**cfg, 'out_dir': str(tmp_path / 'old')})
    new.train({**new.DEFAULTS, **cfg, 'out_dir': str(tmp_path / 'new')})
    a, b = [torch.load(tmp_path / name / 'checkpoint.pt', map_location='cpu', weights_only=False) for name in ('old', 'new')]
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: compare(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b): compare(x, y)
        else:
            assert a == b
    for key in ('G', 'D', 'E', 'prior', 'ema_G', 'ema_E', 'ema_prior', 'optimizer_g', 'optimizer_d', 'rng', 'torch_rng', 'cuda_rng'):
        compare(a[key], b[key])


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA test')
def test_half_g_lr_preserves_first_step_e_prior_d(tmp_path, monkeypatch, deterministic_algorithms):
    # At the first update all gradients/moments are identical; only G's actual
    # parameter delta must change. Also checks restored Adam LRs are overridden.
    factory = new.build_models
    def deterministic(cfg):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        g, d, e = factory(cfg)
        for head, kernel in zip(d.critic.project, (4, 2, 1)):
            head[5] = torch.nn.AvgPool2d(kernel)
        return g, d, e
    monkeypatch.setattr(new, 'build_models', deterministic)
    cfg = {**new.DEFAULTS, 'generator_arch': 'cnn', 'recon_grad': 'encoder_only',
           'arm': 'bounded', 'z_dim': 8, 'width': 8, 'num_particles': 16,
           'batch_size': 2, 'steps': 4, 'reg_every': 4, 'eval_interval': 4,
           'log_interval': 4, 'eval_samples': 0, 'final_samples': 0,
           'recon_samples': 4, 'eval_batch_size': 4, 'out_dir': str(tmp_path / 'parent')}
    new.train(cfg)
    path = tmp_path / 'parent/checkpoint.pt'
    parent = torch.load(path, map_location='cpu', weights_only=False)
    cfg.update(steps=5, resume_checkpoint=str(path), resume_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for name, scale in [('control', 1.), ('half', .5)]:
        new.train({**cfg, 'g_lr_scale': scale, 'out_dir': str(tmp_path/name)})
    a,b = [torch.load(tmp_path/name/'checkpoint.pt', map_location='cpu', weights_only=False) for name in ('control','half')]
    for key in ('D', 'E', 'prior', 'ema_E', 'ema_prior'):
        for k in a[key]: torch.testing.assert_close(a[key][k], b[key][k], rtol=0, atol=0)
    for opt in ('optimizer_g', 'optimizer_d'):
        for pid, state in a[opt]['state'].items():
            for k in state: torch.testing.assert_close(state[k], b[opt]['state'][pid][k], rtol=0, atol=0)
    ga,gb = a['optimizer_g']['param_groups'],b['optimizer_g']['param_groups']
    assert gb[0]['lr'] == ga[0]['lr'] * .5
    assert [x['lr'] for x in gb[1:]] == [x['lr'] for x in ga[1:]]
    assert a['optimizer_d']['param_groups'] == b['optimizer_d']['param_groups']
    assert any(not torch.equal(a['G'][k], b['G'][k]) for k in a['G'])
    for k, value in parent['G'].items():
        torch.testing.assert_close(b['G'][k]-value, (a['G'][k]-value)*.5, rtol=1e-3, atol=1e-7)

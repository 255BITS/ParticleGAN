"""Opt-in GPU integration test: interrupt at a checkpoint and replay exactly.

Uses real CIFAR batches and replaces only the expensive FID metric with a stub.
All temporary run artifacts are pytest temporary files, not experiment results.
"""
import os
from pathlib import Path
import pytest
import torch

@pytest.fixture
def deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='opt-in real-data CUDA integration test')
@pytest.mark.parametrize('architecture,d_backbone', [('unet','pixel'), ('flat_hybrid','pixel'), ('unet','pretrained_resnet18'), ('ncsnpp','pretrained_resnet18')])
def test_checkpoint_resume_matches_uninterrupted(tmp_path, monkeypatch, architecture, d_backbone, deterministic_algorithms):
    from experiments import train_cifar_ddgan as trainer
    from lib import cifar_metrics

    class StubFID:
        def __init__(self, *args):
            pass
        def __call__(self, images):
            return 0.0

    monkeypatch.setattr(cifar_metrics, 'FIDEvaluator', StubFID)
    monkeypatch.setattr(cifar_metrics, 'save_grid', lambda *args: None)
    # Production uses CUDNN benchmarking for speed. Force fixed kernels here so
    # numerical kernel-selection differences cannot masquerade as lost state.
    # Deterministic algorithms also cover pretrained D bilinear-resize gradients.
    model_factory = trainer.build_models
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', True)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', False)
    def deterministic_models(cfg):
        torch.backends.cudnn.benchmark = False
        g, d = model_factory(cfg)
        if cfg['d_backbone'] == 'pretrained_resnet18':
            # CUDA adaptive pooling has no deterministic backward. At these
            # fixed16/8/4 feature sizes, ordinary4/2/1 pooling is the same
            # non-overlapping reduction; use it only for exact replay testing.
            for head, kernel in zip(d.project, (4, 2, 1)):
                head[5] = torch.nn.AvgPool2d(kernel)
        return g, d
    monkeypatch.setattr(trainer, 'build_models', deterministic_models)
    cfg={**trainer.DEFAULTS, 'architecture':architecture, 'd_backbone':d_backbone, 'g_width':16, 'g_depth':1, 'd_width':8, 'z_dim':8,
         'num_particles':100, 'steps':20, 'batch_size':8,
         'log_interval':10, 'eval_interval':10, 'eval_samples':10,
         'final_samples':10, 'eval_batch_size':10, 'tf32':False,
         'out_dir':str(tmp_path/'resumed')}
    save=torch.save
    def interrupting_save(obj,path):
        save(obj,path)
        if obj.get('step')==10:
            Path(path).replace(Path(path).with_name('checkpoint.pt'))
            raise InterruptedError('simulated process interruption after checkpoint')
    monkeypatch.setattr(torch,'save',interrupting_save)
    with pytest.raises(InterruptedError):
        trainer.train(cfg)
    monkeypatch.setattr(torch,'save',save)
    trainer.train(cfg,resume=tmp_path/'resumed/checkpoint.pt')
    trainer.train({**cfg,'out_dir':str(tmp_path/'uninterrupted')})
    a=torch.load(tmp_path/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'uninterrupted/checkpoint.pt',map_location='cpu',weights_only=False)
    for model in ('G','D','prior','ema_G','ema_prior'):
        for key in a[model]:
            torch.testing.assert_close(a[model][key],b[model][key],rtol=1e-5,atol=1e-6)
    for name in a['rngs']:
        assert torch.equal(a['rngs'][name],b['rngs'][name])
    assert a['step']==b['step']==20

"""Checkpoint-preserving capacity growth and exact grown-state continuation."""
import copy
import hashlib
import os
import pytest
import torch
from experiments import train_cifar_ae_growth as trainer


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real checkpoint CUDA expansion')
@pytest.mark.parametrize('grow_g,grow_d', [(False, False), (True, False), (False, True), (True, True)])
def test_real_parent_growth_preserves_state_and_learns(grow_g, grow_d):
    from particlegan import MoGParticlePrior
    torch.set_num_threads(2)
    ck = torch.load(trainer.ROOT / 'runs/cifar_particle_ae/duration_100k/n08/checkpoint_050000.pt',
                    map_location='cpu', weights_only=False)
    cfg = {**trainer.DEFAULTS, **ck['config'], 'grow_g': grow_g, 'grow_d_heads': grow_d}
    g, d, e = [m.cuda() for m in trainer.build_models(cfg)]
    prior = MoGParticlePrior(cfg['num_particles'], cfg['z_dim'], sigma_rel=cfg['sigma_rel']).cuda()
    eg, ee, ep = [copy.deepcopy(m).eval().requires_grad_(False) for m in (g, e, prior)]
    og = torch.optim.Adam([{'params': g.parameters()}, {'params': e.parameters()},
                           {'params': prior.parameters(), 'betas': (.5, .999)}],
                          lr=cfg['lr'], betas=(0., .999), fused=True)
    od = torch.optim.Adam([p for p in d.parameters() if p.requires_grad],
                          lr=cfg['d_lr'], betas=(0., .999), fused=True)
    streams = {name: trainer.rng(42) for name in ck['rng']}
    trainer.restore_checkpoint(ck, (g, d, e, prior, eg, ee, ep), (og, od), streams)
    old_models = [{k: v.detach().clone() for k, v in m.state_dict().items()}
                  for m in (g, d, e, eg, ee)]
    old_parameters = {id(p) for m in (g, d) for p in m.parameters()}
    old_state = [{p: {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                      for k, v in state.items()} for p, state in opt.state.items()}
                 for opt in (og, od)]
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    backend_flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                     torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    trainer.expand_with_audit(cfg, g, d, eg, og, od)
    assert backend_flags == (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for m, before in zip((g, d, e, eg, ee), old_models):
        for key, value in before.items():
            torch.testing.assert_close(m.state_dict()[key], value, rtol=0, atol=0)
    for opt, before in zip((og, od), old_state):
        for p, state in before.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(opt.state[p][key], value, rtol=0, atol=0)
                else:
                    assert opt.state[p][key] == value
        for group in opt.param_groups:
            for p in group['params']:
                if id(p) not in old_parameters and p not in before:
                    assert p not in opt.state  # New parameters have fresh Adam state.
    assert not any(p.requires_grad for p in eg.parameters())
    assert not any(p.requires_grad for p in d.critic.features.parameters())
    assert len(og.param_groups) == 3 and len(od.param_groups) == 1
    # A real loss reaches each new final convolution immediately, then reaches
    # earlier branch weights after the first update; branches aren't dead.
    for model, opt, enabled, shape in ((g, og, grow_g, (4, cfg['z_dim'])),
                                      (d, od, grow_d, (4, 3, 32, 32))):
        if not enabled:
            continue
        new = [p for p in model.parameters() if id(p) not in old_parameters]
        assert new
        inputs = torch.randn(shape, device='cuda')
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new)
            assert any(p.grad.abs().sum() > 0 for p in new)
            opt.step()
        assert all(p in opt.state for p in new)
    # Installing the same expansion twice must not duplicate parameters.
    lengths = [len(group['params']) for opt in (og, od) for group in opt.param_groups]
    trainer.install_growth(cfg, g, d, eg, og, od)
    assert lengths == [len(group['params']) for opt in (og, od) for group in opt.param_groups]

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
           'grow_g': True, 'grow_d_heads': True, 'num_particles': 32, 'batch_size': 4, 'steps': 8, 'reg_every': 4,
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
                        {'grow_g': False}, {'grow_d_heads': False}):
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

"""Verify reconstruction routing changes gradient recipients, not forward values."""
import os
import hashlib
import pytest
import torch
from experiments import train_cifar_ae_routing as trainer
from lib.image_particle_autoencoder import DirectGenerator, ImageRoutingEncoder
from particlegan import MoGParticlePrior

@pytest.mark.parametrize('prior_grad,g_grad', [(True,True),(False,True),(False,False)])
def test_reconstruction_gradient_recipients_and_adversarial_path(prior_grad,g_grad):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg={**trainer.DEFAULTS,'width':8,'z_dim':8,'num_particles':16,
         'recon_prior_grad':prior_grad,'recon_generator_grad':g_grad}
    g=DirectGenerator(8,8);e=ImageRoutingEncoder(8,8)
    p=MoGParticlePrior(16,8)
    x=torch.randn(4,3,32,32)
    reference=(g(e(x,p.means(),p.sigma,cfg['temperature'])[0])-x).square().mean()
    # Build adversarial graph first, as the real training loop does.
    z,_=p.sample(4)
    adv=g(z).square().mean()
    rec=trainer.reconstruction_loss(g,e,p,x,cfg)
    torch.testing.assert_close(rec,reference,rtol=0,atol=0)
    params=[*g.parameters(),*e.parameters(),p.z]
    grads=torch.autograd.grad(rec,params,allow_unused=True,retain_graph=True)
    ng=len(list(g.parameters()))
    assert any(v is not None and bool(v.abs().sum()>0) for v in grads[:ng]) == g_grad
    assert any(v is not None and bool(v.abs().sum()>0) for v in grads[ng:-1])
    assert (grads[-1] is not None and bool(grads[-1].abs().sum()>0)) == prior_grad
    assert all(q.requires_grad for q in g.parameters())
    if not g_grad:
        expected=torch.autograd.grad(adv,list(g.parameters())+[p.z],retain_graph=True)
        actual=torch.autograd.grad(adv+rec,list(g.parameters())+[p.z])
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA replay')
def test_encoder_only_full_state_continuation_matches_uninterrupted(tmp_path, monkeypatch):
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
           'recon_generator_grad': False, 'recon_prior_grad': False,
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


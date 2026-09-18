"""Verify reconstruction routing changes gradient recipients, not forward values."""
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

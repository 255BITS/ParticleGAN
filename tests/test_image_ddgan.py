import copy
import torch
from lib.denoising_toy import DiffusionSchedule, DrawSource, FixedConditionCritic
from lib.image_ddgan import ImageGenerator, ImageDiscriminator, sample_images, update_ema
from lib.grad_regularizers import GradRegularizer
from experiments.train_cifar_ddgan import DEFAULTS, DEFAULT_CONFIG
import yaml
import pytest


def test_image_schedule_and_clean_last_step():
    s=DiffusionSchedule(DEFAULTS['alpha_bar'])
    x=torch.randn(4,3,32,32); t=torch.arange(1,5)
    r=torch.Generator().manual_seed(1)
    prev,xt=s.forward_pair(x,t,r)
    assert prev.shape==xt.shape==x.shape
    torch.testing.assert_close(prev[0],x[0])
    clean=s.reverse(x,xt,torch.ones(4,dtype=torch.long),torch.randn_like(x))
    torch.testing.assert_close(clean,x)
    # Image and flattened operations use identical coefficients and RNG draws.
    p2,t2=s.forward_pair(x.flatten(1),t,torch.Generator().manual_seed(1))
    torch.testing.assert_close(prev.flatten(1),p2)
    torch.testing.assert_close(xt.flatten(1),t2)


@pytest.mark.parametrize('d_norm', ['none', 'group'])
@pytest.mark.parametrize('ucd_target', ['class', 'time_class'])
def test_ucd_particle_and_cap_image_gradients(d_norm, ucd_target):
    torch.set_num_threads(1)
    cfg={**DEFAULTS,'g_width':8,'d_width':8,'z_dim':8,'num_particles':20,'d_norm':d_norm,'ucd_target':ucd_target}
    g,d=ImageGenerator(cfg),ImageDiscriminator(cfg)
    prior=DrawSource('learned',20,8,1,'cpu')
    rng=torch.Generator().manual_seed(2)
    c=torch.tensor([0,1]);t=torch.tensor([1,4]);xt=torch.randn(2,3,32,32)
    z,ids=prior.sample(2,rng);fake=g(z,c,xt,t)
    score,logits=d(fake,c,xt,t)
    torch.testing.assert_close(score,logits[torch.arange(2),d.ucd_labels(c,t)])
    assert logits.shape == (2, 40 if ucd_target == 'time_class' else 10)
    if ucd_target == 'time_class':
        torch.testing.assert_close(d.ucd_labels(c,t),torch.tensor([0,31]))
        torch.testing.assert_close(d(fake,c.flip(0),xt,t.flip(0))[1],logits)
        assert d.time is None
    score.sum().backward()
    assert prior.table.grad[ids].abs().sum()>0
    assert torch.isfinite(prior.table.grad).all()
    # UCD logits are label-independent and per-sample (no batch statistics).
    torch.testing.assert_close(d(fake,c.flip(0),xt,t)[1],logits)
    torch.testing.assert_close(d(fake[:1],c[:1],xt[:1],t[:1])[1],logits[:1],atol=1e-6,rtol=1e-5)
    real=torch.randn_like(fake)
    critic=FixedConditionCritic(d,c,xt,t)
    reg=GradRegularizer('b_cap',1,kappa=0)
    penalty,_=reg.penalty(critic,real,fake.detach(),1)
    penalty.backward()
    assert torch.isfinite(penalty) and penalty>0
    assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
    assert xt.grad is None
    ema=copy.deepcopy(g)
    with torch.no_grad():
        next(g.parameters()).add_(1)
    update_ema(ema,g,.5)
    torch.testing.assert_close(next(ema.parameters()),next(g.parameters())-.5)
    result=sample_images(ema,prior,DiffusionSchedule(cfg['alpha_bar']),c,rng)
    assert result.shape==real.shape and torch.isfinite(result).all()


def test_noarg_defaults_match_config():
    assert yaml.safe_load(DEFAULT_CONFIG.read_text())==DEFAULTS


def test_fid_quantization_and_statistics_api():
    import pytest
    pytest.importorskip('torch_fidelity')
    import numpy as np
    from lib.cifar_metrics import uint8_images
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    x=torch.tensor([-2.,-1.,0.,1.,2.])
    assert uint8_images(x).tolist()==[0,0,128,255,255]
    stats={'mu':np.zeros(4), 'sigma':np.eye(4)}
    assert abs(fid_statistics_to_metric(stats,stats,False)['frechet_inception_distance'])<1e-8

import torch
from experiments.train_cifar_ddgan import DEFAULTS, validate
from lib.image_ncsnpp import NCSNppParticleGenerator
from lib.denoising_toy import DrawSource, DiffusionSchedule
from lib.ddgan_ncsnpp.upfirdn2d import upfirdn2d


def test_ncsnpp_particle_class_and_image_gradients():
    torch.set_num_threads(1)
    cfg = {**DEFAULTS, 'architecture': 'ncsnpp', 'g_width': 16,
           'ncsnpp_res_blocks': 1, 'ncsnpp_z_emb_dim': 32, 'z_dim': 8}
    validate(cfg)
    g = NCSNppParticleGenerator(cfg)
    prior = DrawSource('learned', 20, 8, 1, 'cpu')
    z, ids = prior.sample(2, torch.Generator().manual_seed(3))
    xt = torch.randn(2, 3, 32, 32, requires_grad=True)
    c, t = torch.tensor([0, 1]), torch.tensor([1, 4])
    clean = g(z, c, xt, t)
    assert clean.shape == xt.shape and clean.abs().max() <= 1
    fake = DiffusionSchedule(cfg['alpha_bar']).reverse(clean, xt, t, torch.randn_like(xt))
    (fake * torch.randn_like(fake)).sum().backward()
    for v in (xt.grad, prior.table.grad[ids], g.cls.weight.grad,
              g.net.z_transform[1].weight.grad):
        assert torch.isfinite(v).all() and v.abs().sum() > 0
    torch.testing.assert_close(g(z[:1], c[:1], xt[:1], t[:1]), clean[:1], atol=1e-6, rtol=1e-5)
    # The class extension is optional; zero class embedding recovers upstream.
    with torch.no_grad():
        g.cls.weight.zero_()
        torch.testing.assert_close(g(z, c, xt, t), g.net(xt, t, z))


def test_fir_resampling_matches_explicit_zero_insertion():
    torch.set_num_threads(1)
    x = torch.randn(2, 3, 5, 5, dtype=torch.double, requires_grad=True)
    k = torch.tensor([1., 3., 3., 1.], dtype=torch.double)
    k = k[:, None] * k[None, :] / 64
    y = upfirdn2d(x, k, up=2, down=2, pad=(1, 2))
    expanded = x.new_zeros(2, 3, 10, 10)
    expanded[:, :, ::2, ::2] = x
    ref = torch.nn.functional.conv2d(torch.nn.functional.pad(expanded, (1, 2, 1, 2)),
                                    k.flip((0, 1))[None, None].expand(3, 1, 4, 4), groups=3)[:, :, ::2, ::2]
    torch.testing.assert_close(y, ref)
    a = torch.autograd.grad(y.square().sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(ref.square().sum(), x)[0]
    torch.testing.assert_close(a, b)

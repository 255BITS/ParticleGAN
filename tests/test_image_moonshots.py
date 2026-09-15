import copy
import pytest
import torch
from torch.nn import functional as F
from experiments.train_cifar_ddgan import DEFAULTS, validate
from lib.image_moonshots import build_models
from lib.denoising_toy import DrawSource, DiffusionSchedule, FixedConditionCritic
from lib.grad_regularizers import GradRegularizer


def test_unet_attention_identity_initialization_and_particle_learning():
    from lib.image_ddgan import SpatialAttention
    torch.set_num_threads(1)
    cfg = {**DEFAULTS, 'g_width': 8, 'd_width': 8, 'z_dim': 8,
           'num_particles': 20, 'd_backbone': 'pixel', 'cache_condition': False}
    torch.manual_seed(24)
    baseline, baseline_d = build_models(cfg)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(24)
    cfg = {**cfg, 'g_attn_resolutions': [8, 16]}
    validate(cfg)
    g, d = build_models(cfg)
    assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    for k, v in baseline_d.state_dict().items():
        torch.testing.assert_close(d.state_dict()[k], v, rtol=0, atol=0)
    for k, v in baseline.state_dict().items():
        torch.testing.assert_close(g.state_dict()[k], v, rtol=0, atol=0)
    blocks = [m for m in g.modules() if isinstance(m, SpatialAttention)]
    assert len(blocks) == 4
    prior = DrawSource('learned', 20, 8, 1, 'cpu')
    c, t = torch.tensor([0, 1]), torch.tensor([1, 4])
    xt = torch.randn(2, 3, 32, 32, requires_grad=True)
    z, ids = prior.sample(2, torch.Generator().manual_seed(1))
    clean = g(z, c, xt, t)
    torch.testing.assert_close(clean, baseline(z, c, xt, t), rtol=0, atol=0)
    opt = torch.optim.Adam(g.parameters(), lr=.001)
    schedule = DiffusionSchedule(cfg['alpha_bar'])
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        z = prior.table[ids]
        clean = g(z, c, xt, t)
        fake = schedule.reverse(clean, xt, t, torch.randn_like(xt))
        d(fake, c, xt.detach(), t)[0].sum().backward()
        for v in (xt.grad, prior.table.grad[ids]):
            assert torch.isfinite(v).all() and v.abs().sum() > 0
        for block in blocks:
            assert block.project.weight.grad.abs().sum() > 0
            if step:
                assert torch.isfinite(block.qkv.weight.grad).all()
                assert block.qkv.weight.grad.abs().sum() > 0
        opt.step()
    torch.testing.assert_close(g(z[:1], c[:1], xt[:1], t[:1]),
                               g(z, c, xt, t)[:1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('updates', [
    {'g_attn_resolutions': [4]}, {'g_attn_resolutions': [8, 8]},
    {'g_attn_resolutions': [16], 'g_heads': 3},
    {'g_attn_resolutions': [16], 'architecture': 'flat_hybrid'},
])
def test_unet_attention_invalid_configs(updates):
    with pytest.raises(ValueError):
        validate({**DEFAULTS, **updates})


def test_flat_generator_preserves_grid_and_particle_gradients():
    torch.set_num_threads(1)
    cfg = {**DEFAULTS, 'architecture': 'flat_hybrid', 'd_backbone': 'pixel', 'cache_condition': False, 'g_width': 32, 'g_depth': 2,
           'z_dim': 8, 'num_particles': 20, 'd_width': 8, 'spatial_channels': 4}
    validate(cfg)
    g, d = build_models(cfg)
    prior = DrawSource('learned', 20, 8, 1, 'cpu')
    c, t = torch.tensor([0, 1]), torch.tensor([1, 4])
    xt = torch.randn(2, 3, 32, 32, requires_grad=True)
    torch.testing.assert_close(F.pixel_shuffle(F.pixel_unshuffle(xt, 2), 2), xt)
    shapes = []
    hooks = [b.register_forward_hook(lambda m, a, o: shapes.append(o.shape)) for b in g.blocks]
    z, ids = prior.sample(2, torch.Generator().manual_seed(1))
    clean = g(z, c, xt, t)
    for h in hooks:
        h.remove()
    assert shapes == [torch.Size([2, 256, 32])] * 2
    fake = DiffusionSchedule(cfg['alpha_bar']).reverse(clean, xt, t, torch.randn_like(xt))
    d(fake, c, xt.detach(), t)[0].sum().backward()
    for v in (xt.grad, prior.table.grad[ids], g.spatial.weight.grad, g.embed[0].weight.grad):
        assert torch.isfinite(v).all() and v.abs().sum() > 0
    torch.testing.assert_close(g(z[:1], c[:1], xt[:1], t[:1]), clean[:1], atol=1e-6, rtol=1e-5)


def test_pretrained_joint_ucd_frozen_and_second_order(monkeypatch):
    # Architecture test uses deterministic random backbone; real-weight GPU smokes
    # separately verify the downloaded frozen weights and training path.
    import torchvision.models
    original = torchvision.models.resnet18
    monkeypatch.setattr(torchvision.models, 'resnet18', lambda weights: original(weights=None))
    torch.set_num_threads(1)
    cfg = {**DEFAULTS, 'd_backbone': 'pretrained_resnet18', 'g_width': 8, 'd_width': 8}
    validate(cfg)
    _, d = build_models(cfg)
    state = copy.deepcopy(d.features.state_dict())
    d.train().requires_grad_(False).requires_grad_(True)
    assert not any(p.requires_grad for p in d.features.parameters())
    assert not any(m.training for m in d.features.modules())
    c, t = torch.tensor([0, 1]), torch.tensor([1, 4])
    real, fake, xt = [torch.randn(2, 3, 32, 32) for _ in range(3)]
    real.requires_grad_()
    score, logits = d(real, c, xt, t)
    torch.testing.assert_close(score, logits[torch.arange(2), torch.tensor([0, 31])])
    torch.testing.assert_close(logits, d(real, c.flip(0), xt, t.flip(0))[1])
    torch.testing.assert_close(logits[:1], d(real[:1], c[:1], xt[:1], t[:1])[1], atol=1e-5, rtol=1e-5)
    # Feature branch alone must send gradients to candidate pixels.
    pixels = d.pixel(real, c, xt, t)[1]
    feature_only = logits * (2 ** .5) - pixels
    grad = torch.autograd.grad(feature_only.sum(), real, retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    penalty, _ = GradRegularizer('b_cap', 1, kappa=0).penalty(FixedConditionCritic(d, c, xt, t), real.detach(), fake, 1)
    loss = penalty + F.cross_entropy(logits, d.ucd_labels(c, t))
    opt = torch.optim.Adam([p for p in d.parameters() if p.requires_grad])
    loss.backward(); opt.step()
    assert torch.isfinite(penalty) and penalty > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.project.parameters())
    assert all(p.grad is None for p in d.features.parameters())
    for key, value in d.features.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)

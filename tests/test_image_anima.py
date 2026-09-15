import copy
import hashlib

import pytest
import torch

from experiments.train_cifar_ddgan import DEFAULTS, validate
from lib import image_anima
from lib.image_ddgan import ImageGenerator, update_ema


@pytest.fixture
def donor_cfg(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(image_anima, 'DONOR_DIMS', (48, 24, 4, 8))
    torch.manual_seed(17)
    donor = image_anima.FrozenAnima([0, 1], 48, 24, 4, 8)
    path = tmp_path / 'donor.pt'
    torch.save({'tensors': donor.state_dict(), 'metadata': {'blocks': [0, 1]}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {**DEFAULTS, 'architecture': 'anima_transplant', 'anima_weights': str(path),
            'anima_weights_sha256': digest, 'anima_dtype': 'float32',
            'g_width': 8, 'd_width': 8, 'z_dim': 8, 'num_particles': 20}


@pytest.mark.parametrize('initialization', ['pretrained', 'random'])
def test_transplant_identity_gradients_frozen_ema_and_reload(donor_cfg, initialization):
    cfg = {**donor_cfg, 'anima_init': initialization}
    validate(cfg)
    torch.manual_seed(24)
    baseline = ImageGenerator(cfg)
    baseline_rng = torch.random.get_rng_state()
    torch.manual_seed(24)
    g = image_anima.AnimaTransplantGenerator(cfg)
    assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    g.train().requires_grad_(False).requires_grad_(True)
    assert not any(p.requires_grad for p in g.donor.parameters())
    assert not any(m.training for m in g.donor.modules())
    fixed = copy.deepcopy(g.donor.state_dict())
    ema = copy.deepcopy(g).eval().requires_grad_(False)
    z = torch.randn(2, 8, requires_grad=True)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    c, t = torch.tensor([2, 3]), torch.tensor([1, 4])
    torch.testing.assert_close(g(z, c, x, t), baseline(z, c, x, t), rtol=0, atol=0)
    optimizer = torch.optim.Adam((p for p in g.parameters() if p.requires_grad), lr=.001)
    target = torch.randn_like(x)
    for i in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (g(z, c, x, t) - target).square().mean()
        loss.backward()
        assert z.grad.isfinite().all() and z.grad.abs().sum() > 0
        assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
        assert g.transplant_out.weight.grad.abs().sum() > 0
        if i:
            for p in [g.transplant_in.weight, g.context_z.weight, g.context_c.weight]:
                assert p.grad.isfinite().all() and p.grad.abs().sum() > 0
        assert all(p.grad is None for p in g.donor.parameters())
        optimizer.step()
        update_ema(ema, g, .9)
    for k, v in fixed.items():
        torch.testing.assert_close(v, g.donor.state_dict()[k], rtol=0, atol=0)
        torch.testing.assert_close(v, ema.donor.state_dict()[k], rtol=0, atol=0)
    torch.testing.assert_close(g(z[:1], c[:1], x[:1], t[:1]), g(z, c, x, t)[:1], atol=1e-6, rtol=1e-5)
    restored = image_anima.AnimaTransplantGenerator(cfg)
    restored.load_state_dict(g.state_dict())
    torch.testing.assert_close(g(z, c, x, t), restored(z, c, x, t), rtol=0, atol=0)


def test_pretrained_weights_loaded_and_random_control_adapters_match(donor_cfg):
    torch.manual_seed(10)
    trained = image_anima.AnimaTransplantGenerator(donor_cfg)
    torch.manual_seed(10)
    random = image_anima.AnimaTransplantGenerator({**donor_cfg, 'anima_init': 'random'})
    original = torch.load(donor_cfg['anima_weights'], weights_only=True)['tensors']
    for k, v in original.items():
        torch.testing.assert_close(v, trained.donor.state_dict()[k], rtol=0, atol=0)
    assert not torch.equal(trained.donor.blocks['0'].mlp.layer1.weight, random.donor.blocks['0'].mlp.layer1.weight)
    for k, v in trained.state_dict().items():
        if not k.startswith('donor.'):
            torch.testing.assert_close(v, random.state_dict()[k], rtol=0, atol=0)


def test_rotary_preserves_norm_and_uses_both_spatial_axes():
    cos, sin = image_anima.image_rope(8, 128)
    x = torch.randn(2, 3, 64, 128)
    y = image_anima.rotate(x, cos, sin)
    torch.testing.assert_close(y.square().sum(-1), x.square().sum(-1))
    torch.testing.assert_close(y[:, :, 0], x[:, :, 0], rtol=0, atol=0)
    assert not torch.equal(cos[:, :, 1], cos[:, :, 8])


def test_rejects_wrong_hash(donor_cfg):
    with pytest.raises(ValueError, match='hash mismatch'):
        image_anima.AnimaTransplantGenerator({**donor_cfg, 'anima_weights_sha256': '0' * 64})


@pytest.mark.parametrize('updates', [
    {'anima_blocks': []}, {'anima_blocks': [1, 0]}, {'anima_blocks': [0, 0]},
    {'anima_blocks': [28]}, {'anima_init': 'finetune'}, {'anima_context_tokens': 1},
    {'anima_weights_sha256': 'wrong'}, {'channels_last': True},
])
def test_invalid_config(donor_cfg, updates):
    with pytest.raises(ValueError):
        validate({**donor_cfg, **updates})

"""Explicit init keeps a frozen pretrained ImageNet backbone and initializes the new heads."""
import copy
import pytest
import torch

from experiments.train_cifar_ddgan import DEFAULTS, training_recipe
from lib.image_moonshots import build_models
from particlegan import get_recipe, init
from particlegan.diffusion import DrawSource


@pytest.fixture
def fake_imagenet(monkeypatch):
    # Real code path: resnet18(weights=...) builds, then load_state_dict's the
    # downloaded weights. Reproduce that load with fixed "pretrained" tensors.
    import torchvision.models
    original = torchvision.models.resnet18

    def resnet18(weights):
        net = original(weights=None)
        generator = torch.Generator().manual_seed(7)
        net.load_state_dict({k: torch.randn(v.shape, generator=generator).to(v.dtype)
                             if v.is_floating_point() else v for k, v in net.state_dict().items()})
        return net
    monkeypatch.setattr(torchvision.models, 'resnet18', resnet18)


def _check(d, critic, run):
    backbone = copy.deepcopy(critic.features.state_dict())
    heads = {k: v.clone() for k, v in critic.named_parameters() if not k.startswith('features.')}
    ema = copy.deepcopy(d)
    run(ema)
    for key, value in critic.features.state_dict().items():
        assert torch.equal(value, backbone[key]), key
    changed = {k for k, v in critic.named_parameters() if k in heads and not torch.equal(v, heads[k])}
    assert {k for k in heads if k.endswith('.weight') and critic.get_parameter(k).ndim > 1} <= changed
    for a, b in zip(d.state_dict().values(), ema.state_dict().values()):
        assert torch.equal(a, b)


def test_cifar_ddgan_pretrained_backbone_kept(fake_imagenet):
    torch.set_num_threads(1)
    cfg = {**DEFAULTS, 'd_backbone': 'pretrained_resnet18', 'g_width': 8, 'd_width': 8,
           'z_dim': 8, 'num_particles': 16}
    g, d = build_models(cfg)
    prior = DrawSource(cfg['prior'], cfg['num_particles'], cfg['z_dim'], 1, 'cpu')
    recipe = training_recipe(cfg)

    def run(ema):
        init.deterministic_orthogonal_(g, seed=0)
        init.deterministic_orthogonal_(d, seed=1)
        ema.load_state_dict(d.state_dict())
        recipe.make_optimizers(g, d, prior, ema_critic=ema)
    _check(d, d, run)


def test_particle_direct_critic_pretrained_backbone_kept(fake_imagenet):
    from lib.image_particle_autoencoder import DirectDiscriminator
    torch.set_num_threads(1)
    d = DirectDiscriminator(8)
    g = torch.nn.Linear(4, 4)
    def run(ema):
        init.deterministic_orthogonal_(d, seed=1)
        ema.load_state_dict(d.state_dict())
        get_recipe(z_dim=4, num_particles=8).make_optimizers(g, d, ema_critic=ema)
    _check(d, d.critic, run)

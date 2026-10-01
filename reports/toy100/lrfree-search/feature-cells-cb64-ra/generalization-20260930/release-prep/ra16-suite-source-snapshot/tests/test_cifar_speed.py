import pytest
import torch
from experiments.train_cifar_ddgan import DEFAULTS
from lib.image_moonshots import build_models


@pytest.mark.parametrize('backbone', ['resnet18', 'resnet34'])
def test_condition_cache_preserves_logits_candidate_and_bcap_gradients(monkeypatch, backbone):
    import torchvision.models
    original = getattr(torchvision.models, backbone)
    monkeypatch.setattr(torchvision.models, backbone, lambda weights: original(weights=None))
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = {**DEFAULTS, 'g_width': 8, 'd_width': 8, 'd_backbone': 'pretrained_' + backbone}
    from experiments.train_cifar_ddgan import validate
    validate(cfg)
    _, d = build_models(cfg)
    d.eval()
    c, t = torch.tensor([1, 4]), torch.tensor([1, 3])
    x, xt = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    cached = d.condition_features(xt)
    assert d.pretrained_metadata['weights'] == ('ResNet18_Weights.IMAGENET1K_V1' if backbone == 'resnet18' else 'ResNet34_Weights.IMAGENET1K_V1')
    d.train().requires_grad_(True)
    assert not d.features.training
    assert all(not p.requires_grad for p in d.features.parameters())
    assert all(not v.requires_grad for v in cached)
    params = [p for p in d.parameters() if p.requires_grad]
    records = []
    for features in (None, cached):
        candidate = x.clone().requires_grad_()
        score, logits = d(candidate, c, xt, t, condition_features=features)
        grad = torch.autograd.grad(score.sum(), candidate, create_graph=True)[0]
        penalty = grad.square().sum()  # kappa=0 activates every sample's bcap.
        grads = torch.autograd.grad(penalty, params, allow_unused=True)
        records.append((logits.detach(), grad.detach(), grads))
    for a, b in zip(records[0][:2], records[1][:2]):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for a, b in zip(records[0][2], records[1][2]):
        if a is not None:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)

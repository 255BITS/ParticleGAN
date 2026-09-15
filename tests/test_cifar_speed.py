import copy
import pytest
import torch
from experiments.train_cifar_ddgan import DEFAULTS
from lib.cifar_speed import cifar_penalty, finite_difference_norm
from lib.grad_regularizers import GradRegularizer
from lib.image_moonshots import build_models


def test_fd_recovers_norm_and_parameter_gradient_on_smooth_critic():
    torch.manual_seed(4)
    w = torch.randn(5, dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 5, dtype=torch.float64)
    critic = lambda v: (v.square() * w).sum(1)
    exact = (4 * x.square() * w.square()).sum(1).sqrt()
    fd = finite_difference_norm(critic, x, .001)
    torch.testing.assert_close(fd, exact, rtol=1e-9, atol=1e-9)
    eg = torch.autograd.grad(exact.square().sum(), w)[0]
    fg = torch.autograd.grad(fd.square().sum(), w)[0]
    torch.testing.assert_close(fg, eg, rtol=1e-9, atol=1e-9)


def test_async_exact_and_lazy_match_shared_bcap():
    torch.manual_seed(4)
    w = torch.randn(5, requires_grad=True)
    critic = lambda v: (v.square() * w).sum(1)
    real, fake = torch.randn(3, 5), torch.randn(3, 5)
    reg = GradRegularizer('b_cap', 1, lazy_k=4)
    cfg = {**DEFAULTS, 'reg_sync_stats': False}
    assert cifar_penalty(reg, critic, real, fake, 1, None, cfg) == 0
    reference = reg.penalty(critic, real, fake, 4)[0]
    result = cifar_penalty(reg, critic, real, fake, 4, None, cfg)
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(torch.autograd.grad(result, w)[0], torch.autograd.grad(reference, w)[0])
    every = GradRegularizer('b_cap', 1)
    torch.testing.assert_close(result, 4 * every.penalty(critic, real, fake, 4)[0])


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


def test_shared_fd_lazy_penalty_matches_original_cifar_formula():
    torch.manual_seed(3)
    w = torch.randn(5, dtype=torch.float64, requires_grad=True)
    critic = lambda x: (x.square() * w).sum(1)
    real, fake = torch.randn(3, 5, dtype=torch.float64), torch.randn(3, 5, dtype=torch.float64)
    reg = GradRegularizer('b_cap', 1, method='finite_difference', fd_eps=.05, lazy_k=4)
    cfg = {**DEFAULTS, 'reg_method':'finite_difference', 'reg_every':4, 'reg_sync_stats':False}
    expected = 2 * sum(torch.relu(finite_difference_norm(critic, x, .05)-1).square().mean() for x in (real,fake))
    result = cifar_penalty(reg,critic,real,fake,4,None,cfg)
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(torch.autograd.grad(result,w)[0],torch.autograd.grad(expected,w)[0])
    assert reg.penalty(critic,real,fake,4,collect_stats=False)[1] == {}

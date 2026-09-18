import copy

import pytest
import torch
from torch import nn

from experiments.train_cifar_ae_bundle import discriminator_loss, update_ema_bundle
from lib.image_ddgan import update_ema
from particlegan import GANLoss, GradientPenalty


@pytest.mark.parametrize('step', [7, 8])
def test_bundle_preserves_loss_and_nonzero_bcap_parameter_gradients(step):
    torch.manual_seed(24002)
    d = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1)).double()
    with torch.no_grad():
        d[-1].weight.mul_(30)
    real, fake = torch.randn(6, 3, dtype=torch.float64), torch.randn(6, 3, dtype=torch.float64)
    results, calls = [], []
    for bundle in (False, True):
        model = copy.deepcopy(d)
        count = []
        model.register_forward_hook(lambda *args: count.append(1))
        loss, penalty = discriminator_loss(model, real, fake, GANLoss(), GradientPenalty(lazy_k=8), step, bundle)
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        results.append((loss.detach(), penalty.detach(), grads))
        calls.append(len(count))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    assert results[0][1] > 0 if step == 8 else results[0][1] == 0
    assert calls == ([4, 1] if step == 8 else [2, 1])


def test_bundle_ema_copies_parameters_and_buffers():
    source = nn.Linear(3, 4)
    source.register_buffer('state', torch.tensor(3))
    a, b = copy.deepcopy(source), copy.deepcopy(source)
    with torch.no_grad():
        source.weight.add_(1)
        source.state.add_(1)
    update_ema(a, source, .995)
    update_ema_bundle([b], [source], .995)
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], atol=0, rtol=0)

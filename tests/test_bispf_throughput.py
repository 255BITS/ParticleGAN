"""Software parity and restoration checks; Forge owns trained quality gates."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.convolution_quintic import QuinticGraphCache, quintic_direction
from particlegan.optim.dualnorm import _batched_polar, polar_factor


@pytest.mark.parametrize('smoothing', [0., .001])
@pytest.mark.parametrize('shape', [(4, 3), (12, 24), (24, 12), (32, 32)])
def test_batched_svd_preserves_rank_cutoff_and_smoothing(shape, smoothing):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    values = torch.arange(8 * shape[0] * shape[1], device=device, dtype=torch.float64)
    matrices = values.sin().reshape(8, *shape)
    matrices[0].zero_()
    matrices[1, :, -1].zero_()
    expected = torch.stack([polar_factor(m, smoothing=smoothing) for m in matrices])
    torch.testing.assert_close(_batched_polar(matrices, smoothing=smoothing), expected, rtol=1e-10, atol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph verification')
def test_quintic_graph_replays_eager_bitwise_for_changing_inputs():
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        cache = QuinticGraphCache()
        for offset in [0., .3, 2.]:
            values = torch.arange(4 * 64 * 48, device='cuda', dtype=torch.float32)
            matrices = (values + offset).sin().reshape(4, 64, 48)
            expected = quintic_direction(matrices, smoothing=.001)
            actual = cache.direction(matrices, smoothing=.001)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        matrices[0, 0, 0] = float('nan')
        with pytest.raises(ValueError, match='finite'):
            cache.direction(matrices, smoothing=.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA optimizer dispatch')
@pytest.mark.parametrize('kind', [nn.Conv2d, nn.ConvTranspose2d])
def test_batched_optimizer_handles_groups_zero_slices_and_momentum(kind):
    module = kind(8, 12, (2, 3), groups=2, bias=False, device='cuda')
    native = deepcopy(module)
    recipe = get_recipe('bispf_optimized', optimizer_momentum=.5, constraint_geometry_mode='none')
    a = recipe.make_generator_optimizer(native)
    b = recipe.replace(optimizer_svd_backend='batched').make_generator_optimizer(module)
    for offset in [0., .3]:
        gradient = (torch.arange(module.weight.numel(), device='cuda') + offset).sin().reshape_as(module.weight)
        gradient[:, :, 0, 0] = 0
        native.weight.grad = gradient.clone()
        module.weight.grad = gradient.clone()
        a.step(); b.step()
    torch.testing.assert_close(module.weight, native.weight, rtol=2e-5, atol=2e-6)
    saved = b.state_dict()
    assert saved['dualnorm']['solver_version'] == 1
    b.load_state_dict(saved)
    with pytest.raises(ValueError):
        a.load_state_dict(saved)


def _trainer(precision, device):
    g = nn.Sequential(nn.Linear(4, 64), nn.Unflatten(1, (64, 1, 1)),
                      nn.ConvTranspose2d(64, 1, 4), nn.Sigmoid()).to(device)
    d = nn.Sequential(nn.Conv2d(1, 64, 2), nn.Flatten(), nn.Linear(64 * 9, 1)).to(device)
    deterministic_orthogonal_(g); deterministic_orthogonal_(d)
    recipe = get_recipe('bispf_optimized', z_dim=4, num_particles=16, batch_size=4,
                        total_steps=3, training_precision=precision, optimizer_svd_backend='batched')
    return GANTrainer(recipe, g, d, seed=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='BF16 training and restore')
def test_mixed_precision_is_training_only_and_checkpoint_continuation_is_exact():
    with torch.autograd.set_multithreading_enabled(False):
        trainer = _trainer('bf16_channels_last', 'cuda')
        observed = []
        hook = trainer.G[2].register_forward_hook(lambda m, a, out: observed.append(out.dtype))
        real = torch.arange(4 * 16, device='cuda', dtype=torch.float32).sin().reshape(4, 1, 4, 4)
        trainer.step(real)
        assert torch.bfloat16 in observed
        assert trainer.sample(4).dtype == torch.float32
        assert observed[-1] == torch.float32
        hook.remove()
        checkpoint = trainer.state_dict()
        assert '_quintic_graphs' not in checkpoint['optimizers'][0]
        restored = _trainer('bf16_channels_last', 'cuda')
        restored.load_state_dict(checkpoint)
        a, b = trainer.step(real), restored.step(real)
        torch.testing.assert_close(a['loss_g'], b['loss_g'], rtol=0, atol=0)
        for x, y in zip(trainer.G.parameters(), restored.G.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert all(p.dtype == torch.float32 for p in trainer.G.parameters())
        assert not trainer._in_training_step


def test_precision_is_explicit_and_does_not_change_default_recipe_identity():
    recipe = get_recipe('bispf_optimized')
    assert 'training_precision' not in recipe.to_dict()
    assert type(recipe)(**recipe.to_dict()) == recipe
    with pytest.raises(ValueError, match='training_precision'):
        recipe.replace(training_precision='fp16')
    from experiments.forge.techniques import validate_same_technique
    with pytest.raises(ValueError, match='training_precision'):
        validate_same_technique(recipe, recipe.replace(training_precision='bf16_channels_last'))

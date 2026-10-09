from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe, init
from particlegan.kernel_witness import kernel_witness_loss


def vectors():
    return torch.tensor([[-2., -1.], [-1., 1.], [0., 0.], [1., 2.], [2., 1.], [3., -2.]], dtype=torch.float64)


def test_null_gradient_controls_and_detached_target():
    real = vectors().requires_grad_()
    fake = real.detach().clone().requires_grad_()
    loss = kernel_witness_loss(fake, real)
    grad, target = torch.autograd.grad(loss, (fake, real), allow_unused=True)
    assert abs(float(loss.detach())) < 1e-14
    assert grad.abs().max() < 1e-14
    assert target is None
    assert kernel_witness_loss(fake + 2, real) > .05
    assert kernel_witness_loss(fake * 0, real) > .05
    assert kernel_witness_loss(fake * 2, real) > .01


def test_finite_derivative_equivariance_and_no_rng():
    before = torch.get_rng_state().clone()
    real = vectors()
    fake = (real + torch.tensor([.3, -.7])).requires_grad_()
    assert torch.autograd.gradcheck(lambda v: kernel_witness_loss(v, real), (fake,))
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=real.dtype)
    expected = kernel_witness_loss(fake, real)
    actual = kernel_witness_loss((fake @ rotation) * 7 + 2, (real @ rotation) * 7 + 2)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(kernel_witness_loss(fake.flip(0), real.flip(0)), expected)
    assert torch.equal(before, torch.get_rng_state())


def test_atomic_frame_finite_and_long_range_force():
    real = torch.zeros(4, 2, dtype=torch.float64)
    fake = torch.full((4, 2), 100., dtype=torch.float64, requires_grad=True)
    loss = kernel_witness_loss(fake, real)
    grad, = torch.autograd.grad(loss, fake)
    assert torch.isfinite(grad).all() and (grad > 0).all()
    same = real.clone().requires_grad_()
    null = kernel_witness_loss(same, real)
    assert null == 0
    assert torch.autograd.grad(null, same)[0].count_nonzero() == 0


@pytest.mark.parametrize('weight', [-1, True, float('nan'), float('inf')])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        Recipe(kernel_witness_weight=weight)


def test_forge_mechanism_and_unsupported_fixture_are_explicit():
    from experiments.forge.techniques import validate_same_technique
    from experiments.forge.behavior_adapters import behavior_preflight
    from experiments.forge.contracts import read_json
    root=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match='raw_output_multiscale_cauchy_mmd'):
        validate_same_technique(get_recipe('bcap'), get_recipe('bcap', kernel_witness_weight=1))
    task=read_json(root/'configs/forge/tasks/two_pole.json')
    blockers=behavior_preflight(task,dict(recipe_preset='bcap',recipe_overrides={'kernel_witness_weight':1}))
    assert any('does not consume' in x and 'kernel_witness_weight' in x for x in blockers)


def trainer(weight):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        g = nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 2))
        d = nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 1))
        init.deterministic_orthogonal_(g)
        init.deterministic_orthogonal_(d)
        recipe = get_recipe('bcap', num_particles=12, z_dim=2, batch_size=6, total_steps=4,
                            kernel_witness_weight=weight)
        return GANTrainer(recipe, g, d, seed=0)


def test_public_step_checkpoint_streams_and_default_compatibility():
    base, changed = trainer(0), trainer(1)
    real = vectors().float()
    a, b = base.step(real), changed.step(real)
    assert 'kernel_witness' not in a and 'kernel_witness' in b
    assert torch.equal(a['loss_d'], b['loss_d'])
    assert any(not torch.equal(p, q) for p, q in zip(base.G.parameters(), changed.G.parameters()))
    for name in changed._STREAMS:
        left, right = getattr(base, name), getattr(changed, name)
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.equal(left.get_state(), right.get_state())
    saved = deepcopy(changed.state_dict())
    restored = trainer(1)
    restored.load_state_dict(saved)
    changed.step(real)
    restored.step(real)
    for role in ('G', 'D', 'prior'):
        for name, value in getattr(changed, role).state_dict().items():
            assert torch.equal(value, getattr(restored, role).state_dict()[name])
    for name in changed._STREAMS:
        left, right = getattr(changed, name), getattr(restored, name)
        if left is not None:
            assert torch.equal(left.get_state(), right.get_state())
    legacy = deepcopy(base.state_dict())
    legacy['recipe'].pop('kernel_witness_weight', None)
    trainer(0).load_state_dict(legacy)

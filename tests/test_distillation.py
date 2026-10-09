import pytest
import torch
from torch import nn
from particlegan import GANTrainer, get_recipe
from particlegan.distillation import distillation_loss
from experiments.forge.api import task_policy_blockers
from tests.test_training import assert_checkpoint_equal


def test_empirical_null_and_real_detachment():
    real = torch.tensor([[-2., .1], [-1., .2], [1., -.2], [2., -.1]], dtype=torch.float64, requires_grad=True)
    fake = real.detach().clone().requires_grad_()
    loss = distillation_loss(fake, real)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(fake.grad, torch.zeros_like(fake))
    assert real.grad is None


def test_single_cell_resolves_mass_location_shape_and_detects_collapse():
    real = torch.tensor([[-1.], [1.]], dtype=torch.float64)
    _, null = distillation_loss(real, real, cells=1, return_parts=True)
    _, shifted = distillation_loss(real + 1, real, cells=1, return_parts=True)
    loss, collapsed = distillation_loss(torch.zeros_like(real), real, cells=1, return_parts=True)
    assert all(v == 0 for v in null.values())
    assert shifted['mass'] == 0 and shifted['location'] == 1 and shifted['shape'] == 1
    assert collapsed['mass'] == 0 and collapsed['location'] == 0
    assert collapsed['shape'] == 1 and loss == 1


def test_rotations_and_units_preserve_witness():
    real = torch.tensor([[-2., .1], [-.8, .3], [1., -.2], [2., -.1]], dtype=torch.float64)
    fake = real + torch.tensor([.1, -.3], dtype=torch.float64)
    rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
    base = distillation_loss(fake, real, cells=3)
    transformed = distillation_loss(7 * fake @ rotation + 3, 7 * real @ rotation + 3, cells=3)
    torch.testing.assert_close(base, transformed, rtol=1e-10, atol=1e-10)


def test_atomic_law_and_numerical_derivative():
    real = torch.zeros(4, 2, dtype=torch.float64)
    fake = torch.ones_like(real).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: distillation_loss(x, real, cells=3), (fake,))
    assert distillation_loss(real, real) == 0


def build(weight):
    torch.manual_seed(4)
    recipe = get_recipe('bcap', total_steps=3, num_particles=8, z_dim=2,
                        batch_size=4, sigma_rel=0, distillation_weight=weight)
    return GANTrainer(recipe, nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)),
                      nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), seed=0)


def test_public_trainer_resume_and_sampling_streams():
    trainer = build(1)
    real = torch.tensor([[-2., .1], [-1., .2], [1., -.2], [2., -.1]])
    before_rng = torch.get_rng_state().clone()
    stats = trainer.step(real)
    assert 'distillation_shape' in stats and torch.equal(before_rng, torch.get_rng_state())
    checkpoint = trainer.state_dict()
    trainer.step(real)
    expected = trainer.state_dict()
    restored = build(1)
    restored.load_state_dict(checkpoint)
    restored.sample(5)
    # Evaluation draws are isolated; restore all state for exact resume check.
    restored.load_state_dict(checkpoint)
    restored.step(real)
    assert_checkpoint_equal(expected, restored.state_dict())
    with pytest.raises(ValueError, match='recipe'):
        build(0).load_state_dict(checkpoint)


def test_default_serialization_and_component_blocker():
    assert 'distillation_weight' not in get_recipe('bcap').to_dict()
    assert 'distillation_cells' not in get_recipe('bcap').to_dict()
    candidate = {'recipe_preset': 'bcap', 'recipe_overrides': {'distillation_weight': 1.}}
    task = {'id': 'two_pole', 'execution': {'execution_path': 'public_components'}}
    assert 'does not consume' in task_policy_blockers(task, candidate)[0]
    assert not task_policy_blockers(task, {'recipe_preset': 'bcap'})


def test_matched_arms_consume_identical_training_streams():
    real = torch.tensor([[-2., .1], [-1., .2], [1., -.2], [2., -.1]])
    control, candidate = build(0), build(1)
    for trainer in (control, candidate):
        trainer.step(real)
        trainer.step(real)
    left, right = control.state_dict(), candidate.state_dict()
    for stream in left['streams']:
        assert torch.equal(left['streams'][stream], right['streams'][stream])
    assert any(not torch.equal(left['models']['G'][key], right['models']['G'][key])
               for key in left['models']['G'])


@pytest.mark.parametrize('value', [-1, float('inf'), True])
def test_reject_invalid_weight(value):
    with pytest.raises(ValueError, match='distillation_weight'):
        get_recipe('bcap', distillation_weight=value)

"""Bounded software contracts, not task-quality experiments or gate evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from particlegan import GANTrainer, get_recipe
from particlegan.conditional_transport import OutputMarginalTransport
from particlegan.sinkhorn_transport import auxiliary_rows, sinkhorn_transport_loss
from experiments.forge.api import TRAINER_STREAM_BINDINGS
from experiments.forge.rng import NamedStreams
from experiments.forge.techniques import recipe_field_active
from test_bcap_integration_core import _trainer, _batch, _equal


def _active_trainer():
    base = _trainer(active=False)
    recipe = replace(base.recipe, kinetic_transport_mode='sinkhorn', kinetic_transport_weight=1.)
    streams = NamedStreams(0)
    return GANTrainer(recipe, base.G, base.D, prior=base.prior, seed=0,
        **{name: streams.generator(family, component=component, purpose=purpose)
           for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()})


def test_debiasing_null_force_detached_target_symmetry_and_finite_derivative():
    real = torch.tensor([[-1.1, .2], [-.2, -.5], [.4, .6], [1.3, -.2]], dtype=torch.float64,
                        requires_grad=True)
    fake = real.detach().clone().requires_grad_()
    rng = torch.get_rng_state().clone()
    null = sinkhorn_transport_loss(fake, real)
    assert null.item() == 0
    gradient, target = torch.autograd.grad(null, (fake, real), allow_unused=True)
    torch.testing.assert_close(gradient, torch.zeros_like(fake), atol=1e-14, rtol=0)
    assert target is None
    fake = (fake.detach() + .12).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: sinkhorn_transport_loss(x, real), (fake,))
    # Scale is real-owned; swapping clouds of equal variance tests solver symmetry.
    a = sinkhorn_transport_loss(fake, real)
    b = sinkhorn_transport_loss(real.detach(), fake.detach())
    assert a.item() == pytest.approx(b.item(), abs=1e-14)
    assert a.item() > 0
    assert torch.equal(rng, torch.get_rng_state())


def test_self_term_prevents_entropic_contraction_and_units_preserve_loss():
    real = torch.tensor([[-1.], [-.6], [.6], [1.]], dtype=torch.float64)
    fake = (.2 * real).requires_grad_()
    value = sinkhorn_transport_loss(fake, real)
    gradient, = torch.autograd.grad(value, fake)
    assert bool((gradient * fake < 0).all())
    assert sinkhorn_transport_loss(7 * fake + 4, 7 * real + 4).item() == pytest.approx(value.item())
    assert sinkhorn_transport_loss(fake.flip(0), real.roll(2, 0)).item() == pytest.approx(value.item())


def test_auxiliary_high_dimensional_panel_uses_original_rows_without_rng():
    real = torch.arange(256 * 140, dtype=torch.float64).reshape(256, 140) / 10000
    fake = (real + .1).requires_grad_()
    stats = {}
    before = torch.get_rng_state().clone()
    value = sinkhorn_transport_loss(fake, real, stats=stats)
    gradient, = torch.autograd.grad(value, fake)
    rows = auxiliary_rows(torch.arange(256)[:, None], 128).flatten()
    assert stats['rows'] == 128 and stats['input_rows'] == 256 and stats['iterations'] == 72
    assert torch.isfinite(value) and bool(torch.isfinite(gradient).all())
    excluded = torch.ones(256, dtype=torch.bool)
    excluded[rows] = False
    assert torch.equal(gradient[excluded], torch.zeros_like(gradient[excluded]))
    assert torch.equal(before, torch.get_rng_state())


def test_active_public_trainer_exact_resume_and_counter_validation():
    original = _active_trainer()
    original.step(_batch())
    saved = deepcopy(original.state_dict())
    resumed = _active_trainer()
    resumed.load_state_dict(saved)
    _equal(original.step(_batch()), resumed.step(_batch()))
    _equal(original.state_dict(), resumed.state_dict())
    stats = original.state_dict()['component_transport']
    assert stats['calls'] == 2 and stats['iterations'] == 144 and stats['auxiliary_rows'] == 8
    broken = deepcopy(saved)
    broken['component_transport']['calls'] = -1
    with pytest.raises(ValueError, match='Sinkhorn audit'):
        resumed.load_state_dict(broken)


def test_conditional_consumer_active_checkpoint_and_disabled_compatibility():
    inactive = get_recipe('bcap')
    packet = inactive.to_dict()
    assert not {'kinetic_transport_mode', 'sinkhorn_epsilon', 'sinkhorn_iterations',
                'sinkhorn_max_samples'} & packet.keys()
    assert not recipe_field_active('sinkhorn_epsilon', inactive)
    active = replace(inactive, kinetic_transport_weight=1., kinetic_transport_mode='sinkhorn')
    consumer = OutputMarginalTransport(active)
    consumer.add(torch.tensor(0.), _batch().requires_grad_(), _batch())
    restored = OutputMarginalTransport(active)
    restored.load_state_dict(consumer.state_dict())
    _equal(consumer.state_dict(), restored.state_dict())
    assert consumer.state_dict()['sinkhorn']['calls'] == 1
    trainer = _trainer(active=False)
    assert 'component_transport' not in trainer.state_dict()
    restored_trainer = _trainer(active=False)
    restored_trainer.load_state_dict(trainer.state_dict())
    _equal(trainer.step(_batch()), restored_trainer.step(_batch()))


@pytest.mark.parametrize('field,value', [('sinkhorn_epsilon', 0), ('sinkhorn_iterations', 0),
                                        ('sinkhorn_max_samples', 1), ('kinetic_transport_mode', 'bad')])
def test_recipe_rejects_invalid_finite_solver_contract(field, value):
    with pytest.raises(ValueError):
        get_recipe('bcap', **{field: value})

"""Numerical, inactive API and exact-resume controls; no qualification runs."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.forge.api import FormulationContext
from experiments.forge.boundaries import RECIPE_FIELD_OWNERS
from experiments.forge.techniques import technique_signature
from particlegan import Recipe
from particlegan.anisotropic_transport import (_coordinates, _real_geometry, _distances,
                                             anisotropic_transport_local_loss, new_geometry_stats)
from particlegan.conditional_transport import OutputMarginalTransport
from particlegan.kinetic_transport import kinetic_transport_local_loss
from particlegan.recipe_compat import without_default_additions


def panel(n=32, d=2, dtype=torch.float64):
    row = torch.arange(n, dtype=dtype)[:, None]
    col = torch.arange(1, d + 1, dtype=dtype)[None, :]
    return torch.sin(row * col * .7) / col + torch.cos(row * col * .13) * .1


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def test_geometry_bounds_rank_and_direct_mahalanobis_equation():
    y = panel(2048, 125)
    x, compressed = _coordinates(y + .1, y)
    assert x.shape == (2048, 8)
    anchors, precision = _real_geometry(compressed)
    assert anchors.shape == (64, 8) and precision.shape == (64, 8, 8)
    assert torch.linalg.eigvalsh(precision).min() > 0
    actual = _distances(x[:3], anchors[:2], precision[:2])
    delta = x[:3, None] - anchors[None, :2]
    expected = torch.einsum('naj,ajk,nak->na', delta, precision[:2], delta)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_moment_identity_detached_real_and_no_rng_consumption():
    real = panel().requires_grad_()
    fake = (real.detach() + torch.tensor([.08, -.02])).requires_grad_()
    before = torch.random.get_rng_state().clone()
    assert anisotropic_transport_local_loss(real.detach(), real) == 0
    loss = anisotropic_transport_local_loss(fake, real)
    loss.backward()
    assert real.grad is None and torch.isfinite(fake.grad).all() and fake.grad.abs().sum() > 0
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.autograd.gradcheck(lambda value: anisotropic_transport_local_loss(value, real), (fake,))


def test_low_dim_rotation_scale_translation_invariance():
    y, x = panel(), panel() + torch.tensor([.05, .12])
    rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=y.dtype)
    expected = anisotropic_transport_local_loss(x, y)
    actual = anisotropic_transport_local_loss((x @ rotation) * 3 + 7, (y @ rotation) * 3 + 7)
    torch.testing.assert_close(expected, actual, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('n,d', [(2, 1), (5, 4), (128, 2), (256, 125)])
def test_duplicate_and_rank_deficient_real_panels_are_finite(n, d):
    real = torch.zeros(n, d)
    fake = torch.full_like(real, 1e-8, requires_grad=True)
    loss = anisotropic_transport_local_loss(fake, real)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(fake.grad).all()


def test_default_packet_isotropic_loss_and_disabled_consumer():
    recipe = Recipe(kinetic_transport_local_weight=1.)
    assert torch.equal(recipe.kinetic_transport_local_loss(panel() + .1, panel()),
                       kinetic_transport_local_loss(panel() + .1, panel()))
    assert 'kinetic_transport_local_geometry' not in without_default_additions(recipe.to_dict())
    consumer = OutputMarginalTransport(Recipe())
    marker = object()
    assert consumer.add(marker, None, None) is marker
    assert consumer.active_calls == 0
    assert RECIPE_FIELD_OWNERS['kinetic_transport_local_geometry'] == 'technique'


def build(active, device='cpu'):
    overrides = dict(num_particles=12, z_dim=2, batch_size=8, total_steps=4,
                     optimizer_family='dualnorm', optimizer_smoothing=.001)
    if active:
        overrides.update(kinetic_transport_local_weight=1.,
                         kinetic_transport_local_geometry='anisotropic_knn_v1')
    context = FormulationContext(recipe_preset='bcap', recipe_overrides=overrides, device=device)
    g = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)),
                          component='generator')
    d = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)),
                          component='discriminator')
    g, d = g.to(device), d.to(device)
    return context, context.build_trainer(g, d)


def test_public_active_exact_resume_streams_and_inactive_old_checkpoint():
    context, trainer = build(True)
    trainer.step(panel(8, dtype=torch.float32))
    saved = deepcopy(trainer.state_dict())
    other_context, resumed = build(True)
    resumed.load_state_dict(saved)
    a = trainer.step(panel(8, dtype=torch.float32) + .01)
    b = resumed.step(panel(8, dtype=torch.float32) + .01)
    equal(a, b)
    equal(trainer.state_dict(), resumed.state_dict())
    assert 'kinetic_transport_local' in a
    assert technique_signature(context.recipe)['mechanisms']['real_neighborhood_covariance_features']
    _, inactive = build(False)
    before = deepcopy(inactive.state_dict())
    with pytest.raises(ValueError):
        inactive.load_state_dict(saved)
    equal(before, inactive.state_dict())
    _, old_restored = build(False)
    old_restored.load_state_dict(before)
    equal(before, old_restored.state_dict())


def test_validation_rejects_undeclared_geometry_or_inactive_selector():
    with pytest.raises(ValueError, match='geometry'):
        Recipe(kinetic_transport_local_geometry='unknown')
    with pytest.raises(ValueError, match='requires positive'):
        Recipe(kinetic_transport_local_geometry='anisotropic_knn_v1')


def test_word_like_atoms_expose_zero_covariance_and_vanishing_force():
    real = torch.eye(5).repeat_interleave(32, dim=0)
    fake = torch.full_like(real, .2, requires_grad=True)
    stats = new_geometry_stats()
    loss = anisotropic_transport_local_loss(fake, real, diagnostics=stats)
    loss.backward()
    assert loss == 1 and torch.count_nonzero(fake.grad) == 0
    assert stats['zero_covariance_anchors'] == stats['anchors'] == 64
    assert stats['minimum_ridge'] > 0 and stats['maximum_condition_bound'] == 1


def test_active_consumer_resume_and_invalid_counters_rejected_before_mutation():
    recipe = Recipe(kinetic_transport_local_weight=1., kinetic_transport_local_geometry='anisotropic_knn_v1')
    a, b = OutputMarginalTransport(recipe), OutputMarginalTransport(recipe)
    a.add(torch.zeros(()), panel() + .1, panel())
    saved = a.state_dict()
    b.load_state_dict(saved)
    equal(a.state_dict(), b.state_dict())
    corrupted = deepcopy(saved)
    corrupted['anisotropic_geometry']['calls'] += 1
    with pytest.raises(ValueError, match='calls'):
        b.load_state_dict(corrupted)
    equal(saved, b.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA exact-resume control')
def test_public_cuda_active_exact_resume():
    _, a = build(True, 'cuda:0')
    real = panel(8, dtype=torch.float32).cuda()
    a.step(real)
    saved = deepcopy(a.state_dict())
    _, b = build(True, 'cuda:0')
    b.load_state_dict(saved)
    equal(a.step(real), b.step(real))
    equal(a.state_dict(), b.state_dict())


def test_original_word_consumer_active_exact_resume():
    from benchmarks.toy_audit.api_images import WordFixture
    from experiments.forge.state import state_digest
    from experiments.forge.planning import load_idea
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    overrides = dict(load_idea(root, 'bcap-three-phase-incumbent-v1')['recipe_overrides'])
    overrides.update(num_particles=5, z_dim=2, batch_size=256, total_steps=20000,
                     kinetic_transport_local_weight=1., kinetic_transport_local_geometry='anisotropic_knn_v1')
    def word():
        context = FormulationContext(recipe_preset='bcap', recipe_overrides=overrides,
            prior=dict(kind='particle_cloud', sigma=0., standardize=False, learnable=True,
                       exception_reason='Original finite-word host software control'),
            seed=0, execution_path='public_components', component_transport='output_marginal_v1')
        return context, WordFixture(device='cpu', seed=0, recipe_name=None, max_steps=3, components=context)
    context_a, a = word()
    a.step()
    saved, named = deepcopy(a.state_dict()), deepcopy(context_a.streams.state_dict())
    result_a = [a.step(), a.step()]
    context_b, b = word()
    context_b.streams.load_state_dict(named)
    b.policy.load_state_dict(saved['api_state'])
    b.data_generator.set_state(saved['data_generator'])
    b.restore_component_transport(saved['component_transport'])
    assert state_digest([b.step(), b.step()]) == state_digest(result_a)
    equal(a.state_dict(), b.state_dict())
    equal(context_a.streams.state_dict(), context_b.streams.state_dict())
    assert a.transport.geometry_stats['calls'] == a.transport.active_calls == 3
    assert a.transport.geometry_stats['compressed_calls'] == 3

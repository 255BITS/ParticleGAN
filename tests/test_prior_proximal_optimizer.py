"""Direct tests of ``PriorProximalOptimizer`` (full DualNorm plus the spring).

test_prior_proximal.py covers the composed recipe paths; these construct the
class itself, pin the factory mapping and the constructor/step/checkpoint
refusals, and check the restoration law on a known displacement.
"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.dualnorm import NormalizedOptimizer
from particlegan.optim.prior_proximal import (
    PriorProximalDirectionBlendOptimizer, PriorProximalOptimizer, proximal_optimizer_class,
)


def build(table, *, strength=1., lr=.1, momentum=0., **group):
    return PriorProximalOptimizer([dict(params=[table], role="prior", lr=lr, **group)],
                                  family="dualnorm", momentum=momentum, prior_proximal_strength=strength)


def displaced_table():
    table = nn.Parameter(torch.tensor([[1., 0.], [0., 2.], [-1., 1.]], dtype=torch.float64))
    return table, table.detach().clone()


def test_factory_maps_each_supported_base_and_rejects_others():
    assert proximal_optimizer_class(NormalizedOptimizer) is PriorProximalOptimizer
    assert proximal_optimizer_class(DirectionBlendOptimizer) is PriorProximalDirectionBlendOptimizer
    with pytest.raises(ValueError, match="unsupported prior proximal optimizer composition"):
        proximal_optimizer_class(torch.optim.Adam)


def test_public_recipe_without_geometry_builds_this_class():
    table = nn.Parameter(torch.zeros(4, 2))
    optimizer = get_recipe("bcap", prior_proximal_strength=.5, constraint_geometry_mode="none") \
        .make_generator_optimizer([dict(params=[table], role="prior")])
    assert type(optimizer) is PriorProximalOptimizer
    assert optimizer.prior_proximal_strength == .5


def test_owned_rows_are_pulled_toward_their_initial_locations_with_magnitude():
    table, initial = displaced_table()
    optimizer = build(table, strength=3., lr=.2)
    with torch.no_grad():
        table.add_(torch.tensor([[2., 0.], [0., -4.], [5., 5.]], dtype=torch.float64))
    displaced = table.detach().clone()
    optimizer.set_sampled_rows(table, torch.tensor([0, 1]))
    table.grad = torch.zeros_like(table)  # the base update is zero; only the spring acts
    optimizer.step()
    fraction = .2 * 3. / (1. + .2 * 3.)
    torch.testing.assert_close(table.detach()[:2], torch.lerp(displaced[:2], initial[:2], fraction))
    assert torch.equal(table.detach()[2], displaced[2])
    stats = optimizer.prior_proximal_stats
    assert stats["steps"] == 1 and stats["rows"] == 2
    assert stats["max_restoration"] == pytest.approx(4. * fraction)
    assert stats["sum_squared_restoration"] == pytest.approx((4. + 16.) * fraction ** 2)


@pytest.mark.parametrize("strength", [0., -1., float("nan"), float("inf"), "1"])
def test_constructor_rejects_invalid_strength(strength):
    table, _ = displaced_table()
    with pytest.raises(ValueError, match="finite and positive"):
        build(table, strength=strength)


def test_constructor_requires_zero_momentum_sampled_row_dualnorm_and_finite_initial_rows():
    table, _ = displaced_table()
    with pytest.raises(ValueError, match="zero-momentum full DualNorm"):
        build(table, momentum=.9)
    with pytest.raises(ValueError, match="sampled latent-row ownership"):
        build(table, sampled_rows_required=False)
    with torch.no_grad():
        table[0, 0] = float("nan")
    with pytest.raises(ValueError, match="initial locations must be finite"):
        build(table)


def test_late_prior_groups_and_closures_are_refused():
    table, _ = displaced_table()
    optimizer = build(table)
    with pytest.raises(ValueError, match="after all latent tables are initialized"):
        optimizer.add_param_group(dict(params=[nn.Parameter(torch.zeros(2, 2))], role="prior"))
    with pytest.raises(ValueError, match="explicit backward"):
        optimizer.step(lambda: 0.)


def test_checkpoint_roundtrip_restores_the_original_reference_and_statistics():
    table, initial = displaced_table()
    optimizer = build(table)
    with torch.no_grad():
        table.add_(1.)
    optimizer.set_sampled_rows(table, torch.tensor([2]))
    table.grad = torch.zeros_like(table)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    meta = saved["prior_proximal"]
    assert (meta["schema"], meta["strength"], meta["order"]) == (1, 1., "after_base_update")
    assert torch.equal(meta["initial_locations"][0], initial)

    other = nn.Parameter(table.detach().clone() + 10)  # a later construction has a different reference
    restored = build(other)
    restored.load_state_dict(saved)
    assert torch.equal(restored._prior_initial[other], initial)
    assert restored.prior_proximal_stats == optimizer.prior_proximal_stats


@pytest.mark.parametrize("corrupt,match", [
    (lambda s: s.pop("prior_proximal"), "prior proximal checkpoint settings"),
    (lambda s: s["prior_proximal"].update(strength=2.), "prior proximal checkpoint settings"),
    (lambda s: s["prior_proximal"].update(order="before_base_update"), "prior proximal checkpoint settings"),
    (lambda s: s["prior_proximal"]["stats"].update(rows=-1), "checkpoint statistics"),
    (lambda s: s["prior_proximal"]["stats"].update(max_restoration=float("nan")), "checkpoint statistics"),
    (lambda s: s["prior_proximal"].update(initial_locations=[torch.zeros(2, 2, dtype=torch.float64)]),
     "initial reference"),
])
def test_checkpoint_validation_rejects_corruption_without_mutation(corrupt, match):
    table, initial = displaced_table()
    optimizer = build(table)
    saved = deepcopy(optimizer.state_dict())
    corrupt(saved)
    with pytest.raises(ValueError, match=match):
        optimizer.load_state_dict(saved)
    assert torch.equal(optimizer._prior_initial[table], initial)
    assert optimizer.prior_proximal_stats["steps"] == 0

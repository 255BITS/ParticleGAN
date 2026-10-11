"""Public recipe contract for two isolated BCAP capabilities."""
from copy import deepcopy
import pytest
import torch
from particlegan import get_recipe
from particlegan.optim.constraint_geometry import constraint_geometry_backward


def test_explicit_inherited_prior_monitoring_preserves_protected_update():
    def update(value):
        recipe = get_recipe('bcap', optimizer_prior_smoothing=value)
        weight = torch.nn.Parameter(torch.tensor([[.2, -.1], [.3, .4]]))
        table = torch.nn.Parameter(torch.tensor([[.1, .2], [.3, .4]]))
        optimizer = recipe.make_generator_optimizer([
            dict(params=[weight], role='generator'), dict(params=[table], role='prior', lr=.03)], latent_table=table)
        optimizer.set_sampled_rows(table, torch.tensor([0]))
        loss = weight.square().sum() + table[0].square().sum()
        constraint_geometry_backward(loss, optimizer, [loss])
        optimizer.step()
        return weight.detach(), table.detach(), optimizer
    a, b = update(None), update(.001)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert b[2].prior_smoothing_stats['steps'] == 1
    assert b[2].constraint_geometry_stats['projected_steps'] == 0


def test_recipe_builds_hybrid_penalty_and_retains_cap():
    recipe = get_recipe('bcap', reg_zero_coeff=.05)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(.2)
    optimizer = recipe.make_critic_optimizer(critic)
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
    value = penalty(critic, torch.zeros(3, 1), torch.ones(3, 1))
    value.backward()
    assert float(value.detach()) == pytest.approx(.05 * .2 ** 2)
    assert float(critic.weight.grad) == pytest.approx(2 * .05 * .2)
    assert penalty.last_stats['cap_pen'] == 0.
    assert penalty.last_stats['zero_pen'] > 0.


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf'), True])
def test_recipe_rejects_invalid_new_scales(value):
    for field in ('reg_zero_coeff', 'optimizer_prior_smoothing'):
        with pytest.raises(ValueError):
            get_recipe('bcap', **{field: value})


def test_disabled_settings_keep_original_recipe_packet():
    recipe = get_recipe('bcap')
    assert 'reg_zero_coeff' not in recipe.to_dict()
    assert 'optimizer_prior_smoothing' not in recipe.to_dict()
    with pytest.raises(ValueError):
        get_recipe('ka2', reg_zero_coeff=.05)
    with pytest.raises(ValueError):
        get_recipe('bcap_adam', optimizer_prior_smoothing=.01)


@pytest.mark.parametrize('geometry', ['none', 'direction_blend'])
def test_smoothing_and_proximal_spring_share_checkpoint_but_distinct_motion(geometry):
    recipe = get_recipe('bcap', constraint_geometry_mode=geometry,
                        optimizer_prior_smoothing=.01, prior_proximal_strength=.2)
    initial = torch.tensor([[.1, .2], [.3, .4]], dtype=torch.float64)

    def build(value):
        table = torch.nn.Parameter(value.clone())
        optimizer = recipe.make_generator_optimizer(
            [dict(params=[table], role='prior', lr=.03)], latent_table=table)
        return table, optimizer

    def step(table, optimizer):
        optimizer.zero_grad()
        optimizer.set_sampled_rows(table, torch.tensor([0]))
        loss = table[0].square().sum()
        constraint_geometry_backward(loss, optimizer, [loss])
        optimizer.step()

    table, optimizer = build(initial)
    step(table, optimizer)
    assert torch.equal(table[1], initial[1])
    base_energy = optimizer.prior_smoothing_stats['optimizer_displacement_squared_sum']
    actual_energy = float((table.detach() - initial).square().sum())
    assert actual_energy == pytest.approx(base_energy / (1 + .03 * .2) ** 2)
    assert actual_energy < base_energy
    saved = deepcopy(optimizer.state_dict())
    restored_table, restored = build(table.detach())
    restored.load_state_dict(saved)
    assert restored.prior_smoothing_stats == optimizer.prior_smoothing_stats
    assert restored.prior_proximal_stats == optimizer.prior_proximal_stats
    step(table, optimizer)
    step(restored_table, restored)
    assert torch.equal(table, restored_table)
    assert restored.prior_smoothing_stats == optimizer.prior_smoothing_stats
    assert restored.prior_proximal_stats == optimizer.prior_proximal_stats

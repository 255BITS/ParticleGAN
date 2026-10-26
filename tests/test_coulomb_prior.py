"""Force, state and public-host contracts; these do not qualify Tier 2 science."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import CoulombPriorRegularizer, Recipe
from particlegan.coulomb_prior import CENTER_BUFFER, RADIUS_BUFFER
from experiments.forge.api import FormulationContext
from experiments.forge.behavior_adapters import run_behavior
from experiments.forge.priors import prior_policy_receipt, validate_prior_policy
from experiments.forge.techniques import recipe_field_active, validate_same_technique

ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def context(*, particles=12, **settings):
    leader = json.loads((ROOT / 'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())
    overrides = dict(leader['recipe_overrides'])
    overrides.update(num_particles=particles, z_dim=2, batch_size=4, total_steps=4)
    overrides.update(settings)
    return FormulationContext(recipe_preset='bcap', recipe_overrides=overrides,
        prior=dict(kind='mog', sigma=.025, standardize=False),
        prior_contract='recipe_owned_v1', seed=0)


def trainer(c):
    g = c.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component='generator')
    d = c.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component='discriminator')
    return c.build_trainer(g, d)


def test_two_particle_energy_and_analytic_force_are_exact():
    rows = torch.tensor([[-.2, 0.], [.2, 0.]], dtype=torch.float64, requires_grad=True)
    regularizer = CoulombPriorRegularizer(center=torch.zeros(2, dtype=rows.dtype), radius=rows.new_tensor(2.),
        weight=.3, softening=.1, confinement=.35, neighbors=8)
    actual = regularizer(rows)
    # q separation .2, mean q squared norm .01.
    expected = .3 * (rows.new_tensor(.05).rsqrt() + .35 / 2 * .01)
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    grad = torch.autograd.grad(actual, rows)[0]
    expected_right = .3 * (-.4 / (4 * .05**1.5) + .35 * .2 / (2 * 4))
    assert grad[1, 0].item() == pytest.approx(expected_right, abs=1e-13)
    assert grad[0, 0].item() == pytest.approx(-expected_right, abs=1e-13)
    assert torch.equal(grad[:, 1], torch.zeros(2, dtype=rows.dtype))


def test_confinement_restores_translation_and_prevents_unbounded_expansion():
    loss = CoulombPriorRegularizer(center=torch.zeros(2), radius=torch.tensor(1.))
    rows = torch.tensor([[-1., 0.], [1., 0.]], requires_grad=True)
    translated = rows + torch.tensor([0., 3.])
    gradient = torch.autograd.grad(loss(translated), rows)[0]
    assert gradient[:, 1].sum().item() == pytest.approx(.35 * 3)
    assert loss(rows * 100) > loss(rows * 10) > loss(rows)


def test_collisions_empty_and_single_row_are_finite_without_self_pairs():
    loss = CoulombPriorRegularizer(center=torch.zeros(2), radius=torch.tensor(1.))
    rows = torch.zeros(3, 2, requires_grad=True)
    assert loss(rows).item() == pytest.approx(10.)
    assert torch.equal(torch.autograd.grad(loss(rows), rows)[0], torch.zeros_like(rows))
    one = torch.tensor([[2., 0.]], requires_grad=True)
    assert loss(one).item() == pytest.approx(.7)  # confinement only
    assert loss(one[:0]).item() == 0


@pytest.mark.parametrize('kwargs', [dict(softening=0), dict(confinement=0), dict(neighbors=65),
    dict(neighbors=True), dict(weight=-1), dict(softening=float('nan'))])
def test_invalid_force_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        CoulombPriorRegularizer(**kwargs)


def test_20k_rows_use_bounded_pair_count_and_no_rng(monkeypatch):
    calls = []
    original = torch.Tensor.roll
    def observed(tensor, offset, dim):
        calls.append(offset)
        return original(tensor, offset, dim)
    monkeypatch.setattr(torch.Tensor, 'roll', observed)
    rows = torch.arange(40_000, dtype=torch.float32).reshape(20_000, 2) / 40_000
    rows.requires_grad_(True)
    loss = CoulombPriorRegularizer(center=torch.zeros(2), radius=torch.tensor(1.), neighbors=8)
    before = torch.get_rng_state().clone()
    value = loss(rows)
    assert len(calls) == 8 and all(1 <= offset < len(rows) for offset in calls)
    assert torch.isfinite(torch.autograd.grad(value, rows)[0]).all()
    assert torch.equal(before, torch.get_rng_state())


def test_initial_reference_after_public_init_is_fixed_and_checkpointed():
    c = context(prior_regularizer='coulomb', prior_reg=.01)
    t = trainer(c)
    center, radius = getattr(t.prior, CENTER_BUFFER), getattr(t.prior, RADIUS_BUFFER)
    assert torch.equal(center, t.prior.z.detach().mean(0))
    assert torch.equal(radius, (t.prior.z.detach() - center).square().sum(1).mean().sqrt())
    initial_center, initial_radius = center.clone(), radius.clone()
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    t.step(real)
    t.recipe.apply_prior_policy(t.prior)
    assert torch.equal(center, initial_center) and torch.equal(radius, initial_radius)
    saved = c.state_dict()
    t.step(real)
    expected = c.state_dict()
    restored = context(prior_regularizer='coulomb', prior_reg=.01)
    other = trainer(restored)
    # Simulate a freshly constructed reference that differs from the saved one.
    with torch.no_grad():
        getattr(other.prior, CENTER_BUFFER).add_(100)
        getattr(other.prior, RADIUS_BUFFER).mul_(10)
    restored.load_state_dict(saved)
    assert other.prior_regularizer.center is getattr(other.prior, CENTER_BUFFER)
    assert other.prior_regularizer.radius is getattr(other.prior, RADIUS_BUFFER)
    other.step(real)
    equal(expected, restored.state_dict())


def test_disabled_exact_model_optimizer_and_rng_parity():
    baseline, disabled = trainer(context()), trainer(context(prior_regularizer='coulomb', prior_reg=0))
    assert not hasattr(disabled.prior, CENTER_BUFFER)
    for name in ('G', 'D', 'prior'):
        equal(getattr(baseline, name).state_dict(), getattr(disabled, name).state_dict())
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    baseline.step(real)
    disabled.step(real)
    for name in ('G', 'D', 'prior', 'opt_g', 'opt_d'):
        equal(getattr(baseline, name).state_dict(), getattr(disabled, name).state_dict())
    for name in baseline._STREAMS:
        equal(getattr(baseline, name).get_state(), getattr(disabled, name).get_state())


def test_unsampled_rows_remain_unchanged_under_rownorm():
    t = trainer(context(particles=64, prior_regularizer='coulomb', prior_reg=.1))
    before = t.prior.z.detach().clone()
    sampled = []
    original = t.prior.sample
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        sampled.append(result[1])
        return result
    t.prior.sample = observed
    t.step(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4)
    mask = torch.ones(len(before), dtype=torch.bool)
    mask[torch.unique(sampled[-1])] = False
    assert mask.any() and torch.equal(before[mask], t.prior.z.detach()[mask])
    assert not torch.equal(before[~mask], t.prior.z.detach()[~mask])


@pytest.mark.parametrize('settings', [dict(prior_update='frozen', prior_reg=.1), dict(prior_reg=0)])
def test_disabled_and_frozen_skip_force_work(settings, monkeypatch):
    monkeypatch.setattr(CoulombPriorRegularizer, 'forward', lambda *a: pytest.fail('inactive force'))
    t = trainer(context(prior_regularizer='coulomb', **settings))
    t.step(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4)
    assert not hasattr(t.prior, CENTER_BUFFER)


@pytest.mark.parametrize('name', ['trajectory', 'residual_student', 'cover_leftover', 'ae_gan_hold'])
def test_behavior_hosts_bind_initialized_tables_and_apply_weight_once(name, tmp_path, monkeypatch):
    task = json.loads((ROOT / 'configs/forge/tasks' / (name + '.json')).read_text())
    task['execution']['steps'] = 2
    seen = []
    original = CoulombPriorRegularizer.forward
    def observed(module, rows):
        seen.append((module.weight, module.center.clone(), module.radius.clone()))
        return original(module, rows)
    monkeypatch.setattr(CoulombPriorRegularizer, 'forward', observed)
    leader = json.loads((ROOT / 'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())
    leader['recipe_overrides'].update(prior_regularizer='coulomb', prior_reg=.01)
    result = run_behavior(dict(candidate=leader, protocol=dict(seed=0)), task, tmp_path)
    assert len(seen) == 2 and all(weight == .01 for weight, _, _ in seen)
    assert torch.equal(seen[0][1], seen[1][1]) and torch.equal(seen[0][2], seen[1][2])
    assert result['evidence']['prior_policy']['regularizer'] == 'coulomb'
    assert result['evidence']['prior_policy']['coulomb']['references']


def test_forge_search_preserves_positive_strength_and_rejects_kind_change():
    base = context(prior_regularizer='coulomb', prior_reg=.01).recipe
    validate_same_technique(base, base.replace(prior_reg=.1))
    with pytest.raises(ValueError, match='prior_regularizer'):
        validate_same_technique(base, base.replace(prior_regularizer='vicreg'))
    assert recipe_field_active('prior_coulomb_softening', base)
    assert not recipe_field_active('prior_reg_target_std', base)
    assert not recipe_field_active('prior_coulomb_softening', base.replace(prior_reg=0))


def test_receipt_cannot_misstate_force_settings_or_drop_reference():
    t = trainer(context(prior_regularizer='coulomb', prior_reg=.01))
    task = json.loads((ROOT / 'configs/forge/tasks/vector_two_broad.json').read_text())
    proof = prior_policy_receipt(t.recipe, t.prior)
    assert validate_prior_policy(task, dict(prior_policy=proof), asdict(t.recipe)) is None
    wrong = deepcopy(proof)
    wrong['coulomb']['softening'] = 2
    assert validate_prior_policy(task, dict(prior_policy=wrong), asdict(t.recipe))['status'] == 'INVALID'
    wrong = deepcopy(proof)
    wrong['coulomb']['references'] = []
    assert validate_prior_policy(task, dict(prior_policy=wrong), asdict(t.recipe))['status'] == 'INCOMPLETE'


@pytest.mark.parametrize("corruption", ["negative_radius", "nan_center"])
def test_malformed_reference_checkpoint_is_rejected_without_mutation(corruption):
    t = trainer(context(prior_regularizer="coulomb", prior_reg=.01))
    before = t.state_dict()
    wrong = deepcopy(before)
    if corruption == "negative_radius":
        wrong["models"]["prior"][RADIUS_BUFFER].fill_(-1)
    else:
        wrong["models"]["prior"][CENTER_BUFFER].fill_(float("nan"))
    with pytest.raises(ValueError, match="Coulomb initial reference"):
        t.load_state_dict(wrong)
    equal(before, t.state_dict())


@pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_force_and_gradients_match_cpu():
    rows = torch.tensor([[-.7, .2], [.8, -.3], [.1, .6]], dtype=torch.float64, requires_grad=True)
    cpu = CoulombPriorRegularizer(center=torch.zeros(2, dtype=rows.dtype), radius=rows.new_tensor(1.))
    value = cpu(rows)
    gradient = torch.autograd.grad(value, rows)[0]
    device_rows = rows.detach().cuda().requires_grad_(True)
    gpu = deepcopy(cpu).cuda()
    other = gpu(device_rows)
    other_gradient = torch.autograd.grad(other, device_rows)[0]
    assert torch.allclose(value, other.cpu(), atol=1e-12, rtol=1e-12)
    assert torch.allclose(gradient, other_gradient.cpu(), atol=1e-12, rtol=1e-12)


@pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="word acquisition requires CUDA")
def test_word_host_binds_reference_and_records_complete_state(tmp_path):
    from experiments.forge.word_adapter import run_word
    task = json.loads((ROOT / 'configs/forge/tasks/five_word_joint_smoke.json').read_text())
    leader = json.loads((ROOT / 'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())
    leader['recipe_overrides'].update(prior_regularizer='coulomb', prior_reg=.01)
    result = run_word(dict(candidate=leader, protocol=dict(seed=0)), task, tmp_path, 'cuda:0', execution_limit=2)
    assert result['scope'] == 'integration_demo_only'
    assert result['evidence']['prior_policy']['regularizer'] == 'coulomb'
    assert len(result['evidence']['prior_policy']['coulomb']['references']) == 1

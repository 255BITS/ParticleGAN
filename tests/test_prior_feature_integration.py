"""Combined-build software invariants; no benchmark qualification or scoring."""
from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import Recipe, get_recipe
from experiments.forge.api import FormulationContext
from experiments.forge.techniques import recipe_field_active

SELECTORS = ('vicreg', 'coulomb', 'sigreg', 'moment_tether', 'initial_mmd')
ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def build(preset='bcap', settings=None, *, resolved=None):
    from json import loads
    leader = loads((ROOT / 'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())['recipe_overrides']
    values = {**(leader if preset == 'bcap' else {}), 'num_particles': 12,
              'z_dim': 2, 'batch_size': 4, 'total_steps': 4, **(settings or {})}
    if resolved is not None:
        # Restore through Recipe's saved-configuration contract, independent of
        # the mutable model-family default and omitted restoration defaults.
        preset, values = None, dict(resolved)
    ctx = FormulationContext(recipe_preset=preset, recipe_overrides=values,
        prior=dict(kind='mog', sigma=.025, standardize=False), prior_contract='recipe_owned_v1', seed=0)
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component='generator')
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component='discriminator')
    return ctx, ctx.build_trainer(g, d)


@pytest.fixture(scope='module')
def frozen():
    archive = os.environ.get('PARTICLEGAN_PRIOR_REVIEW_FIXTURES')
    if not archive:
        pytest.skip('Run source-bound probe.py exports first; see integration review README')
    return {name: torch.load(Path(archive) / (name + '.pt'), map_location='cpu', weights_only=False)
            for name in ('develop', 'proximal', 'coulomb', 'sigreg', 'moment_tether', 'initial_mmd')}


@pytest.mark.parametrize('name', ['bcap_zero', 'bcap_vicreg', 'bcap_frozen', 'bcap_none', 'ka2_zero', 'ka2_vicreg'])
def test_frozen_develop_initial_update_checkpoint_and_consumed_stream_parity(frozen, name):
    old = frozen['develop'][name]
    torch.manual_seed(0)
    # Saved configurations retain their original trainer after a default promotion.
    ctx, trainer = build(old['preset'], old['settings'], resolved=old['initial']['recipe'])
    equal(ctx.state_dict(), old['initial'])
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    equal(trainer.step(real), old['first_metrics'])
    equal(ctx.state_dict(), old['checkpoint'])
    equal(trainer.step(real), old['second_metrics'])
    equal(ctx.state_dict(), old['final'])
    equal(torch.get_rng_state(), old['global_rng'])


@pytest.mark.parametrize('name', ['bcap_zero', 'bcap_vicreg', 'bcap_frozen', 'bcap_none', 'ka2_zero', 'ka2_vicreg'])
def test_frozen_develop_checkpoints_resume_in_combined_build(frozen, name):
    old = frozen['develop'][name]
    ctx, trainer = build(old['preset'], old['settings'], resolved=old['initial']['recipe'])
    ctx.load_state_dict(deepcopy(old['checkpoint']))
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    equal(trainer.step(real), old['second_metrics'])
    equal(ctx.state_dict(), old['final'])


@pytest.mark.parametrize('role', ['proximal', 'coulomb', 'sigreg', 'moment_tether', 'initial_mmd'])
def test_original_positive_feature_checkpoint_replays_exact_next_update(frozen, role):
    old = frozen[role][role + '_positive']
    ctx, trainer = build(old['preset'], old['settings'], resolved=old['initial']['recipe'])
    ctx.load_state_dict(deepcopy(old['checkpoint']))
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    equal(trainer.step(real), old['second_metrics'])
    equal(ctx.state_dict(), old['final'])


def test_historical_presets_keep_frozen_develop_values_when_trainer_is_pinned(frozen):
    for name, old in frozen['develop']['_presets'].items():
        recipe = (get_recipe(name) if name == 'bcap' else
                  get_recipe(name, trainer='ka2').replace(name=old['name']))
        equal(recipe.to_dict(), old)


@pytest.fixture(scope='module')
def promoted_develop():
    archive = os.environ.get('PARTICLEGAN_PRIOR_REVIEW_FIXTURES')
    path = Path(archive) / 'promoted-develop.pt' if archive else None
    if path is None or not path.is_file():
        pytest.skip('Export probe.py from the default-promotion develop source first')
    return torch.load(path, map_location='cpu', weights_only=False)


@pytest.mark.parametrize('name', ['ka2_zero', 'ka2_vicreg'])
def test_promoted_default_preserves_develop_updates_resume_and_consumed_rng(promoted_develop, name):
    # probe.py retains historical packet labels; these two packets resolve gan
    # through the promoted BCAP selector in source 3f9846cb, without pinning it.
    old = promoted_develop[name]
    torch.manual_seed(0)
    ctx, trainer = build(old['preset'], old['settings'])
    equal(ctx.state_dict(), old['initial'])
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    equal(trainer.step(real), old['first_metrics'])
    equal(ctx.state_dict(), old['checkpoint'])
    ctx.load_state_dict(deepcopy(old['checkpoint']))
    equal(trainer.step(real), old['second_metrics'])
    equal(ctx.state_dict(), old['final'])
    equal(torch.get_rng_state(), old['global_rng'])


def test_promoted_default_model_presets_equal_develop(promoted_develop):
    for name, old in promoted_develop['_presets'].items():
        equal(get_recipe(name).to_dict(), old)


@pytest.mark.parametrize('selector', SELECTORS + ('none',))
def test_selected_zero_has_identical_models_optimizers_consumed_rng_and_no_reference_packet(selector):
    plain, a = build()
    alternative, b = build(settings={'prior_regularizer': selector, 'prior_reg': 0})
    before = deepcopy(b.prior_regularizer.state_dict())
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    for _ in range(2):
        x, y = a.step(real), b.step(real)
        for key in ('loss_g', 'loss_d'):
            equal(x[key], y[key])
    for key in ('models', 'optimizers', 'streams', 'initial_lrs', 'completed_steps'):
        equal(a.state_dict()[key], b.state_dict()[key])
    equal(plain.streams.state_dict(), alternative.streams.state_dict())
    equal(before, b.prior_regularizer.state_dict())
    assert 'prior_regularizer' not in b.state_dict()


@pytest.mark.parametrize('selector', SELECTORS)
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=[pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')])])
def test_positive_selector_and_proximal_both_activate_and_resume(selector, device):
    ctx, a = build(settings={'prior_regularizer': selector, 'prior_reg': .001, 'prior_proximal_strength': 1})
    # This fixture constructs on CPU; CUDA coverage uses a fresh public context.
    if device != 'cpu':
        values = dict(ctx.recipe.to_dict())
        ctx = FormulationContext(recipe_preset='bcap', recipe_overrides=values, device=device,
            prior=dict(kind='mog', sigma=.025, standardize=False), prior_contract='recipe_owned_v1', seed=0)
        g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component='generator').to(device)
        d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component='discriminator').to(device)
        a = ctx.build_trainer(g, d)
    calls = []
    hook = a.prior_regularizer.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
    real = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2) / 4
    before_rows = a.prior.z.detach().clone()
    owned = []
    original = a.opt_g.set_sampled_rows
    def record(parameter, rows):
        if parameter is a.prior.z:
            owned.append(rows.detach().clone())
        return original(parameter, rows)
    a.opt_g.set_sampled_rows = record
    a.step(real)
    del a.opt_g.set_sampled_rows
    assert calls == [12]
    hook.remove()
    saved = deepcopy(a.state_dict())
    assert saved['optimizers'][0]['prior_proximal']['stats']['steps'] == 1
    assert owned
    mask = torch.zeros(len(a.prior.z), dtype=torch.bool, device=device)
    mask[torch.unique(owned[-1])] = True
    assert saved['optimizers'][0]['prior_proximal']['stats']['rows'] == mask.sum().item()
    assert torch.equal(a.prior.z[~mask], before_rows[~mask])
    assert torch.equal(a.opt_g._prior_initial[a.prior.z], before_rows)
    assert ('prior_regularizer' in saved) == (selector == 'sigreg')
    a.step(real)
    expected = deepcopy(a.state_dict())
    a.load_state_dict(saved)
    a.step(real)
    equal(a.state_dict(), expected)


@pytest.mark.parametrize('selector', SELECTORS)
def test_non_sampled_host_has_no_auxiliary_regularizer_packet(selector):
    recipe = get_recipe('bcap', prior_regularizer=selector, prior_reg=.001)
    assert recipe.prior_regularizer_checkpoint(None) == {}
    recipe.restore_prior_regularizer(None, None)


def test_moment_eps_remains_searchable_and_other_family_knobs_are_inactive():
    moment = get_recipe('bcap', prior_regularizer='moment_tether', prior_reg=.1)
    assert recipe_field_active('prior_reg_eps', moment)
    assert not recipe_field_active('prior_reg_target_std', moment)
    assert not recipe_field_active('prior_sigreg_knots', moment)
    assert not recipe_field_active('prior_coulomb_softening', moment)


@pytest.mark.parametrize('selector', SELECTORS)
def test_frozen_positive_selector_and_spring_do_no_prior_work(selector):
    plain, a = build(settings={'prior_update': 'frozen'})
    alternative, b = build(settings={'prior_update': 'frozen', 'prior_regularizer': selector,
                                    'prior_reg': .001, 'prior_proximal_strength': 1})
    assert not hasattr(b.opt_g, '_prior_initial')
    assert 'prior_regularizer' not in b.state_dict()
    before = deepcopy(b.prior.state_dict())
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    a.step(real)
    b.step(real)
    equal(before, b.prior.state_dict())
    for key in ('models', 'optimizers', 'streams', 'initial_lrs', 'completed_steps'):
        equal(a.state_dict()[key], b.state_dict()[key])
    equal(plain.streams.state_dict(), alternative.streams.state_dict())


@pytest.mark.parametrize('task_id,policy', [('vector_two_broad', 'frozen'), ('two_pole', 'learned')])
def test_coulomb_knob_receipts_are_inactive_for_frozen_or_non_sampled_recipe_owned_prior(task_id, policy):
    from experiments.forge.api import task_formulation_context
    from json import loads
    task = loads((ROOT / 'configs/forge/tasks' / (task_id + '.json')).read_text())
    knobs = {'prior_coulomb_softening': .2, 'prior_coulomb_confinement': .4,
             'prior_coulomb_neighbors': 4}
    candidate = {'recipe_preset': 'bcap', 'recipe_overrides': {
        'prior_update': policy, 'prior_regularizer': 'coulomb', 'prior_reg': .001, **knobs}}
    context = task_formulation_context(candidate, task)
    fields = context.receipt()['field_ownership']['recipe_fields']
    for name, value in knobs.items():
        assert fields[name]['status'] == 'not_applicable'
        assert fields[name]['value'] is None
        assert fields[name]['reference_recipe_value'] == value
        assert not recipe_field_active(name, context.recipe, task=task)

"""Named component configurations share current hyperparameters, without loops."""
import json
from pathlib import Path

import pytest
from torch import nn

from particlegan import GANLoss, GANTrainer, Recipe, get_recipe


K3P_CONFIG = json.loads((Path(__file__).parents[1] / 'reports/toy100/gap-fill-20260925/sources/k3p/config.json').read_text())


def without_name(values):
    return {key: value for key, value in values.items() if key != 'name'}


def test_default_matches_every_recorded_winning_field():
    # The frozen K3P config ran arm a_r1r2 under the K3P patch; the package
    # has no arm/loss switches: K3P and RpGAN logistic are the formulation.
    actual = json.loads(json.dumps(get_recipe().to_dict()))
    assert not {'reg_arm', 'loss_type', 'gan_mode', 'reg_method'} & set(actual)
    assert (K3P_CONFIG['loss_type'], K3P_CONFIG['gan_mode']) == ('logistic', 'rp')
    # The frozen config fixed the G/D horizon as a 1600-update cap; the
    # default now states it as a fraction of the budget that resolves to the
    # same 1600 updates at the frozen 7000.
    shared = (set(actual) & set(K3P_CONFIG)) - {'name', 'network_lr_horizon_cap'}
    assert {key: actual[key] for key in shared} == {key: K3P_CONFIG[key] for key in shared}
    assert {'network_lr_floor', 'total_steps', 'input_noise_std', 'output_noise_std',
            'output_noise_warmup', 'input_noise_anneal_end', 'batch_size', 'z_dim'} <= shared
    assert K3P_CONFIG['network_lr_horizon_cap'] == Recipe().network_lr_horizon == 1600
    assert actual['name'] == 'k3p'
    assert get_recipe() == Recipe()


def test_recipe_stays_configuration_and_components_without_training_lifecycle():
    recipe = get_recipe()
    for method in ('make_trainer', 'step', 'sample', 'state_dict', 'load_state_dict'):
        assert not hasattr(recipe, method)
    # Persisting configuration does not create networks, optimizers or a trainer.
    assert Recipe(**recipe.to_dict()) == recipe
    assert isinstance(recipe.make_loss(), GANLoss)


@pytest.mark.parametrize('name,model,prior,encoder,conditioning', [
    ('gan', 'gan', 'particles', 'none', 'scalar'),
    ('mog', 'gan', 'mog', 'none', 'scalar'),
    ('ddgan', 'ddgan', 'particles', 'none', 'ucd'),
    ('ddgan_mog', 'ddgan', 'mog', 'none', 'ucd'),
    ('ae_gan', 'gan', 'mog', 'ae', 'scalar'),
    ('vae_gan', 'gan', 'mog', 'hard', 'scalar'),
    ('ae_ddgan', 'ddgan', 'mog', 'ae', 'scalar'),
])
def test_named_components_share_current_training_hyperparameters(name, model, prior, encoder, conditioning):
    recipe = get_recipe(name)
    assert recipe == get_recipe(name=name)
    assert (recipe.model, recipe.prior_kind, recipe.encoder_mode, recipe.conditioning) == (
        model, prior, encoder, conditioning)
    fields = ('lr', 'd_lr_mult', 'prior_lr_mult', 'betas', 'prior_betas',
              'reg_coeff', 'reg_kappa', 'reg_every', 'reg_anchor_weight',
              'direct_particle_gain', 'prior_reg', 'ema_decay',
              'lr_anneal_start', 'lr_floor', 'total_steps')
    for field in fields:
        assert getattr(recipe, field) == getattr(get_recipe(), field)
    if prior == 'mog':
        assert recipe.sigma_rel == .025
    assert Recipe(**recipe.to_dict()) == recipe


@pytest.mark.parametrize('name', ['gan_v1', 'gan_v2', 'gan_v3', 'gan_legacy', 'unknown'])
def test_historical_versions_are_not_selectable(name):
    with pytest.raises(ValueError, match='Unknown recipe'):
        get_recipe(name)


def test_named_recipe_overrides_and_small_objects():
    recipe = get_recipe('ae_gan', num_particles=8, z_dim=3, lr=.002)
    assert recipe.lr == .002 and recipe.z_dim == 3
    assert recipe.make_prior().z.shape == (8, 3)
    assert get_recipe('ddgan', num_classes=2).num_classes == 2
    assert get_recipe('gan') == Recipe()
    with pytest.raises(ValueError, match='particle encoders require'):
        get_recipe('ae_gan', prior_kind='particles', sigma_rel=0)
    with pytest.raises(TypeError):
        get_recipe('gan', not_a_field=True)


@pytest.mark.parametrize('name,overrides', [
    ('gan', dict(conditioning='ucd', num_classes=2)),
    ('ae_gan', {}), ('vae_gan', {}), ('ddgan', {}), ('mog', {}),
])
def test_named_components_do_not_implicitly_choose_training_control_flow(name, overrides):
    recipe = get_recipe(name, **overrides)
    with pytest.raises(ValueError, match='unconditional scalar GANs'):
        GANTrainer(recipe, nn.Linear(recipe.z_dim, 2), nn.Linear(2, 1))


def test_v3_optimizer_roles_resolve_to_recorded_absolute_rates():
    recipe = get_recipe(num_particles=8, z_dim=2)
    generator, discriminator = nn.Linear(2, 2), nn.Linear(2, 1)
    opt_g, opt_d = recipe.make_optimizers(generator, discriminator, recipe.make_prior())
    assert [group['lr'] for group in opt_g.param_groups] == [.00425, .0085]
    assert [group['lr'] for group in opt_d.param_groups] == [.00425]
    assert all(group['betas'] == (0., .999) for opt in (opt_g, opt_d) for group in opt.param_groups)


def test_json_integer_zero_moments_construct_real_adam_optimizers():
    recipe = get_recipe(num_particles=8, z_dim=2, betas=[0, .999], prior_betas=[0, 0])
    optimizers = recipe.make_optimizers(nn.Linear(2, 2), nn.Linear(2, 1), recipe.make_prior())
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            assert all(type(value) is float for value in group['betas'])
    assert optimizers[0].param_groups[1]['betas'] == (0., 0.)

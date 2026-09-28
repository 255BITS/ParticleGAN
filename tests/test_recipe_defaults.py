"""Named component configurations share current hyperparameters, without loops."""
import pytest
from torch import nn

from particlegan import GANLoss, GANTrainer, Recipe, get_recipe, learning_rate_scales


# The promoted formulation: PR #155's dv12-ams-rc3 (DV12 controller, KA2 with
# reg_coeff 3, AMSGrad, constant output noise .029, no input noise). Its
# schedule fields are inert under the controller and are not live here.
PROMOTED = dict(lr=.00425, d_lr_mult=1.0, prior_lr_mult=2.0, betas=(0.0, .999), amsgrad=True,
                reg_coeff=3.0, reg_kappa=1.0, reg_every=1, reg_anchor_min_decay=.9, reg_anchor_weight=1.0,
                d_guard_ratio=5.0, d_guard_min_steps=200, latent_damping_max_rate=.5,
                input_noise_std=0.0, output_noise_std=.029, output_noise_warmup=0.0, lr_floor=1.0,
                network_lr_floor=None, network_lr_horizon_cap=None, ema_decay=.995, prior_reg=0.0,
                batch_size=2048, z_dim=2, num_particles=20_000, total_steps=None)


def without_name(values):
    return {key: value for key, value in values.items() if key != 'name'}


def test_default_is_the_promoted_dv12_ams_rc3_formulation():
    recipe = get_recipe()
    actual = recipe.to_dict()
    assert not {'reg_arm', 'loss_type', 'gan_mode', 'reg_method', 'direct_particle_gain',
                'direct_particle_betas', 'reg_anchor_decay', 'continuous_policy', 'initialization'} & set(actual)
    assert {key: getattr(recipe, key) for key in PROMOTED} == PROMOTED
    assert all(learning_rate_scales(step, recipe) == (1.0, 1.0) for step in range(0, 7001, 500))
    assert actual['name'] == 'dv12'
    assert recipe == Recipe()


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
              'reg_coeff', 'reg_kappa', 'reg_every', 'reg_anchor_weight', 'reg_anchor_min_decay',
              'amsgrad', 'output_noise_std', 'prior_reg', 'ema_decay',
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


def test_optimizer_roles_resolve_to_the_promoted_peak_rates():
    recipe = get_recipe(num_particles=8, z_dim=2)
    generator, discriminator = nn.Linear(2, 2), nn.Linear(2, 1)
    opt_g, opt_d = recipe.make_optimizers(generator, discriminator, recipe.make_prior())
    assert [group['lr'] for group in opt_g.param_groups] == [.00425, .0085]
    assert [group['lr'] for group in opt_d.param_groups] == [.00425]
    assert all(group['betas'] == (0., .999) for opt in (opt_g, opt_d) for group in opt.param_groups)
    assert all(group['amsgrad'] for opt in (opt_g, opt_d) for group in opt.param_groups)
    assert opt_g.controller is opt_d.controller


def test_json_integer_zero_moments_construct_real_adam_optimizers():
    recipe = get_recipe(num_particles=8, z_dim=2, betas=[0, .999], prior_betas=[0, 0])
    optimizers = recipe.make_optimizers(nn.Linear(2, 2), nn.Linear(2, 1), recipe.make_prior())
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            assert all(type(value) is float for value in group['betas'])
    assert optimizers[0].param_groups[1]['betas'] == (0., 0.)

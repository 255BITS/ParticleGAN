"""Pin public GAN versions to the recipes recorded by the toy-suite runs."""
import json
from pathlib import Path

import pytest
from torch import nn

from particlegan import Recipe, get_recipe


ARCHIVED = json.loads((Path(__file__).with_name('fixtures') / 'gan_recipe_versions.json').read_text())


def without_name(values):
    return {key: value for key, value in values.items() if key != 'name'}


@pytest.mark.parametrize('version', ['gan_v1', 'gan_v2', 'gan_v3'])
def test_version_matches_every_recorded_recipe_field(version):
    # JSON conversion matches the tuple representation used in run receipts.
    actual = json.loads(json.dumps(get_recipe(version).to_dict()))
    assert without_name(actual) == without_name(ARCHIVED['recipes'][version])
    assert set(actual) == set(ARCHIVED['recipes'][version])


def test_public_aliases_keep_version_names_and_original_particle_budget():
    assert Recipe() == get_recipe() == get_recipe('gan') == get_recipe('gan_v3')
    assert get_recipe().name == 'gan_v3'
    assert (get_recipe().num_particles, get_recipe().total_steps) == (20_000, 7_000)
    for alias, version in (('gan_legacy', 'gan_v1'), ('100gaussians', 'gan_v2')):
        named = get_recipe(alias)
        canonical = get_recipe(version)
        assert named.name == alias and canonical.name == version
        assert without_name(named.to_dict()) == without_name(canonical.to_dict())


def test_v3_optimizer_roles_resolve_to_recorded_absolute_rates():
    recipe = get_recipe('gan_v3', num_particles=8, z_dim=2)
    generator, discriminator = nn.Linear(2, 2), nn.Linear(2, 1)
    opt_g, opt_d = recipe.make_optimizers(generator, discriminator, recipe.make_prior())
    assert [group['lr'] for group in opt_g.param_groups] == [.00425, .0085]
    assert [group['lr'] for group in opt_d.param_groups] == [.00425]
    assert all(group['betas'] == (0., .99) for opt in (opt_g, opt_d) for group in opt.param_groups)


@pytest.mark.parametrize('version', ['gan_v1', 'gan_v2', 'gan_v3'])
def test_all_versions_keep_the_original_checkpoint_schema(version):
    recipe = get_recipe(version, num_particles=8, z_dim=2, batch_size=4, total_steps=2)
    trainer = recipe.make_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    checkpoint = trainer.state_dict()
    assert checkpoint['schema'] == 1 and checkpoint['recipe'] == recipe.to_dict()
    assert Recipe(**checkpoint['recipe']) == recipe
    restored = recipe.make_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    restored.load_state_dict(checkpoint)
    assert restored.state_dict()['recipe'] == checkpoint['recipe']

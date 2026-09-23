"""One public default; old complete checkpoint receipts remain readable."""
import json
from pathlib import Path

import pytest
from torch import nn

from particlegan import Recipe, get_recipe


ARCHIVED = json.loads((Path(__file__).with_name('fixtures') / 'gan_recipe_versions.json').read_text())


def without_name(values):
    return {key: value for key, value in values.items() if key != 'name'}


def test_default_matches_every_recorded_winning_field():
    actual = json.loads(json.dumps(get_recipe().to_dict()))
    assert without_name(actual) == without_name(ARCHIVED['recipes']['gan_v3'])
    assert get_recipe() == Recipe()


@pytest.mark.parametrize('name', ['gan', 'gan_v1', 'gan_v2', 'gan_v3', 'gan_legacy', 'mog', 'ddgan'])
def test_no_positional_preset_selection(name):
    with pytest.raises(TypeError):
        get_recipe(name)
    # Names in saved receipts identify runs, never choose a formulation.
    labeled = get_recipe(name=name)
    assert without_name(labeled.to_dict()) == without_name(get_recipe().to_dict())


def test_v3_optimizer_roles_resolve_to_recorded_absolute_rates():
    recipe = get_recipe(num_particles=8, z_dim=2)
    generator, discriminator = nn.Linear(2, 2), nn.Linear(2, 1)
    opt_g, opt_d = recipe.make_optimizers(generator, discriminator, recipe.make_prior())
    assert [group['lr'] for group in opt_g.param_groups] == [.00425, .0085]
    assert [group['lr'] for group in opt_d.param_groups] == [.00425]
    assert all(group['betas'] == (0., .99) for opt in (opt_g, opt_d) for group in opt.param_groups)


@pytest.mark.parametrize('version', ['gan_v1', 'gan_v2', 'gan_v3'])
def test_complete_old_receipts_still_restore_without_preset_dispatch(version):
    recipe = Recipe(**ARCHIVED['recipes'][version]).replace(num_particles=8, z_dim=2, batch_size=4, total_steps=2)
    trainer = recipe.make_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    checkpoint = trainer.state_dict()
    assert checkpoint['schema'] == 1 and checkpoint['recipe'] == recipe.to_dict()
    assert Recipe(**checkpoint['recipe']) == recipe
    restored = recipe.make_trainer(nn.Linear(2, 2), nn.Linear(2, 1))
    restored.load_state_dict(checkpoint)
    assert restored.state_dict()['recipe'] == checkpoint['recipe']

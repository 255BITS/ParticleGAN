"""The installed E22 preset is independent of research configuration files."""
import json
from pathlib import Path

import pytest

from particlegan import Recipe, get_recipe, learning_rate_scales


def test_e22_preset_matches_declared_policy_without_task_placeholders():
    fields = json.loads((Path(__file__).parents[1] / "configs/100gaussians/e22-noout.json").read_text())
    fields.update(num_particles=20_000, z_dim=2, batch_size=2048)
    expected = Recipe(**fields).replace(name="e22")
    assert get_recipe("e22") == expected
    assert get_recipe() == Recipe()


def test_e22_task_overrides_and_resolved_recipe_round_trip():
    recipe = get_recipe("e22", num_particles=256, z_dim=8, batch_size=32,
                        output_noise_std=.1, lr=.001)
    assert (recipe.num_particles, recipe.z_dim, recipe.batch_size) == (256, 8, 32)
    assert recipe.output_noise_std == .1 and recipe.lr == .001
    assert recipe.total_steps is None and recipe.continuous_policy == "dv12"
    assert recipe.row_evidence_gate and recipe.particle_birth_death
    assert Recipe(**recipe.to_dict()) == recipe
    assert recipe.replace(serve_average=2).serve_average == 2
    with pytest.raises(TypeError):
        get_recipe("e22", unknown_control=True)


@pytest.mark.parametrize("conditioning", ["conditional", "ucd"])
@pytest.mark.parametrize("controls", [dict(particle_birth_death=True),
                                    dict(row_evidence_gate=True)])
def test_conditional_rows_are_rejected_for_row_mechanisms(conditioning, controls):
    with pytest.raises(ValueError, match="independently sampled unconditional rows"):
        get_recipe(conditioning=conditioning, num_classes=2,
                   continuous_policy="dv12", total_steps=None,
                   lr_control="stationarity", **controls)


def test_conditional_loops_may_select_controls_without_row_mechanisms():
    recipe = get_recipe("e22", conditioning="conditional", num_classes=2,
                        particle_birth_death=False, row_evidence_gate=False,
                        birth_death_isolation=False, birth_death_feature_scale="none")
    assert recipe.conditioning == "conditional" and recipe.lr_control == "stationarity"
    with pytest.raises(ValueError, match="E22Policy/UpdatePolicy"):
        learning_rate_scales(0, recipe)

"""continuous_probe grades ModeHold on the shared runner; constant mode is recipe fields only."""

import pytest

from benchmarks.toy100 import continuous_probe as probe
from particlegan.recipes import learning_rate_scales


def test_constant_mode_is_a_flat_recipe_schedule():
    recipe = probe.probe_recipe("constant")
    assert recipe.total_steps == probe.FROZEN_STEPS
    for step in (0, 600, 1199, 1200, 3600):
        assert learning_rate_scales(step, recipe) == (1.0, 1.0)
    scheduled = probe.probe_recipe("scheduled")
    assert learning_rate_scales(1200, scheduled)[0] < 1.0


def test_window_reports_passing_suffix():
    points = [dict(step=s, modes=8 if s > 100 else 6, hq=0.95) for s in range(50, 351, 50)]
    window = probe._window(points)
    assert window["failing_steps"] == [50, 100]
    assert window["stable_from_step"] == 150 and window["pass_suffix"]


@pytest.mark.parametrize("kwargs", [dict(steps=600), dict(diagnostic_every=7),
                                    dict(steps=2400, shift_step=1000), dict(mode="other")])
def test_rejects_invalid_protocols(kwargs):
    with pytest.raises(ValueError):
        probe.run_probe(**kwargs)

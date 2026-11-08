"""The accelerated backend is explicit and cannot silently change old recipes."""
import pytest

from particlegan import get_recipe


def test_accelerated_backend_keeps_native_preset_and_other_recipe_fields():
    original = get_recipe("bcap")
    accelerated = get_recipe("bcap", optimizer_svd_backend="newton_schulz")
    assert original.optimizer_svd_backend == "native"
    expected = original.to_dict()
    expected["optimizer_svd_backend"] = "newton_schulz"
    assert accelerated.to_dict() == expected


@pytest.mark.parametrize("overrides", [
    {"optimizer_smoothing": 0.},
    {"optimizer_convolution": "none"},
    {"optimizer_family": "adam", "constraint_geometry_mode": "none",
     "optimizer_smoothing": 0., "optimizer_convolution": "none"},
])
def test_accelerated_backend_rejects_unsupported_recipe_laws(overrides):
    with pytest.raises(ValueError, match="newton_schulz|dualnorm"):
        get_recipe("bcap", optimizer_svd_backend="newton_schulz", **overrides)

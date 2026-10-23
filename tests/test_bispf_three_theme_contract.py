"""Admission/serialization controls for the explicitly opted-in research laws."""
import pytest

from experiments.forge.api import CapabilityError, FormulationContext
from particlegan import get_recipe


@pytest.mark.parametrize("delta", [
    {"output_energy_weight": 1.0},
    {"fidelity_transport_weight": 1.0},
    {"bispf_thermal_application": "displacement"},
])
def test_component_hosts_refuse_mechanisms_they_do_not_consume(delta):
    recipe = get_recipe("bispf_optimized", prior_kind="mog", sigma_rel=0.0,
                        standardize=False).replace(**delta)
    with pytest.raises(CapabilityError, match="does not consume"):
        FormulationContext(recipe_overrides=recipe.to_dict(), execution_path="public_components",
                           prior={"kind": "mog", "sigma": .025, "standardize": False})


def test_inactive_additions_preserve_older_serialized_recipe_packets():
    recipe = get_recipe("bispf_optimized")
    for key in ("output_energy_weight", "fidelity_transport_weight", "bispf_thermal_application"):
        assert key not in recipe.to_dict()
    assert get_recipe("bispf_optimized", output_energy_weight=1).to_dict()["output_energy_weight"] == 1
    assert get_recipe("bispf_optimized", fidelity_transport_weight=1).to_dict()["fidelity_transport_weight"] == 1
    assert get_recipe("bispf_optimized", bispf_thermal_application="displacement").to_dict()["bispf_thermal_application"] == "displacement"

"""One public extension binding reaches both supported host execution paths."""
import pytest

from experiments.forge import api


def test_single_default_binding_reaches_both_paths_and_has_ownership_receipt(monkeypatch):
    definition = api.ExtensionSpec("anchor_gain", "float", "recipe", "reg_anchor_weight",
        "Expose the public anchor gain", producer="idea.extensions.anchor_gain",
        gradient_ownership="critic penalty, no new parameter", optimizer_binding="critic optimizer",
        initialization="scalar declaration, no draws", checkpoint="Recipe.reg_anchor_weight",
        shape="scalar")
    monkeypatch.setattr(api, "default_registry", lambda: api.CapabilityRegistry().register_extension(definition))
    for execution_path in ("public_trainer", "public_components"):
        context = api.FormulationContext(execution_path=execution_path, extensions={"anchor_gain": .6})
        assert context.recipe.reg_anchor_weight == .6
        receipt = context.receipt()
        assert receipt["api_changes"][0]["public_definition"] == "particlegan.Recipe.reg_anchor_weight"
        assert receipt["api_changes"][0]["optimizer_binding"] == "critic optimizer"
        assert receipt["api_changes"][0]["shape"] == "scalar"


def test_trainer_only_variable_is_not_silently_dropped_by_component_host(monkeypatch):
    definition = api.ExtensionSpec("extra_horizon", "int", "trainer", "max_steps", "public continuation bound")
    monkeypatch.setattr(api, "default_registry", lambda: api.CapabilityRegistry().register_extension(definition))
    with pytest.raises(api.CapabilityError, match="unsupported by public_components"):
        api.FormulationContext(execution_path="public_components", extensions={"extra_horizon": 99})


def test_nonpublic_variable_cannot_be_registered():
    with pytest.raises(ValueError, match="supported"):
        api.CapabilityRegistry().register_extension(api.ExtensionSpec(
            "secret_scale", "float", "recipe", "_sigma_intrinsic_scale", "unsupported variable"))

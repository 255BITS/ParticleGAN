"""Pure native model constructors shared by explicit host profiles.

No initializer, device move, sampler or recipe lives here. The legacy benchmark
keeps its original construction order and RNG route; these primitives use the
same public model classes without claiming identical initialization fixtures.
"""
from torch import nn
from lib.toy_models import SimpleMLPDiscriminator


def native_component(spec, *, role):
    """Construct one exact declared component, rejecting ignored model fields."""
    if role == "generator":
        if set(spec) != {"kind", "in_features", "out_features", "bias"} or spec["kind"] != "linear":
            raise ValueError("native generator requires an explicit linear card")
        if any(type(spec[k]) is not int or spec[k] < 1 for k in ("in_features", "out_features")) or type(spec["bias"]) is not bool:
            raise ValueError("invalid native linear dimensions or bias declaration")
        return nn.Linear(spec["in_features"], spec["out_features"], bias=spec["bias"])
    if role == "discriminator":
        keys = {"kind", "in_dim", "hidden_dim", "n_hidden", "fourier"}
        if set(spec) != keys or spec["kind"] != "simple_mlp":
            raise ValueError("native discriminator requires an explicit simple_mlp card")
        if any(type(spec[k]) is not int or spec[k] < (0 if k == "fourier" else 1) for k in keys - {"kind"}):
            raise ValueError("invalid native discriminator dimensions")
        return SimpleMLPDiscriminator(**{key: spec[key] for key in keys - {"kind"}})
    raise ValueError("unknown native component role")

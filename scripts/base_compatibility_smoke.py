"""Bounded installed-base API and exact continuation check; no quality claim.

CI copies this script outside the checkout and runs it using minimum base
dependency versions. Only Torch and NumPy are required by the installed wheel.
"""
from io import BytesIO
import importlib.metadata
import json

import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe, init
from particlegan_forge import NamedStreams


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert (left.dtype, left.shape) == (right.dtype, right.shape)
        # Uninitialized optimizer diagnostics can contain NaN sentinels.
        assert torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def make_trainer(recipe):
    generator = nn.Linear(2, 2)
    critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1))
    prior = recipe.make_prior()
    for seed, module in enumerate((generator, critic, prior)):
        init.deterministic_orthogonal_(module, seed=seed)
    return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


def main():
    torch.set_num_threads(1)
    recipe = get_recipe(num_particles=16, z_dim=2, batch_size=8, total_steps=4)
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    source = make_trainer(recipe)
    data = torch.Generator().manual_seed(0)
    source.step(torch.randn(8, 2, generator=data))

    serialized = BytesIO()
    torch.save({"trainer": source.state_dict(), "data": data.get_state()}, serialized)
    serialized.seek(0)
    saved = torch.load(serialized, weights_only=True)
    real = torch.randn(8, 2, generator=data)
    source.step(real)
    expected = source.state_dict()

    resumed = make_trainer(Recipe.from_dict(saved["trainer"]["recipe"]))
    resumed.load_state_dict(saved["trainer"])
    restored_data = torch.Generator().set_state(saved["data"])
    restored_real = torch.randn(8, 2, generator=restored_data)
    assert_same(real, restored_real)
    resumed.step(restored_real)
    assert_same(expected, resumed.state_dict())
    samples = resumed.sample(4, generator=torch.Generator().manual_seed(0))
    assert samples.shape == (4, 2) and torch.isfinite(samples).all()
    streams = NamedStreams(seed=0)
    stream_state = streams.state_dict()
    streams.load_state_dict(stream_state)
    print(json.dumps({"scope": "base_api_software_compatibility", "exact_continuation": True,
                      "completed_updates": 2,
                      "versions": {name: importlib.metadata.version(name)
                                   for name in ("torch", "numpy", "particlegan")}}))


if __name__ == "__main__":
    main()

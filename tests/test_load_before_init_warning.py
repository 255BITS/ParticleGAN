"""make_optimizers warns before overwriting weights changed after construction."""
import copy
import io
import warnings

import pytest
import torch
from torch import nn

from particlegan import BatchDistanceDiscriminator, GANTrainer, get_recipe, initialize_

MESSAGE = "make_optimizers is overwriting"


def networks():
    g = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
    return g, BatchDistanceDiscriminator(hidden_dim=8, n_hidden=1)


def trained_state(module):
    # Round-trip through torch.save/torch.load like a real checkpoint.
    buffer = io.BytesIO()
    torch.save({k: v + .25 for k, v in module.state_dict().items()}, buffer)
    buffer.seek(0)
    return torch.load(buffer)


def overwrite_warnings(caught):
    return [str(w.message) for w in caught if MESSAGE in str(w.message)]


def make(recipe, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = recipe.make_optimizers(*args, **kwargs)
    return result, overwrite_warnings(caught)


def test_load_then_make_optimizers_warns_and_names_fixes():
    g, d = networks()
    g.load_state_dict(trained_state(g))
    _, messages = make(get_recipe(num_particles=8), g, d)
    assert len(messages) == 1
    assert "generator (G)" in messages[0] and "'0.weight'" in messages[0]
    assert "call make_optimizers before loading" in messages[0]
    assert "initialization=None" in messages[0]


def test_make_optimizers_then_load_keeps_loaded_weights_without_warning():
    g, d = networks()
    state_g, state_d = trained_state(g), trained_state(d)
    recipe = get_recipe(num_particles=8)
    _, messages = make(recipe, g, d)
    assert messages == []
    g.load_state_dict(state_g)
    d.load_state_dict(state_d)
    _, messages = make(recipe, g, d)  # e.g. re-created optimizers on resume
    assert messages == []
    for module, state in ((g, state_g), (d, state_d)):
        for name, value in module.state_dict().items():
            assert torch.equal(value, state[name]), name


@pytest.mark.parametrize("build", [
    networks,
    lambda: (nn.Sequential(nn.Embedding(9, 4, padding_idx=0), nn.Linear(4, 4)),
             nn.Sequential(nn.Conv2d(1, 4, 3), nn.ConvTranspose2d(4, 1, 3))),
    lambda: (nn.TransformerEncoderLayer(8, 2, dim_feedforward=16),
             nn.MultiheadAttention(8, 2, kdim=4, vdim=4, add_bias_kv=True)),
])
def test_fresh_modules_do_not_warn(build):
    g, d = build()
    _, messages = make(get_recipe(num_particles=8), g, d, ema_critic=copy.deepcopy(d))
    assert messages == []


def test_fresh_trainer_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GANTrainer(get_recipe(num_particles=8, z_dim=2, batch_size=8), *networks())
    assert overwrite_warnings(caught) == []


def test_initialization_none_does_not_warn_and_keeps_weights():
    g, d = networks()
    g.load_state_dict(trained_state(g))
    saved = copy.deepcopy(g.state_dict())
    _, messages = make(get_recipe(num_particles=8, initialization=None), g, d)
    assert messages == []
    assert all(torch.equal(v, saved[k]) for k, v in g.state_dict().items())


def test_device_and_dtype_moves_do_not_warn():
    g, d = networks()
    g.to("cpu").double().float()
    d.to(torch.float64)
    _, messages = make(get_recipe(num_particles=8), g, d)
    assert messages == []


@pytest.mark.parametrize("modify", ["copy_", "optimizer_step"])
def test_in_place_changes_warn(modify):
    g, d = networks()
    if modify == "copy_":
        with torch.no_grad():
            d.layers[0].weight.copy_(torch.randn_like(d.layers[0].weight))
    else:
        optimizer = torch.optim.Adam(d.parameters())
        d(torch.randn(8, 2)).sum().backward()
        optimizer.step()
    _, messages = make(get_recipe(num_particles=8), g, d)
    assert len(messages) == 1 and "discriminator (D)" in messages[0]


def test_loaded_critic_warns_about_ema_sync():
    g, d = networks()
    ema = copy.deepcopy(d)
    state = trained_state(d)
    d.load_state_dict(state)
    ema.load_state_dict(state)
    _, messages = make(get_recipe(num_particles=8), g, d, ema_critic=ema)
    assert len(messages) == 1 and "EMA critic" in messages[0]
    # Behavior is unchanged: D is initialized and the EMA follows it.
    for a, b in zip(d.parameters(), ema.parameters()):
        assert torch.equal(a, b)
    assert not torch.equal(d.layers[0].weight, state["layers.0.weight"])


class XavierGenerator(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
        for layer in (self[0], self[2]):
            nn.init.xavier_uniform_(layer.weight)


def test_custom_constructor_init_warns_and_explicit_initialize_is_silent():
    _, messages = make(get_recipe(num_particles=8), XavierGenerator(), networks()[1])
    assert len(messages) == 1 and "custom init in the module's constructor" in messages[0]
    assert "particlegan.initialize_(module, key=0)" in messages[0]
    g, d = XavierGenerator(), networks()[1]
    expected = copy.deepcopy(g)
    make(get_recipe(num_particles=8), expected, copy.deepcopy(d))
    initialize_(g, key=0)
    _, messages = make(get_recipe(num_particles=8), g, d)
    assert messages == []
    for a, b in zip(g.parameters(), expected.parameters()):
        assert torch.equal(a, b)  # explicit init gives the weights the recipe would

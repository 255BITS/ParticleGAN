"""Public initialization contracts and recipe/EMA integration."""
import copy
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import BatchDistanceDiscriminator, GANTrainer, Recipe, get_recipe, initialize_


def test_default_recipe_repeats_weights_and_prior_without_resetting_rng():
    recipe = get_recipe(num_particles=16, z_dim=2, batch_size=8, total_steps=4)
    def build():
        return GANTrainer(recipe, nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)),
                          BatchDistanceDiscriminator(hidden_dim=8, n_hidden=1))
    torch.manual_seed(0)
    first, second = build(), build()
    for role in ("G", "D", "prior", "ema_G", "ema_D", "ema_prior"):
        for a, b in zip(getattr(first, role).parameters(), getattr(second, role).parameters()):
            assert torch.equal(a, b), role
    for a, b in zip(first.D.parameters(), first.ema_D.parameters()):
        assert torch.equal(a, b)
    assert torch.count_nonzero(first.D.head.weight[:, -4:]) == 0
    x = torch.randn(8, 2)
    assert all(torch.isfinite(torch.as_tensor(value)) for value in first.step(x).values())


def test_explicit_init_is_rng_neutral_and_recipe_does_not_reset_learned_weights():
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), BatchDistanceDiscriminator(hidden_dim=8, n_hidden=1)
    before = torch.get_rng_state().clone()
    assert initialize_(g) is g
    initialize_(d, key=1)
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        g.weight.add_(.125)
        d.head.weight[:, -4:] = .25
    saved = [p.clone() for m in (g, d) for p in m.parameters()]
    recipe = get_recipe(num_particles=8)
    for _ in range(2):
        recipe.make_optimizers(g, d, ema_critic=copy.deepcopy(d))
    for a, b in zip(saved, [p for m in (g, d) for p in m.parameters()]):
        assert torch.equal(a, b)


def test_preserve_supplied_weights_prior_and_ema():
    recipe = get_recipe(initialization=None, num_particles=8)
    g, d = nn.Linear(2, 2), nn.Linear(2, 1)
    prior, ema = recipe.make_prior(), copy.deepcopy(d)
    with torch.no_grad():
        ema.weight.add_(.5)
    modules = (g, d, prior, ema)
    before = [[p.clone() for p in m.parameters()] for m in modules]
    recipe.make_optimizers(g, d, prior, ema_critic=ema)
    for saved, module in zip(before, modules):
        assert all(torch.equal(a, b) for a, b in zip(saved, module.parameters()))
    # Even with initialization enabled, an explicitly supplied prior is kept.
    saved_prior = prior.z.clone()
    GANTrainer(recipe.replace(initialization="batch_feature_zero"), g, d, prior=prior)
    assert torch.equal(prior.z, saved_prior)


def test_prior_only_factory_and_archived_recipe_keep_existing_contracts():
    from benchmarks.legacy.recipe import LegacyRecipe, get_recipe as historical_recipe
    recipe = get_recipe(num_particles=8)
    prior = recipe.make_prior()
    opt_g, _ = recipe.make_optimizers(None, nn.Linear(2, 1), prior)
    assert opt_g.param_groups[0]["params"][0] is prior.z
    assert LegacyRecipe().initialization is None
    assert historical_recipe().initialization is None
    assert "initialization" not in historical_recipe().to_dict()
    assert historical_recipe(initialization="batch_feature_zero").to_dict()["initialization"] == "batch_feature_zero"


@pytest.mark.parametrize("layer", [
    lambda: nn.Linear(1, 1), lambda: nn.Linear(7, 11),
    lambda: nn.Conv2d(4, 6, 3, groups=2), lambda: nn.ConvTranspose2d(4, 6, 3),
])
def test_matrix_gram_and_declared_scale(layer):
    module = initialize_(layer().double())
    matrix = module.weight.flatten(1)
    rows, cols = matrix.shape
    gain2 = max(rows, cols) / (3 * cols)
    gram = matrix.T @ matrix if rows >= cols else matrix @ matrix.T
    torch.testing.assert_close(gram, torch.eye(min(rows, cols), dtype=torch.float64) * gain2,
                               atol=1e-14, rtol=1e-13)


def test_host_reinitialized_weights_are_kept_default_draws_replaced():
    # examples/100gaussians.py and benchmarks/toy100 xavier-initialize their MLPs;
    # replacing those at the PyTorch default RMS shrank the toy100 critic 4x.
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(14, 128), nn.LeakyReLU(), nn.Linear(128, 128), nn.Linear(128, 1))
    nn.init.xavier_uniform_(net[0].weight)
    nn.init.kaiming_normal_(net[3].weight)
    with torch.no_grad():
        net[2].weight.mul_(.1)
    custom = {i: net[i].weight.clone() for i in (0, 2, 3)}
    default = [net[i].bias.clone() for i in (0, 2, 3)]
    initialize_(net)
    for i, saved in custom.items():
        assert torch.equal(net[i].weight, saved), i
    assert all(not torch.equal(a, b) for a, b in zip(default, (net[i].bias for i in (0, 2, 3))))
    # Default draws of any size stay in the replaced set (tiny tensors included).
    for shape in ((1, 1), (2, 1), (1, 128), (128, 14), (512, 512)):
        layer = nn.Linear(shape[1], shape[0])
        before = layer.weight.clone()
        initialize_(layer)
        assert not torch.equal(before, layer.weight), shape


def test_transformer_layers_and_lora_preserve_special_parameters():
    first = initialize_(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16).double())
    second = initialize_(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16).double())
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert torch.equal(first.norm1.weight, torch.ones_like(first.norm1.weight))
    assert torch.count_nonzero(first.self_attn.in_proj_bias) == 0
    embedded = initialize_(nn.Embedding(9, 4, padding_idx=0))
    assert torch.count_nonzero(embedded.weight[0]) == 0
    adapter = nn.ModuleDict({"base": nn.Linear(8, 6).requires_grad_(False),
                             "A": nn.Linear(8, 2, bias=False),
                             "B": nn.Linear(2, 6, bias=False)})
    nn.init.zeros_(adapter["B"].weight)
    saved = adapter["base"].weight.clone()
    initialize_(adapter)
    assert torch.equal(saved, adapter["base"].weight)
    assert torch.count_nonzero(adapter["B"].weight) == 0
    assert torch.count_nonzero(adapter["A"].weight) > 0


def test_explicit_api_matches_measured_hook_on_standard_networks():
    # Isolate the research hook; ordinary API calls never install it.
    script = r'''
import copy
import torch
from torch import nn
from particlegan import BatchDistanceDiscriminator, initialize_
from particlegan.batch_feature_init import install, uninstall
torch.set_num_threads(1)
torch.manual_seed(0)
install()
g = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
d = BatchDistanceDiscriminator(hidden_dim=8, n_hidden=1)
a, b = copy.deepcopy(g), copy.deepcopy(d)
torch.optim.Adam(g.parameters())
torch.optim.Adam(d.parameters())
uninstall()
initialize_(a, key=0)
initialize_(b, key=1)
for x, y in zip(list(g.parameters()) + list(d.parameters()), list(a.parameters()) + list(b.parameters())):
    assert torch.equal(x, y)
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_invalid_configuration_and_lazy_modules():
    assert Recipe("k3p", "gan", 3).z_dim == 3
    with pytest.raises(ValueError, match="initialization"):
        get_recipe(initialization="unknown")
    with pytest.raises(ValueError, match="key"):
        initialize_(nn.Linear(2, 2), key=-1)
    with pytest.raises(ValueError, match="materialize"):
        initialize_(nn.LazyLinear(2))


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_restores_original_initialization_and_continues(legacy):
    recipe = get_recipe(initialization=None, num_particles=8, batch_size=4, total_steps=4)
    def build(r):
        return GANTrainer(r, nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)))
    original = build(recipe)
    x = torch.arange(8).reshape(4, 2).float() / 8
    original.step(x)
    checkpoint = copy.deepcopy(original.state_dict())
    if legacy:
        del checkpoint["recipe"]["initialization"]
    expected = original.step(x)
    restored = build(recipe.replace(initialization="batch_feature_zero"))
    restored.load_state_dict(checkpoint)
    assert restored.recipe.initialization is None
    actual = restored.step(x)
    for key in expected:
        torch.testing.assert_close(torch.as_tensor(actual[key]), torch.as_tensor(expected[key]), rtol=0, atol=0)

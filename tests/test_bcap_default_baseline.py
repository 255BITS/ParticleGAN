"""Named-preset adoption checks; bounded software updates, not qualification.

The reference subprocess imports actual develop 5737ade, including its old
public preset. Neither expected updates nor checkpoints are rebuilt from the
new optimizer. Research grades remain in their frozen PR383 publication.
"""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])

import pytest
import torch
from torch import nn

from experiments.forge.api import TRAINER_STREAM_BINDINGS, resolve_public_recipe
from experiments.forge.rng import NamedStreams
from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.init import deterministic_orthogonal_


ROOT = Path(__file__).resolve().parents[1]
DEVELOP = "5737ade47dca89b04d338ada20667781d1f2b5df"
OTHER_PRESETS = ("gan", "ka2", "k3p", "bcap_adam", "e22", "e22_routed", "atlas",
                 "mog", "ddgan", "ddgan_mog", "ae_gan", "vae_gan", "ae_ddgan", "halloween")


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _build(recipe, device="cpu"):
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(0)
        recipe = recipe.replace(z_dim=2, num_particles=8, batch_size=4, total_steps=4,
                                prior_kind="mog", sigma_rel=.025, standardize=False)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)).to(device)
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).to(device)
        prior = recipe.make_prior().to(device)
        for component in (generator, critic, prior):
            deterministic_orthogonal_(component, seed=0)
        streams = NamedStreams(0, device=device)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          **{name: streams.generator(family, component=component, purpose=purpose)
                             for name, (family, component, purpose) in TRAINER_STREAM_BINDINGS.items()})


def _batch(device="cpu"):
    return torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]], device=device)


def _reference(output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    packets = {"presets": {name: get_recipe(name).to_dict() for name in (*OTHER_PRESETS, "bcap")}}
    for name in ("bcap", "bcap_adam"):
        trainer = _build(get_recipe(name))
        trainer.step(_batch())
        checkpoint = deepcopy(trainer.state_dict())
        losses = trainer.step(_batch())
        continued = deepcopy(trainer.state_dict())
        sample = trainer.sample(11)
        packets[name] = dict(checkpoint=checkpoint, losses=losses, continued=continued,
                             sample=sample, sampled=deepcopy(trainer.state_dict()))
    torch.save(packets, output)


@pytest.fixture(scope="module")
def actual_develop(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bcap-default-actual-develop")
    source = directory / "source"
    source.mkdir()
    archive = subprocess.run(["git", "archive", DEVELOP, "particlegan", "experiments"],
                             cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(source, filter="data")
    output, log = directory / "reference.pt", directory / "reference.log"
    env = {**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    with log.open("w") as stdout:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(source), str(output)],
                                cwd=source, env=env, stdout=stdout, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, log.read_text()
    return torch.load(output, weights_only=False)


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("name", OTHER_PRESETS)
def test_other_named_presets_retain_actual_develop_fields(actual_develop, name):
    recipe = get_recipe(name)
    assert json.loads(json.dumps(recipe.to_dict())) == json.loads(json.dumps(actual_develop["presets"][name]))
    assert recipe.constraint_geometry_mode == recipe.critic_step_mode == "none"
    assert recipe.optimizer_svd_backend == "native"
    assert recipe.kinetic_transport_weight == recipe.kinetic_transport_local_weight == 0.


def test_named_bcap_changes_only_direction_mode_and_matches_measured_recipe(actual_develop):
    selected = json.loads((ROOT / "reports/forge/bcap-three-phase/research-baseline.json").read_text())
    recipe = get_recipe("bcap")
    for field, value in selected["global_recipe_overrides"].items():
        assert getattr(recipe, field) == value, field
    assert recipe.optimizer_momentum == 0. and recipe.optimizer_svd_backend == "native"
    assert recipe.critic_step_mode == "none"
    assert recipe.kinetic_transport_weight == recipe.kinetic_transport_local_weight == 0.
    expected = {**actual_develop["presets"]["bcap"], "constraint_geometry_mode": "direction_blend"}
    assert json.loads(json.dumps(recipe.to_dict())) == json.loads(json.dumps(expected))
    assert Recipe(**recipe.to_dict()) == recipe
    assert Recipe().constraint_geometry_mode == "none"


def test_explicit_none_restores_incumbent_and_forge_v1_remains_pinned(actual_develop):
    incumbent = get_recipe("bcap", constraint_geometry_mode="none")
    assert json.loads(json.dumps(incumbent.to_dict())) == json.loads(json.dumps(actual_develop["presets"]["bcap"]))
    archived = resolve_public_recipe(dict(recipe_preset="bcap", recipe_overrides={}))
    expected = {**actual_develop["presets"]["bcap_adam"], "name": "bcap"}
    assert json.loads(json.dumps(archived.to_dict())) == json.loads(json.dumps(expected))
    assert archived.constraint_geometry_mode == "none"


@pytest.mark.parametrize("name", ("bcap", "bcap_adam"))
def test_actual_develop_checkpoint_resumes_exactly_from_saved_recipe(actual_develop, name):
    saved = actual_develop[name]
    restored = _build(Recipe(**saved["checkpoint"]["recipe"]))
    assert not hasattr(restored.opt_g, "bind_protected_losses")
    restored.load_state_dict(saved["checkpoint"])
    _equal(saved["losses"], restored.step(_batch()))
    _equal(saved["continued"], restored.state_dict())
    _equal(saved["sample"], restored.sample(11))
    _equal(saved["sampled"], restored.state_dict())
    current = _build(get_recipe("bcap"))
    before = current.state_dict()
    with pytest.raises(ValueError, match="recipe"):
        current.load_state_dict(saved["checkpoint"])
    _equal(before, current.state_dict())


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_current_named_preset_replays_all_checkpointed_streams(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU compatibility remains required")
    from particlegan.optim.direction_blend import DirectionBlendOptimizer
    from particlegan.optim.dualnorm import NormalizedOptimizer
    trainer = _build(get_recipe("bcap"), device)
    assert type(trainer.opt_g) is DirectionBlendOptimizer
    assert type(trainer.opt_d) is NormalizedOptimizer
    trainer.step(_batch(device))
    trainer.sample(11)
    checkpoint = deepcopy(trainer.state_dict())
    expected_losses = trainer.step(_batch(device))
    expected_sample = trainer.sample(11)
    expected_state = deepcopy(trainer.state_dict())
    restored = _build(get_recipe("bcap"), device)
    restored.load_state_dict(checkpoint)
    _equal(expected_losses, restored.step(_batch(device)))
    _equal(expected_sample, restored.sample(11))
    _equal(expected_state, restored.state_dict())
    assert restored.opt_g.constraint_geometry_stats["steps"] == 2
    assert not {"kinetic_transport", "kinetic_transport_local"} & expected_losses.keys()


def test_disabled_public_components_need_no_projection_or_transport_hook(monkeypatch):
    from particlegan.optim.direction_blend import DirectionBlendOptimizer
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled mechanism was consumed")
    monkeypatch.setattr(DirectionBlendOptimizer, "bind_protected_losses", forbidden)
    monkeypatch.setattr(Recipe, "kinetic_transport_loss", forbidden)
    monkeypatch.setattr(Recipe, "kinetic_transport_local_loss", forbidden)
    for preset in ("bcap", "bcap_adam", "ka2", "k3p"):
        recipe = get_recipe(preset, constraint_geometry_mode="none")
        trainer = _build(recipe)
        assert not hasattr(trainer.opt_g, "bind_protected_losses")
        losses = trainer.step(_batch())
        assert not {"kinetic_transport", "kinetic_transport_local"} & losses.keys()
        assert not {"constraint_geometry", "direction_blend"} & trainer.opt_g.state_dict().keys()


def test_current_named_component_fails_closed_without_binding_and_handles_opposed_goals():
    from particlegan.optim.constraint_geometry import constraint_geometry_backward
    model = nn.Linear(2, 1, bias=False)
    optimizer = get_recipe("bcap").make_generator_optimizer(model)
    initial_model, initial_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    (-model.weight[:, 0].sum()).backward()
    with pytest.raises(ValueError, match="protected-loss backward"):
        optimizer.step()
    _equal(initial_model, model.state_dict())
    _equal(initial_optimizer, optimizer.state_dict())
    optimizer.zero_grad()
    goal = model.weight[:, 0].sum()
    constraint_geometry_backward(-goal, optimizer, (goal, -goal))
    optimizer.step()
    _equal(initial_model, model.state_dict())
    assert optimizer.direction_blend_stats["pareto_stalls"] == 1
    assert optimizer.constraint_geometry_stats["steps"] == 1
    assert optimizer.state_dict()["constraint_geometry"]["pending"] is None


if __name__ == "__main__":
    _reference(Path(sys.argv[2]))

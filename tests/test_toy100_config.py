"""A single declared manifest must execute every named toy with visible overrides."""

import json
from types import SimpleNamespace

import pytest

from benchmarks.toy100 import __main__ as cli
from benchmarks.toy100.config import resolve_problem_config, validate_manifest
from benchmarks.toy100.problems import PROBLEM_NAMES


def test_flat_problem_configs_apply_only_declared_override():
    manifest = {"name": "candidate", "steps": 1000, "seed": 1234,
                "problem_overrides": {"staggered100": {"batch_size": 1024}}}
    configs = {name: resolve_problem_config(manifest, name) for name in PROBLEM_NAMES}
    assert [configs[name].get("batch_size") for name in PROBLEM_NAMES] == [None, None, 1024]
    assert all("problem_overrides" not in config for config in configs.values())
    assert [configs[name]["problem"] for name in PROBLEM_NAMES] == list(PROBLEM_NAMES)
    assert all(config["seed"] == 1234 for config in configs.values())


@pytest.mark.parametrize("bad", [
    {"problem_overrides": {"unknown100": {"batch_size": 1024}}},
    {"problem_overrides": {"staggered100": 1024}},
    {"problem_overrides": {"staggered100": {"problem": "grid100"}}},
    {"problem_overrides": {"staggered100": {"seed": 99}}},
    {"problem_overrides": {"staggered100": {"problem_overrides": {}}}},
    {"problem_overrides": {"staggered100": {"unknown_knob": 1}}},
    {"problem_overrides": []},
])
def test_invalid_or_hidden_override_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_manifest({"steps": 1000, **bad})


def test_run_executes_all_three_with_declared_configs(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"name": "candidate", "steps": 1000,
                                       "seed": 1234, "batch_size": 512,
                                       "problem_overrides": {"staggered100": {"batch_size": 1024}}}))
    seen = []

    def fake_train(config, folder):
        seen.append((config, folder))
        return {"status": "complete"}

    monkeypatch.setattr(cli, "train", fake_train)
    monkeypatch.setattr(cli, "evaluate_suite", lambda output, problem=None: {
        "status": "PASS", "passed_problems": 3, "required_problems": 3})
    args = SimpleNamespace(config=config_path, output=tmp_path / "run", problem=None,
                           steps=None, device=None, no_render=True)
    assert cli._run(args) == 0
    assert [config["problem"] for config, _ in seen] == list(PROBLEM_NAMES)
    assert [config["batch_size"] for config, _ in seen] == [512, 512, 1024]
    assert all("problem_overrides" not in config for config, _ in seen)
    archived = json.loads((args.output / "run_manifest.json").read_text())
    assert archived["resolved_problem_configs"]["staggered100"]["batch_size"] == 1024


def test_unselected_invalid_override_blocks_individual_run(tmp_path, monkeypatch):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"steps": 1000,
                                       "problem_overrides": {"staggered100": {"unknown_knob": 1}}}))
    monkeypatch.setattr(cli, "train", lambda *_: pytest.fail("training must not start"))
    args = SimpleNamespace(config=config_path, output=tmp_path / "out", problem="grid100",
                           steps=None, device=None, no_render=True)
    with pytest.raises(ValueError, match="unknown fields"):
        cli._run(args)
    assert not args.output.exists()


def test_default_command_uses_the_package_default_and_strict_accuracy():
    args = cli._parser().parse_args(["run", "--output", "/tmp/toy100-example", "--no-render"])
    assert str(args.config) == "configs/toy100/default.json"
    assert args.require_accuracy is True


def test_shipped_default_config_trains_the_package_default_recipe():
    from pathlib import Path
    from particlegan import Recipe, get_recipe
    from benchmarks.toy100.train import load_config, resolve_config
    root = Path(__file__).resolve().parents[1]
    default = cli._parser().parse_args(["run", "--output", "x"]).config
    assert default == Path("configs/toy100/default.json")
    manifest = load_config(root / default)
    for name in PROBLEM_NAMES:
        flat, recipe = resolve_config(resolve_problem_config(manifest, name))
        assert type(recipe) is Recipe
        assert recipe == get_recipe(total_steps=7000).replace(name="particlegan_default")
        # The benchmark's own noise wrappers stay off; the trainer applies the
        # recipe's constant output noise, recorded under its own key.
        assert flat["input_noise_std"] == flat["output_noise_std"] == 0.0
        assert flat["recipe_output_noise_std"] == recipe.output_noise_std == 0.029
        assert "network_lr_horizon_cap" not in flat and resolve_config(flat)[1] == recipe
    with pytest.raises(ValueError, match="package formulation"):
        resolve_config({"recipe_defaults": "particlegan", "reg_arm": "b_cap"})
    with pytest.raises(ValueError, match="recipe_defaults"):
        resolve_config({"recipe_defaults": "gan_v4"})
    with pytest.raises(ValueError, match="recipe_output_noise_std"):
        resolve_config({"recipe_output_noise_std": 0.029})
    # Every other config keeps resolving on the archived GAN v3 fields.
    legacy = load_config(root / "configs/toy100/constraints_simple_regularization.json")
    assert resolve_config(resolve_problem_config(legacy, "grid100"))[1].reg_arm == "b_cap"

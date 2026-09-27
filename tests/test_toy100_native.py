"""toy100 declared as a problem on the shared runner (benchmarks.toy100.native)."""
import ast
import json
from pathlib import Path

import torch
from torch import nn

from benchmarks.toy100 import __main__ as cli
from benchmarks.toy100.native import PRIOR_BOX, Toy100, record
from benchmarks.toy100.problems import PROBLEM_NAMES
from benchmarks.toy_runner import ToyRun
from particlegan import get_recipe, learning_rate_scales

ROOT = Path(__file__).resolve().parents[1]


def test_native_module_is_problem_only():
    source = (ROOT / "benchmarks/toy100/native.py").read_text()
    tree = ast.parse(source)
    names = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "learning_rate_scales",
                "schedule_optimizer", "GANTrainer", "step_with_policy", "LegacyRecipe"} & names
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
              and isinstance(n.slice, ast.Constant) and n.slice.value == "lr"]
    assert not stores  # rates are only read back as receipts
    assert "GANLoss" not in source and "GradientPenalty" not in source and "reg_arm" not in source


def test_recipe_is_shipped_with_task_shape_only():
    shipped, recipe = get_recipe().to_dict(), Toy100("grid100", steps=1234).recipe().to_dict()
    changed = {key for key in shipped if shipped[key] != recipe[key]}
    assert changed <= {"z_dim", "num_particles", "batch_size", "total_steps"}
    assert recipe["total_steps"] == 1234


def test_networks_are_the_affine_square_model_with_xavier_critic():
    first = ToyRun(Toy100("rotated100"), seed=1234)
    again = ToyRun(Toy100("rotated100"), seed=1234)
    nets = first.nets
    assert torch.equal(nets.generator.weight, torch.eye(2)) and not nets.generator.bias.any()
    z = nets.prior.z.detach()
    assert z.shape == (20_000, 2) and z.abs().max() <= PRIOR_BOX
    assert torch.equal(z, again.nets.prior.z)
    for layer in (m for m in nets.critics.modules() if isinstance(m, nn.Linear)):
        bound = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
        assert layer.weight.abs().max() <= bound and not layer.bias.any()
    assert [g.get("role") for g in first.opt_g.param_groups] == ["network", "prior"]


def test_record_writes_gate_evidence_and_optimizer_owned_rates(tmp_path):
    steps = 12
    summary = record("grid100", tmp_path / "grid100", steps=steps)
    assert summary["status"] == "complete" and summary["completed_steps"] == steps
    folder = tmp_path / "grid100"
    for name in ("config.json", "events.jsonl", "final_samples.npz", "holdout_samples.npz"):
        assert (folder / name).is_file()
    config = json.loads((folder / "config.json").read_text())
    assert summary["config"] == config and config["runner"] == "benchmarks.toy_runner"
    recipe = Toy100("grid100", steps=steps).recipe()
    rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    train = [row for row in rows if row["event"] == "train"]
    assert [row["step"] for row in train] == list(range(1, steps + 1))
    for row in train:
        network, prior = learning_rate_scales(row["step"] - 1, recipe)
        assert abs(row["lr_g"] - recipe.lr * network) < 1e-12
        assert abs(row["lr_d"] - recipe.lr * recipe.d_lr_mult * network) < 1e-12
        assert abs(row["lr_prior"] - recipe.lr * recipe.prior_lr_mult * prior) < 1e-12
    evals = [row for row in rows if row["event"] == "eval" and row["model"] == "live"]
    assert [row["step"] for row in evals] == summary["eval_steps"] == [0, 1, 10, steps]


def test_default_run_command_is_native():
    args = cli._parser().parse_args(["run", "--output", "/tmp/toy100-example", "--no-render"])
    assert args.config is None and args.require_accuracy is True
    assert set(PROBLEM_NAMES) == {"grid100", "rotated100", "staggered100"}

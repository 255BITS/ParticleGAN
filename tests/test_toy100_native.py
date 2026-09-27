"""toy100 declared as a problem on the shared runner (benchmarks.toy100.native)."""
import ast
import json
from pathlib import Path

import torch
from torch import nn

from benchmarks.toy100 import __main__ as cli
from benchmarks.toy100.evidence import record
from benchmarks.toy100.native import PRIOR_STD, Toy100
from benchmarks.toy100.problems import PROBLEM_NAMES
from benchmarks.toy_runner import ToyRun
from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_native_module_is_problem_only():
    source = (ROOT / "benchmarks/toy100/native.py").read_text()
    tree = ast.parse(source)
    names = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "learning_rate_scales",
                "schedule_optimizer", "GANTrainer", "step_with_policy", "LegacyRecipe", "ToyRun", "run",
                "nn.init", "uniform_", "normal_", "xavier_uniform_"} & names
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]  # the runner owns the loop
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {"train", "gate", "schedule", "models", "config"} & imports
    assert "GANLoss" not in source and "GradientPenalty" not in source and "reg_arm" not in source


def test_evidence_harness_uses_the_shared_run_loop():
    tree = ast.parse((ROOT / "benchmarks/toy100/evidence.py").read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))
                and "step" in ast.unparse(n.iter if isinstance(n, ast.For) else n.test)
                and "range" in ast.unparse(n)]
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
              and isinstance(n.slice, ast.Constant) and n.slice.value == "lr"]
    assert not stores and "param_groups" not in ast.unparse(tree)


def test_recipe_is_shipped_with_task_shape_only():
    shipped, recipe = get_recipe().to_dict(), Toy100("grid100", steps=1234).recipe().to_dict()
    changed = {key for key in shipped if shipped[key] != recipe[key]}
    assert changed <= {"z_dim", "num_particles", "batch_size", "total_steps"}
    assert recipe["total_steps"] == 1234


def test_networks_are_declared_with_particlegan_init():
    first = ToyRun(Toy100("rotated100"), seed=0)
    again = ToyRun(Toy100("rotated100"), seed=0)
    nets = first.nets
    assert torch.equal(nets.generator.weight, torch.eye(2)) and not nets.generator.bias.any()
    z = nets.prior.z.detach()
    assert z.shape == (20_000, 2) and abs(float(z.std()) - PRIOR_STD) < 0.05 * PRIOR_STD
    assert torch.equal(z, again.nets.prior.z)
    for a, b in zip(nets.critics.parameters(), again.nets.critics.parameters()):
        assert torch.equal(a, b)
    for layer in (m for m in nets.critics.modules() if isinstance(m, nn.Linear)):
        if layer.out_features <= layer.in_features:
            gram = layer.weight @ layer.weight.T
            assert torch.allclose(gram / gram[0, 0], torch.eye(layer.out_features), atol=1e-4)
    assert [g.get("role") for g in first.opt_g.param_groups] == ["network", "prior"]


def test_metrics_are_the_declared_ones():
    toy = ToyRun(Toy100("grid100"), seed=0)
    metrics = toy.measure()
    assert metrics["n"] == 20_000 and metrics["verdict"] in {"PASS", "FAIL"} and "modes" in metrics


def test_record_writes_gate_evidence(tmp_path):
    steps = 12
    summary = record("grid100", tmp_path / "grid100", steps=steps)
    assert summary["status"] == "complete" and summary["completed_steps"] == steps
    folder = tmp_path / "grid100"
    for name in ("config.json", "events.jsonl", "final_samples.npz", "holdout_samples.npz", "progress.log"):
        assert (folder / name).is_file()
    config = json.loads((folder / "config.json").read_text())
    assert summary["config"] == config and config["runner"] == "benchmarks.toy_runner" and config["seed"] == 0
    rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    for model in ("live", "ema"):
        evals = [row for row in rows if row["event"] == "eval" and row["model"] == model]
        assert [row["step"] for row in evals] == summary["eval_steps"] == [0, 1, 10, steps]
    live = [row["metrics"] for row in rows if row["model"] == "live"]
    assert live[-1] == summary["final"]["live"]
    progress = [json.loads(line) for line in (folder / "progress.log").read_text().splitlines()]
    assert progress[-1]["event"] == "final" and progress[-1]["live"]["modes"] == live[-1]["modes"]


def test_default_run_command_is_native():
    args = cli._parser().parse_args(["run", "--output", "/tmp/toy100-example", "--no-render"])
    assert args.config is None and args.require_accuracy is True
    assert set(PROBLEM_NAMES) == {"grid100", "rotated100", "staggered100"}

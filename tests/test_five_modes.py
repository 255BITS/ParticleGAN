"""examples/five_modes.py declares only its problem and runs on the shared toy runner."""
import ast
from pathlib import Path

import torch

from benchmarks.toy_runner import ToyRun, run
from examples.five_modes import FiveModes, str_to_tensor, tensor_to_str, WORDS

ROOT = Path(__file__).resolve().parents[1]


def test_five_modes_is_problem_only():
    source = (ROOT / "examples/five_modes.py").read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    banned = {"torch.optim", "scale_learning_rates", "learning_rate_scale", "schedule_optimizer",
              "make_optimizers", "make_critic_penalty", "make_loss", "matplotlib", "plt", "nn.init"}
    assert not banned & names
    assert '["lr"]' not in source and "torch.cuda" not in source


def test_five_modes_recipe_is_shipped_at_task_shape():
    recipe = FiveModes().recipe()
    shipped = type(recipe)().to_dict()
    changed = {k for k, v in recipe.to_dict().items() if shipped[k] != v}
    assert changed <= {"z_dim", "num_particles", "batch_size", "total_steps"}


def test_text_codec_round_trips():
    assert tensor_to_str(str_to_tensor(WORDS)) == WORDS


def test_five_modes_runs_and_resumes_exactly():
    problem = FiveModes()
    recipe = problem.recipe().replace(total_steps=6, batch_size=32)
    result = run(problem, recipe=recipe, observe_every=3)
    assert result["verdict"] in {"PASS", "FAIL"}
    assert set(result["ema"]) >= {"recon_acc", "particle_words", "sample_modes", "verdict"}

    straight = ToyRun(problem, recipe=recipe)
    for _ in range(4):
        straight.step()
    resumed = ToyRun(problem, recipe=recipe)
    for _ in range(2):
        resumed.step()
    state = resumed.state_dict()
    resumed = ToyRun(problem, recipe=recipe)
    resumed.load_state_dict(state)
    for _ in range(2):
        resumed.step()
    for a, b in zip(straight.nets.generator_side(), resumed.nets.generator_side()):
        for p, q in zip(a.parameters(), b.parameters()):
            assert torch.equal(p, q)

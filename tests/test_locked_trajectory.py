"""The locked trajectory toy is problem-only on the shared runner."""
import ast
from pathlib import Path

import pytest
import torch

from benchmarks.locked_shared import trajectory
from benchmarks.toy_runner import ToyRun, run
from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_trajectory_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/trajectory.py").read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer",
                "requires_grad_", "noise_policy"} & names
    for banned in ('["lr"]', "GANLoss", "GradientPenalty", "make_gan_loss", "make_b_cap", "ParticleRegularizer",
                   "gan_factory", "cap_factory"):
        assert banned not in source


def test_recipe_is_shipped_at_task_shape():
    recipe = trajectory.Trajectory().recipe()
    shape = dict(z_dim=4, num_particles=12, batch_size=12, total_steps=400)
    assert recipe == get_recipe(**shape)


def test_conditional_critic_sees_the_pairing():
    problem = trajectory.Trajectory(pairing="stranger")
    real = problem.real(12, None)
    assert torch.equal(real.condition[0], problem.slow)
    assert torch.equal(real.x, problem.fast[(torch.arange(12) + 6) % 12])
    with pytest.raises(ValueError, match="whole set"):
        problem.real(8, None)


def test_short_run_is_deterministic_and_resumes_exactly():
    recipe = trajectory.Trajectory().recipe().replace(total_steps=12)
    first = run(trajectory.Trajectory(detailed=True), recipe=recipe, observe_every=6)
    again = run(trajectory.Trajectory(detailed=True), recipe=recipe, observe_every=6)
    assert first["live"] == again["live"] and first["ema"] == again["ema"]
    assert {"identity_mse", "paired_target_mse", "critic_gradient_median", "own_nearest_fraction"} <= set(first["live"])
    assert first["verdict"] == ("PASS" if first["live"]["identity_mse"] <= 0.02 else "FAIL")

    straight = ToyRun(trajectory.Trajectory(), recipe=recipe)
    for _ in range(12):
        straight.step()
    resumed = ToyRun(trajectory.Trajectory(), recipe=recipe)
    for _ in range(5):
        resumed.step()
    state = resumed.state_dict()
    resumed = ToyRun(trajectory.Trajectory(), recipe=recipe)
    resumed.load_state_dict(state)
    for _ in range(7):
        resumed.step()
    assert straight.measure() == resumed.measure()


def test_residual_student_exports_are_unchanged():
    slow, fast = trajectory.trajectories()
    assert slow.shape == fast.shape == (12, 16)
    assert trajectory.PROTOCOL["n_particles"] == 12 and trajectory.PROTOCOL["steps"] == 400
    assert trajectory.pairing_index("nearest_stranger", slow).ne(torch.arange(12)).all()

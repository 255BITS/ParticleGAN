"""ExtraAdam must evaluate a joint game and correct from original weights."""

import ast
from copy import deepcopy
import inspect
import io
import hashlib
import tarfile

import pytest
import torch

from benchmarks.locked_shared import mode_hold, trajectory
from reports.toy100.extra_adam_scratch import (
    ExtraAdamRecorder, HOSTS, transformed_function,
)
from reports.toy100.extra_adam_probe import _verify_transform


@pytest.mark.parametrize("method,evaluations", [("extra_adam", 2), ("sim_adam", 1)])
def test_bilinear_joint_gradients_and_exact_moment_formula(method, evaluations):
    d = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    g = torch.nn.Parameter(torch.tensor(-.4, dtype=torch.float64))
    opt_d = torch.optim.Adam([d], lr=.03, betas=(.2, .8), eps=1e-7)
    opt_g = torch.optim.Adam([g], lr=.05, betas=(.2, .8), eps=1e-7)
    recorder = ExtraAdamRecorder(method)
    expected = torch.tensor([.7, -.4], dtype=torch.float64)
    rates = torch.tensor([.03, .05], dtype=torch.float64)
    first = torch.zeros(2, dtype=torch.float64)
    second = torch.zeros(2, dtype=torch.float64)
    moments = 0
    for outer in range(5):
        base = expected.clone()
        for phase in recorder.phases(outer, opt_d, opt_g):
            gradient = torch.stack((-expected[1], expected[0]))
            first = .2 * first + .8 * gradient
            second = .8 * second + .2 * gradient.square()
            moments += 1
            direction = (first / (1 - .2 ** moments)) / ((second / (1 - .8 ** moments)).sqrt() + 1e-7)
            expected = base - rates * direction
            old = torch.stack((d.detach(), g.detach()))
            opt_d.zero_grad()
            (-d * g).backward()
            recorder.step(opt_d, torch.optim.Adam.step)
            assert torch.equal(torch.stack((d.detach(), g.detach())), old)
            opt_g.zero_grad()
            (d * g).backward()
            recorder.step(opt_g, torch.optim.Adam.step)
            assert torch.allclose(torch.stack((d.detach(), g.detach())), expected, atol=2e-15, rtol=0)
    receipt = recorder.receipt()
    assert receipt["outer_steps"] == 5
    assert receipt["joint_points_verified"] == 5 * evaluations
    assert receipt["base_restores_verified"] == (5 if evaluations == 2 else 0)
    for row in receipt["optimizers"]:
        assert row["calls"] == 5 * evaluations
        assert row["groups"][0]["moment_steps"] == [5 * evaluations]
        assert row["rates"] == [[row["groups"][0]["lr"]]] * (5 * evaluations)


@pytest.mark.parametrize("task,module", [("mode_hold", mode_hold), ("trajectory", trajectory)])
def test_transform_wraps_only_gradients_and_is_repeatable(task, module):
    before = inspect.getsource(getattr(module, HOSTS[task]))
    tree, source, original_sha = transformed_function(module, task)
    assert inspect.getsource(getattr(module, HOSTS[task])) == before
    assert transformed_function(module, task)[1:] == (source, original_sha)
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "_extra_phase")
    calls = [ast.unparse(node.func) for node in ast.walk(loop) if isinstance(node, ast.Call)]
    assert calls.count("opt_d.step") == calls.count("opt_g.step") == 1
    assert calls.count("d_loss.backward") == calls.count("g_loss.backward") == 1
    assert "checkpoint" not in calls
    assert "noise_policy.set_step" not in calls
    assert len([node for node in ast.walk(loop) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "_extra_phase == 0"]) == 2


def test_unexpected_parameter_mutation_is_rejected():
    d = torch.nn.Parameter(torch.tensor(.7))
    g = torch.nn.Parameter(torch.tensor(-.4))
    opt_d, opt_g = torch.optim.Adam([d]), torch.optim.Adam([g])
    recorder = ExtraAdamRecorder("extra_adam")
    phases = recorder.phases(0, opt_d, opt_g)
    next(phases)
    d.grad = torch.ones_like(d)
    recorder.step(opt_d, torch.optim.Adam.step)
    with torch.no_grad():
        d.add_(.1)
    g.grad = torch.ones_like(g)
    with pytest.raises(RuntimeError, match="before both game gradients"):
        recorder.step(opt_g, torch.optim.Adam.step)


def test_optimizer_state_preserves_both_moment_evaluations():
    parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    opponent = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    opt_d, opt_g = torch.optim.Adam([opponent]), torch.optim.Adam([parameter])
    recorder = ExtraAdamRecorder("extra_adam")
    for phase in recorder.phases(0, opt_d, opt_g):
        opponent.grad = torch.tensor([float(phase + 1)], dtype=torch.float64)
        recorder.step(opt_d, torch.optim.Adam.step)
        parameter.grad = torch.tensor([float(3 + phase)], dtype=torch.float64)
        recorder.step(opt_g, torch.optim.Adam.step)
    assert opt_g.state[parameter]["step"] == 2
    assert opt_g.state[parameter]["exp_avg"].item() == pytest.approx(.9 * .3 + .1 * 4)
    saved = deepcopy(opt_g.state_dict())
    restored = torch.optim.Adam([torch.nn.Parameter(parameter.detach().clone())])
    restored.load_state_dict(saved)
    assert restored.state_dict()["state"][0]["step"] == 2


@pytest.mark.parametrize("task,module", [("mode_hold", mode_hold), ("trajectory", trajectory)])
def test_archived_transform_regrade_rejects_rehashed_extra_assignment(tmp_path, task, module):
    _, source, _ = transformed_function(module, task)
    path = tmp_path / "generated_host.py"
    path.write_text(source)
    original = inspect.getsource(module).encode()
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        item = tarfile.TarInfo(f"benchmarks/locked_shared/{task}.py")
        item.size = len(original)
        archive.addfile(item, io.BytesIO(original))
    expected = dict(generated_function_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    _verify_transform(tmp_path, task, expected)
    tree = ast.parse(source)
    tree.body[0].body.insert(0, ast.Assign(targets=[ast.Name(id="unexpected_change", ctx=ast.Store())],
                                        value=ast.Constant(1)))
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree) + "\n")
    expected["generated_function_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(AssertionError):
        _verify_transform(tmp_path, task, expected)

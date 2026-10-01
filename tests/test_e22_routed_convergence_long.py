"""Opt-in learned convergence regression for the declared two-site family.

PARTICLEGAN_RUN_ROUTED_CONVERGENCE_LONG=1 python -m pytest -q \
    tests/test_e22_routed_convergence_long.py

This trains the fixed three-arm parent and sole H/b-neutral particle arm. It
then scores actual saved generators under all four learned native critics.
No pre-existing receipt, output-error threshold, selected seed, checkpoint,
or critic can satisfy the learned regression. This is not a Forge/Supra gate.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.nn.functional as F

from examples import e22_routed_convergence as base
from examples import e22_routed_convergence_neutral as neutral


ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def _game(loop, judge, panels, *, ablate=False):
    values = loop.data["test"]
    prediction = torch.cat([
        base.forward(loop, values["context"][i:i + base.BATCH_SIZE], code_ablation=ablate)
        for i in range(0, len(values["context"]), base.BATCH_SIZE)
    ])
    residual = (prediction - values["targets"]) / loop.data["scale"]
    condition = values["context"][:, 0, base.WIDTH:]
    native_loss = base.get_recipe("e22").make_loss()
    scores = []
    for panel in panels["test"]:
        fake, real = judge(panel + residual, condition), judge(panel, condition)
        value = F.softplus(real - fake).mean()
        torch.testing.assert_close(value, native_loss.g_loss(fake, real), rtol=0, atol=1e-7)
        scores.append(value)
    result = float(torch.stack(scores).mean())
    assert torch.isfinite(torch.tensor(result))
    return result


def _assert_registered_game_regression(parent, candidate):
    """Independent tensor-based assertion, also usable on qualified artifacts."""
    original, modified = base.make_data(), neutral.make_neutral_data()
    parent_states = {arm: _load(parent / arm / "step-6400.pt") for arm in base.ARMS}
    candidate_state = _load(candidate / "step-6400.pt")
    assert all(state["step"] == 6400 for state in (*parent_states.values(), candidate_state))
    loops = {}
    for arm, state in parent_states.items():
        loops[arm] = base.make_loop(arm, original, bindings=state["law"]["bindings"])
        base.restore(loops[arm], state)
        assert base.digest(base.checkpoint(loops[arm])) == base.digest(state)
    proposed = neutral.make_neutral_loop(modified, bindings=candidate_state["law"]["bindings"])
    base.restore(proposed, candidate_state)
    assert base.digest(base.checkpoint(proposed)) == base.digest(candidate_state)
    assert proposed.policy.routed_control.spec.max_context_harm == 0
    assert not proposed.policy.routed_control.spec.output_error_guard

    # Actual learned ownership, not a receipt's particle-support flag.
    assert proposed.policy.table.requires_grad and proposed.policy.table.grad.norm() > 0
    assert (proposed.policy.table.grad.norm(dim=1) > 0).sum() == base.PARTICLES
    assert all(parameter.requires_grad and parameter.grad is not None and parameter.grad.norm() > 0
               for parameter in proposed.policy.router.parameters())
    for site in ("first", "second"):
        bridge = getattr(proposed.G, site).bridge
        assert bridge.weight.requires_grad and bridge.bias.requires_grad
        assert bridge.weight[:, base.RANK:].count_nonzero() > 0

    panels = base.evaluation_panels(original)
    assert base.digest(panels) == base.digest(base.evaluation_panels(modified))
    before = {name: base.digest(base.checkpoint(loop)) for name, loop in loops.items()}
    candidate_before = base.digest(base.checkpoint(proposed))
    results = {}
    for arm in ("ordinary_native_game", "particle_native_game"):
        for step in (800, 6400):
            state = _load(parent / arm / f"step-{step:04d}.pt")
            assert state["step"] == step and state["law"] == parent_states[arm]["law"]
            with torch.random.fork_rng(devices=[]):
                judge = base.ConditionalCritic(original["scale"]).eval().requires_grad_(False)
            judge.load_state_dict(state["training"]["models"]["critic"], strict=True)
            values = {name: _game(loop, judge, panels) for name, loop in loops.items()}
            values["neutral_particle"] = _game(proposed, judge, panels)
            values["zero_code_minus_live"] = _game(proposed, judge, panels, ablate=True) - values["neutral_particle"]
            improvement = values["particle_native_game"] - values["neutral_particle"]
            assert improvement > 1e-4, (arm, step, values)
            assert abs(values["zero_code_minus_live"]) > 1e-6, (arm, step, values)
            if step == 6400:
                gap = values["particle_native_game"] - values["ordinary_native_game"]
                assert gap > 1e-4, (arm, step, values)
                assert improvement / gap >= .5, (arm, step, values)
            results[f"{arm}@{step}"] = values
    assert all(before[name] == base.digest(base.checkpoint(loop)) for name, loop in loops.items())
    assert candidate_before == base.digest(base.checkpoint(proposed))
    return results


@pytest.mark.skipif(
    os.environ.get("PARTICLEGAN_RUN_ROUTED_CONVERGENCE_LONG") != "1",
    reason="opt-in fixed 6400-step learned regression; parent 2700s + particle 900s budgets",
)
def test_hb_neutral_initialization_improves_learned_game_with_live_particles(tmp_path):
    parent, candidate = tmp_path / "parent", tmp_path / "neutral"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    commands = [
        ([sys.executable, "-u", "examples/run_e22_routed_convergence.py", "--out", str(parent)], 2730),
        ([sys.executable, "-u", "examples/e22_routed_convergence_neutral.py",
          "--parent", str(parent), "--out", str(candidate)], 930),
    ]
    execution_seconds = []
    for index, (command, timeout) in enumerate(commands):
        # Preserve complete local output for diagnosing a learned failure.
        start = time.monotonic()
        with (tmp_path / f"run-{index}.log").open("w") as output:
            subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                           stderr=subprocess.STDOUT, timeout=timeout, check=True)
        execution_seconds.append(time.monotonic() - start)
    scoring_start = time.monotonic()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            results = _assert_registered_game_regression(parent, candidate)
    finally:
        torch.set_num_threads(threads)
    assert execution_seconds[0] <= 2700
    assert execution_seconds[1] + time.monotonic() - scoring_start <= 900
    print(json.dumps({"registered_family_game_regression": results}, allow_nan=False))

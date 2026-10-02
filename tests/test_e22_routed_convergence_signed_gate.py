"""Synthetic scoring controls for the beneficial-particle assertion.

These use three-update tensor fixtures and explicitly synthetic endpoint
labels/scores. They exercise the real regression's ownership, restoration and
state-purity checks, but supply no learned 6,400-update qualification evidence.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

from examples import e22_routed_convergence as base
from examples import e22_routed_convergence_neutral as neutral
from examples import evaluate_e22_routed_convergence_neutral as recovery


@pytest.fixture(scope="module")
def scoring_fixture():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        original = base.make_data()
        modified = neutral.make_neutral_data(original)
        states = {}
        for arm in base.ARMS:
            loop = base.make_loop(arm, original)
            for _ in range(3):
                base.update(loop)
            states[arm] = base.checkpoint(loop)
        loop = neutral.make_neutral_loop(modified)
        for _ in range(3):
            base.update(loop)
        states["neutral"] = base.checkpoint(loop)
        path = Path(__file__).with_name("test_e22_routed_convergence_long.py")
        spec = importlib.util.spec_from_file_location("routed_long_gate_control", path)
        regression = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regression)
        yield regression, states
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


JUDGES = tuple(f"{arm}@{step}" for arm in ("ordinary_native_game", "particle_native_game")
               for step in (800, 6400))


@pytest.mark.parametrize("harmful_judge", (None, *JUDGES))
def test_actual_regression_rejects_harmful_code_effect_under_any_required_judge(
        monkeypatch, scoring_fixture, harmful_judge):
    regression, states = scoring_fixture
    current_judge = [None]

    def synthetic_load(path):
        arm, step = path.parent.name, int(path.stem.split("-")[1])
        current_judge[0] = f"{arm}@{step}"
        state = deepcopy(states[arm])
        # Synthetic labels only: keep fixture and policy counters consistent
        # so the complete assertion reaches its scoring gate.
        state["step"] = step
        if "completed_steps" in state["training"]:
            state["training"]["completed_steps"] = step
        return state

    def synthetic_game(loop, judge, panels, *, ablate=False):
        if loop.law["task"] == neutral.TASK:
            if ablate:
                return .79 if current_judge[0] == harmful_judge else .91
            return .8
        return {"ordinary_native_game": 1., "particle_native_game": 2.,
                "ordinary_mse_reference": .9}[loop.arm]

    monkeypatch.setattr(regression, "_load", synthetic_load)
    monkeypatch.setattr(regression, "_game", synthetic_game)
    if harmful_judge is None:
        result = regression._assert_registered_game_regression(Path("parent"), Path("neutral"))
        assert tuple(result) == JUDGES
        assert all(value["zero_code_minus_live"] > 0 for value in result.values())
    else:
        with pytest.raises(AssertionError) as error:
            regression._assert_registered_game_regression(Path("parent"), Path("neutral"))
        arm, step, values = error.value.args[0]
        assert f"{arm}@{step}" == harmful_judge
        assert values["zero_code_minus_live"] < 0
        assert values["particle_native_game"] - values["neutral_particle"] > 1e-4


def archived_and_repaired_helpers():
    repaired = Path(neutral.__file__).read_text()
    signed = 'all(value > 1e-6 for value in particle_witness["zero_code_minus_live_test_game"].values())'
    unsigned = signed.replace("all(value", "all(abs(value)")
    assert repaired.count(signed) == 1
    original = repaired.replace(signed, unsigned).replace(
        '        loop = make_neutral_loop(data, bindings=bindings)\n        with (args.out / "common-judge-curves.jsonl")',
        '        with (args.out / "common-judge-curves.jsonl")',
    )
    return original, repaired


def test_recovery_proof_records_both_evaluation_changes_and_old_training_gate():
    original, repaired = archived_and_repaired_helpers()
    proof = recovery.observational_source_repair(original, repaired)
    assert proof["training_ast_unchanged"]
    assert proof["trained_retained_particle_gate"] == "unsigned_effect_v1"
    assert proof["evaluated_retained_particle_gate"] == "beneficial_signed_v2"
    assert len(proof["changes"]) == 2


def test_recovery_proof_rejects_an_extra_training_change():
    original, repaired = archived_and_repaired_helpers()
    assert "range(1, 6401)" in repaired
    changed = repaired.replace("range(1, 6401)", "range(1, 6402)")
    with pytest.raises(ValueError, match="more than"):
        recovery.observational_source_repair(original, changed)


def test_recovery_proof_rejects_a_missing_signed_gate():
    original, _ = archived_and_repaired_helpers()
    with pytest.raises(ValueError, match="exactly one"):
        recovery.observational_source_repair(original, original)

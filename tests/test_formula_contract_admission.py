"""Semantic admission controls: unsupported execution cannot repair the formula."""
from copy import deepcopy
import json

import pytest

from ops import formulaic_coverage as coverage


@pytest.mark.parametrize("task", sorted(coverage.DDGAN_TASKS | coverage.HELD_TIER34_TASKS))
@pytest.mark.parametrize("candidate", [
    {"id": "e22", "schema_version": 1, "recipe_overrides": {}},
    {"id": "control", "schema_version": 3,
     "prior_mechanism": {"kind": "mog", "sigma": 0.025},
     "recipe_overrides": {"output_noise_mode": "learnable", "output_noise_std": .03,
                          "input_noise_std": .1, "d_guard_min_steps": 200}},
])
def test_legacy_accessor_never_substitutes_any_recipe_field(task, candidate):
    original = deepcopy(candidate)
    adapted = coverage.adapt_candidate_for_task(candidate, {"id": task})
    assert adapted == original and candidate == original
    adapted["recipe_overrides"]["caller_edit"] = True
    assert candidate == original


def test_retired_coverage_cannot_launch_or_overwrite_evidence_even_with_force(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    receipt = state / "result.json"
    original = '{"status":"FAIL","source_digest":"original"}\n'
    receipt.write_text(original)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    with pytest.raises(coverage.RetiredCoverageExecutionError, match="frozen study"):
        coverage.ensure_formulaic_attempt_coverage(
            tmp_path, state, ["0", "1"], force=True, tasks="all")
    with pytest.raises(coverage.RetiredCoverageExecutionError, match="retired"):
        coverage.execute_single_task(tmp_path, "family", "candidate", "task", "cpu", state)
    assert receipt.read_text() == original
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_legacy_batch_cli_refuses_without_reading_or_creating_files(tmp_path, capsys):
    missing = tmp_path / "no-spec.json"
    assert coverage.main(["--run-batch", str(missing), "--force"]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "BLOCKED"
    assert refusal["qualification_input"] is False
    assert refusal["execution_authorized"] is False
    assert list(tmp_path.iterdir()) == []


def test_refresh_surfaces_explicit_admission_refusal(tmp_path):
    from experiments.forge.contracts import atomic_json
    from experiments.forge.leaderboard_contract import FIXED_TEST_LIST
    from ops.refresh_forge_readouts import CodeChangeRequiredError, refresh_attempt_coverage

    atomic_json(tmp_path / FIXED_TEST_LIST, {"tests": ["grid100"]})
    with pytest.raises(CodeChangeRequiredError, match="alternate formulaic coverage executor is retired"):
        refresh_attempt_coverage(tmp_path, tmp_path / "queue", tmp_path / "coverage", gpus=["0", "1"])

"""Importance must not conceal failure or make pathological tests veto selection."""
import copy

import pytest

from benchmarks.transfer_suite.protocol import (
    reference_evidence, selection_key, summarize, test_verdict as verdict, validate_manifest,
)


def spec(name, tier, family="family", split="development"):
    return dict(name=name, tier=tier, family=family, split=split,
                phase="reserved" if split == "reserved" else "fit", steps=24,
                thresholds=[["quality", ">=", .9]], importance_reason="Declared use case.",
                limitations="Synthetic fixture.", runner="fixture")


def result(passes=True, first=1):
    curve = [dict(step=i, quality=1. if passes and i >= first else 0.) for i in range(1, 25)]
    return dict(live=curve[-1], observations=curve, ema={"quality": 1.})


def test_diagnostics_cannot_block_or_change_rank():
    manifest = dict(tasks=[spec("core", "required"), spec("useful", "ranking"),
                           spec("impossible", "diagnostic")])
    rows = dict(core=result(), useful=result())
    missing = summarize(manifest, rows)
    failed = summarize(manifest, {**rows, "impossible": result(False)})
    passed = summarize(manifest, {**rows, "impossible": result()})
    assert missing["selection_ready"] and failed["selection_ready"] and passed["selection_ready"]
    assert selection_key(missing) == selection_key(failed) == selection_key(passed)
    assert failed["counts"]["diagnostic"]["passed"] == 0


def test_ranking_failures_do_not_veto_but_required_failures_do():
    manifest = dict(tasks=[spec("core", "required"), spec("useful", "ranking")])
    low = summarize(manifest, dict(core=result(), useful=result(False)))
    high = summarize(manifest, dict(core=result(), useful=result()))
    bad_core = summarize(manifest, dict(core=result(False), useful=result()))
    assert low["selection_ready"] and high["selection_ready"]
    assert not bad_core["eligible"]
    assert selection_key(high) < selection_key(low) < selection_key(bad_core)


def test_duplicate_family_cases_do_not_outvote_other_families():
    tasks = [spec("core", "required"), spec("easy", "ranking", "easy"), spec("hard", "ranking", "hard")]
    rows = dict(core=result(), easy=result(), hard=result(False))
    before = summarize(dict(tasks=tasks), rows)
    for i in range(20):
        tasks.append(spec(f"clone{i}", "ranking", "easy"))
        rows[f"clone{i}"] = result()
    after = summarize(dict(tasks=tasks), rows)
    assert before["ranking_pass_fraction"] == after["ranking_pass_fraction"] == .5
    assert selection_key(before) == selection_key(after)


def test_missing_ranking_attempt_not_selectable_and_error_is_visible_failure():
    manifest = dict(tasks=[spec("core", "required"), spec("useful", "ranking")])
    missing = summarize(manifest, dict(core=result()))
    crashed = summarize(manifest, dict(core=result(), useful={"error": "nonfinite training"}))
    assert missing["eligible"] and not missing["selection_ready"]
    assert crashed["selection_ready"] and crashed["ranking_pass_fraction"] == 0
    assert crashed["verdicts"]["useful"]["status"] == "ERROR"


def test_many_vector_families_cannot_drown_out_image_domain():
    tasks = [spec("core", "required"), {**spec("image", "ranking", "cnn"), "runner": "image"}]
    rows = dict(core=result(), image=result(False))
    for i in range(10):
        tasks.append({**spec(f"vector{i}", "ranking", f"family{i}"), "runner": "vector"})
        rows[f"vector{i}"] = result()
    score = summarize(dict(tasks=tasks), rows)
    assert score["ranking_pass_fraction"] == .5
    assert score["domains"]["image"]["pass_fraction"] == 0


def test_live_complete_sustained_curve_required_even_with_ema_or_stale_stamp():
    task = spec("core", "required")
    transient = result(first=21)  # Four passing points cannot satisfy the five-point rule.
    transient["convergence"] = {"confirmed_step": 24}
    assert verdict(task, transient)["status"] == "FAIL"
    incomplete = result()
    incomplete["observations"].pop(3)
    assert verdict(task, incomplete)["status"] == "INCOMPLETE"
    nonfinite = result()
    nonfinite["live"]["quality"] = float("nan")
    assert not verdict(task, nonfinite)["passed"]


@pytest.mark.parametrize("modes,reconstruction", [(3, 1), (5, 0), (6, 1)])
def test_exact_word_gates_reject_complete_wrong_counts_or_pairs(modes, reconstruction):
    task = {**spec("words", "required"), "thresholds": [
        ["modes", "==", 5], ["reconstruction_exact", "==", 1]]}
    curve = [{"step": i, "modes": modes, "reconstruction_exact": reconstruction}
             for i in range(1, 25)]
    measured = verdict(task, {"live": curve[-1], "observations": curve})
    assert measured["status"] == "FAIL"
    assert measured["convergence"]["passing_observations"] == 0
    assert any(cell["margin"] < 0 for cell in measured["metrics"])


def test_exact_oracle_and_unknown_operator_are_explicit():
    from benchmarks.locked_shared.baseline import score_metrics
    from benchmarks.locked_shared.observation import sustained
    requirements = [["modes", "==", 5], ["reconstruction_exact", "==", 1]]
    curve = [{"step": i, "modes": 5, "reconstruction_exact": 1} for i in range(1, 25)]
    task = {**spec("words", "required"), "thresholds": requirements}
    validate_manifest({"tasks": [task]})
    assert verdict(task, {"live": curve[-1], "observations": curve})["status"] == "PASS"
    for check in (lambda: sustained([], [["modes", "!=", 5]], expected_steps=[]),
                  lambda: score_metrics({}, [["modes", "!=", 5]])):
        with pytest.raises(ValueError, match="unsupported metric operator"):
            check()
    assert sustained(curve, [["modes", ">=", float("nan")]], expected_steps=range(1, 25))["passing_suffix"] == 0


def test_reference_solvability_does_not_change_importance():
    task = spec("hard", "ranking")
    original = copy.deepcopy(task)
    assert reference_evidence(task, {})["status"] == "unmeasured"
    assert reference_evidence(task, {"cosine": result(False)})["status"] == "not demonstrated"
    evidence = reference_evidence(task, {"cosine": result(False), "reference": result()})
    assert evidence["passing_references"] == ["reference"]
    assert task == original


def test_whole_reserved_family_cannot_appear_in_development():
    manifest = dict(tasks=[spec("core", "required"), spec("holdout", "ranking", split="reserved")])
    with pytest.raises(ValueError, match="reserved family"):
        validate_manifest(manifest)
    manifest["tasks"][1]["family"] = "unseen"
    validate_manifest(manifest)

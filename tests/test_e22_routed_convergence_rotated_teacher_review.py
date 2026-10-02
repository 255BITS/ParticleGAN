"""Cheap corruption guards for the independent reviewer; no quality training."""
from copy import deepcopy
import json
import math

import pytest
import torch

from examples import e22_routed_convergence as base
from examples import e22_routed_convergence_rotated_teacher as law
from examples import review_e22_routed_convergence_rotated_teacher as review
from examples import run_e22_routed_convergence_rotated_teacher as runner


@pytest.fixture(scope="module")
def fixture():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        data = law.make_rotated_data()
        yield data
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def test_independent_scoring_rejects_context_corruption_that_preserves_reported_mean(fixture):
    loop = law.make_rotated_loop(law.ARMS[1], fixture)
    judge = loop.policy.D.eval().requires_grad_(False)
    panels = base.evaluation_panels(fixture)
    rng, before = torch.get_rng_state().clone(), base.digest(base.checkpoint(loop))
    prediction = review.predictions(loop, fixture)["test"]
    actual = base.evaluate(loop, judge, "test", panels)
    expected = review.score_prediction(judge, fixture, "test", prediction, panels)
    review.compare_score(review.Review(), actual, expected, fixture, "test", "actual fixed test pool")
    changed = deepcopy(actual)
    changed["paired_game_by_context"][0] += .1
    changed["paired_game_by_context"][1] -= .1
    assert math.isclose(sum(changed["paired_game_by_context"]), sum(actual["paired_game_by_context"]))
    with pytest.raises(AssertionError, match="context score"):
        review.compare_score(review.Review(), changed, expected, fixture, "test", "forged pool")
    assert torch.equal(rng, torch.get_rng_state())
    assert base.digest(base.checkpoint(loop)) == before


def test_actual_initial_state_audit_rejects_changed_frozen_ema_or_recipe(fixture):
    loop = law.make_rotated_loop(law.ARMS[1], fixture)
    initial = base.checkpoint(loop)
    streams = (loop.data_rng.get_state(), loop.paired_rng.get_state())
    review.check_state(review.Review(), initial, initial, loop.law, law.ARMS[1], 0, streams)
    changed = deepcopy(initial)
    changed["training"]["averages"]["generator"]["first.base.weight"][0, 0] += 1
    with pytest.raises(AssertionError, match="frozen FAST/EMA"):
        review.check_state(review.Review(), changed, initial, loop.law, law.ARMS[1], 0, streams)
    changed = deepcopy(initial)
    changed["training"]["recipe"]["row_evidence_gate"] = False
    with pytest.raises(AssertionError, match="unchanged recipe"):
        review.check_state(review.Review(), changed, initial, loop.law, law.ARMS[1], 0, streams)
    changed = deepcopy(initial)
    changed["training"]["optimizers"][0]["param_groups"][0]["amsgrad"] = False
    with pytest.raises(AssertionError, match="AMSGrad"):
        review.check_state(review.Review(), changed, initial, loop.law, law.ARMS[1], 0, streams)


def test_nominally_finite_diagnostics_do_not_hide_nonfinite_optimizer_moments(fixture):
    loop = law.make_rotated_loop(law.ARMS[1], fixture)
    state = base.checkpoint(loop)
    state["training"]["lr_settle"][0][0]["diagnostic_only"] = float("nan")
    assert review.diagnostic_nonfinite(state["training"]["lr_settle"])
    review.finite_tensors(review.Review(), state["training"]["models"], "actual finite owners")
    state["training"]["optimizers"][0]["state"][0] = {"exp_avg": torch.tensor([float("nan")])}
    with pytest.raises(AssertionError, match="optimizer"):
        review.finite_tensors(review.Review(), state["training"]["optimizers"][0]["state"], "optimizer")


def test_independent_denominator_policy_keeps_missing_head_and_mixed_gap_visible():
    scores = {f"{arm}@6400": {name: {"test": {"paired_game": score}}
                             for name in review.JUDGES}
              for arm, score in zip(law.ARMS, (1., 1.5, .9))}
    witnesses = {arm: {"bridge_still_trainable": True, "bank_still_trainable": True,
                      "router_still_trainable": True, "C_norms": dict.fromkeys(law.SITES, .1),
                      "live_bank_updates": 4, "live_query_updates": 4,
                      "zero_code_minus_live_test_game": dict.fromkeys(review.JUDGES, -.1)}
                 for arm in law.ARMS[1:]}
    actual = review.independent_gates(scores, witnesses)
    assert actual == runner.final_gates(scores, witnesses)
    assert actual["neutral_beats_ordinary_all_four"] and actual["H_b_support_gate"]
    assert all(value < 0 for value in witnesses[law.ARMS[2]]["zero_code_minus_live_test_game"].values())
    witnesses[law.ARMS[2]]["zero_code_minus_live_test_game"].pop(review.JUDGES[0])
    actual = review.independent_gates(scores, witnesses)
    assert not actual["retained_particle_gate"][law.ARMS[2]]
    assert not actual["neutral_beats_ordinary_all_four"] and not actual["H_b_support_gate"]
    scores[f"{law.ARMS[1]}@6400"][review.JUDGES[-1]]["test"]["paired_game"] = .9
    actual = review.independent_gates(scores, witnesses)
    assert actual["endpoint_gap_reduction"] is None and not actual["support_gate_applicable"]


def test_fabricated_complete_receipt_cannot_qualify_held_source_only_task(tmp_path, monkeypatch):
    card = json.loads(runner.CARD.read_text())
    card["execution"]["execution_authorized"] = False
    held = tmp_path / "held-card.json"
    held.write_text(json.dumps(card))
    monkeypatch.setattr(runner, "CARD", held)
    (tmp_path / "receipt.json").write_text(json.dumps({
        "schema": "routed_convergence_rotated_execution_v1", "task": law.TASK,
        "contract": card, "status": "complete"}))
    with pytest.raises(AssertionError, match="source-only"):
        review.review_run(tmp_path)
    forged = json.loads((tmp_path / "receipt.json").read_text())
    forged["contract"]["execution"]["execution_authorized"] = True
    (tmp_path / "receipt.json").write_text(json.dumps(forged))
    with pytest.raises(AssertionError, match="exact held card"):
        review.review_run(tmp_path)

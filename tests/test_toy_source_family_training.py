"""Read-only observers must preserve exact routed update/replay state."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.toy_audit.source_family_ablation import zero_code_snapshot
from benchmarks.toy_audit.source_family_report import compact_observations, forbid_inline_streams
from benchmarks.toy_audit.source_routed_ring_training import (
    routed_modules, routed_parity, routed_state, state_digest, suffix,
)


QUALITY = SimpleNamespace(paired_edit_metrics=lambda prediction, target, neutral:
                          {"relative_mse": float(np.mean((prediction - target) ** 2)), "passed": False})


@pytest.mark.parametrize("name", ["paired", "replay"])
def test_clean_heldout_observer_preserves_parameters_adam_modes_and_streams(name):
    torch.set_num_threads(1)
    module, make, update, evaluate, checkpoint, neutral = routed_modules(name)
    report = routed_parity(make, update, evaluate, checkpoint, neutral, QUALITY, steps=2)
    assert report["state_and_trace_exact"]
    assert report["control"] == report["observed"]


def test_checkpointed_routed_forward_matches_plain_updates_without_rewinding_live_rng():
    torch.set_num_threads(1)
    module, make, checkpointed_update, evaluate, checkpoint, neutral = routed_modules("replay")
    initial = torch.get_rng_state().clone()
    observed = []
    for update in (module.update, checkpointed_update):
        torch.set_rng_state(initial)
        loop = make()
        rows = []
        for _ in range(2):
            with torch.autograd.set_multithreading_enabled(False):
                rows.append(update(loop))
        observed.append((state_digest(routed_state(loop, checkpoint)), state_digest(rows)))
    torch.set_rng_state(initial)
    assert observed[0] == observed[1]


@dataclass
class Candidate:
    table: torch.Tensor
    codes: torch.Tensor


def test_zero_code_intervention_keeps_bank_and_routing_weights():
    table = torch.tensor([[3., 4.], [5., 6.]])
    codes = torch.ones(3, 2)
    weights = torch.tensor([[.3, .7]]).expand(3, -1)
    def generate(models, context, candidate, actual_weights):
        assert torch.equal(candidate.table, table)
        assert torch.equal(actual_weights, weights)
        return candidate.codes
    routing = SimpleNamespace(model_forward=None, generate=generate)
    served = SimpleNamespace(routing=routing)
    result = zero_code_snapshot(served)
    assert not result["table_zeroed"]
    assert served.routing is not routing
    assert torch.equal(served.routing.generate({}, None, Candidate(table, codes), weights), torch.zeros_like(codes))
    assert torch.equal(codes, torch.ones_like(codes))
    assert torch.equal(table, torch.tensor([[3., 4.], [5., 6.]]))


def test_two_partial_passing_checks_cannot_become_a_convergence_claim():
    assert suffix([True, True])["passing_suffix"] == 2
    assert not suffix([True, True])["passed"]
    assert suffix([False, True, True, True, True, True])["passed"]


def test_fresh_catalog_join_keeps_budget_gate_and_replay_limits_explicit():
    path = Path(__file__).resolve().parents[1] / "reports/toy_audit/source-family-training.json"
    if not path.exists():
        pytest.skip("Compact publication receipt has not been generated yet")
    report = json.loads(path.read_text())
    records = {row["fixture"]: row for row in report["fixtures"]}
    assert set(records) == {"paired", "support", "moving", "replay", "ring"}
    assert {records[key]["catalog_id"] for key in ("paired", "support", "moving", "replay")} == {"source-family-14"}
    assert records["ring"]["catalog_id"] == "source-family-10"
    assert all(records[key]["original_scientific_status"] == "NO_FROZEN_GATE" for key in ("paired", "support", "moving", "replay"))
    assert records["ring"]["original_scientific_status"] == "FAIL"
    assert records["support"]["declared_budget_complete"] is False
    assert records["support"]["full_added_gate_status"] == "INCOMPLETE"
    assert records["replay"]["convergence_qualified"] is False
    assert records["ring"]["phases"]["uninterrupted_hold_2400"] == "NOT_RUN_FAILED_ACQUISITION"
    assert all(row["media"]["frames_are_actual_states"] for row in report["fixtures"])
    assert report["publication_schema"] == "source-family-training-compact-v2"
    for row in report["fixtures"]:
        assert "observations" not in row and "captured_metrics" not in row
        assert row["observation_count"] == row["media"]["media"]["frames"]
        assert row["observation_count"] == len(row["media"]["actual_step_indices"])
        assert row["curve_archive"]["path"].endswith("captured-metrics.json")
        assert row["binding_ref"] in report["source_bindings"]
        assert row["executed_audit_source_ref"] in report["executed_audit_sources"]
    forbid_inline_streams(report)


def test_compact_publication_keeps_real_step_indices_and_external_curve_identity():
    metrics = [dict(step=step, event="check", angle=0.,
                    original=dict(heldout_rmse=.01, points=[[0., 0.]]),
                    correspondence=dict(relative_mse=.001, passed=True)) for step in (100, 200)]
    record = dict(observations=metrics, initial=metrics[0], final=metrics[-1], best=metrics[0],
                  captured_metrics=dict(path="captured-metrics.json", sha256="abc"),
                  media=dict(raw_observations=dict(path="observations.npz", sha256="def")))
    compact_observations(record, metrics)
    assert "observations" not in record and "initial" not in record
    assert "points" not in record["final"]["original"]
    assert record["media"]["actual_step_indices"] == [100, 200]
    assert record["observation_count"] == 2
    assert record["curve_archive"] == {"path": "captured-metrics.json", "sha256": "abc"}
    assert "points" in metrics[-1]["original"]  # the raw external stream is unchanged
    forbid_inline_streams(record)
    with pytest.raises(ValueError, match="must remain in the external archive"):
        forbid_inline_streams({"nested": {"diagnostic": metrics}})

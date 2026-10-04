"""Actual routed-owner observation controls, with no convergence claim."""
from pathlib import Path

from experiments.forge.policy_adapters import DIGEST_KIND, controls_receipt, typed_state_digest
from experiments.forge.routed_policy_adapters import UnusedTokenRoutedFixture
from experiments.forge.routed_policy_contracts import COHORT, SHARED_OVERRIDES, make_unused_variant


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    task = make_unused_variant(ROOT)
    request = {"candidate": {"task_cohort": COHORT, "recipe_preset": "atlas",
                             "recipe_overrides": dict(SHARED_OVERRIDES)},
               "protocol": {"seed": 0}}
    return UnusedTokenRoutedFixture(request, task, device="cpu")


def test_receipt_reads_real_full_forward_and_row_owners_without_state_change():
    run = fixture()
    before = typed_state_digest(run.state_dict())
    receipt = controls_receipt(run.policy, 0)
    owner = receipt["routed_owner"]
    assert owner["owner"] == "particlegan.routing.RoutedRowControl"
    assert owner["model_forward"] == {
        "callable": True, "module": "experiments.forge.routed_policy_adapters",
        "qualname": "complete_slot_forward"}
    assert owner["config"]["sites"] == ["slot_lookup"]
    assert owner["config"]["row_buffers"] == ["slot_ids"]
    assert owner["config"]["model_forward"] is True
    assert owner["config"]["routed_geometry"] == "mass_atoms_v1"
    assert owner["config"]["output_error_guard"] is True
    assert owner["config"]["max_context_harm"] == 0
    assert owner["config"]["max_output_context_harm"] == 0
    assert owner["table_matches_policy"] is owner["averaged_table_matches_policy"] is True
    assert owner["table_shape"] == [2, 2]
    assert owner["row_ownership"]["table"]["parameter"] is True
    assert type(owner["row_ownership"]["table"]["optimizer"]) is int
    assert owner["row_ownership"]["router.slot_ids"]["optimizer"] is None
    assert owner["state_digest_kind"] == DIGEST_KIND
    assert owner["state_sha256"] == typed_state_digest(run.policy.routed_control.state_dict())
    assert owner["fit_fill"] == owner["guard_fill"] == 0
    assert typed_state_digest(run.state_dict()) == before


def test_actual_protected_pool_and_clock_receipts_follow_observed_updates():
    run = fixture()
    run.step()
    run.step()
    before = typed_state_digest(run.state_dict())
    receipt = controls_receipt(run.policy, 2)
    owner = receipt["routed_owner"]
    routed = run.policy.routed_control
    assert owner["fit_fill"] == routed.fit_fill > 0
    assert owner["guard_fill"] == routed.guard_fill > 0
    assert owner["probe_clock"] == routed.probe_clock
    assert owner["probe_clock"]["observed_updates"] == 2
    assert owner["counters"] == routed.counters
    assert receipt["row_evidence_observations"] == 2
    assert receipt["lifecycle"]["complete"] is True
    assert typed_state_digest(run.state_dict()) == before

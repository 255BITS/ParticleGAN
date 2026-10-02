"""Software-only contracts for the isolated guided campaign adapter."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest
import torch

from examples import e22_routed_convergence_guided_campaign as campaign


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(previous)


def software_contract():
    """Current-byte, unauthorized fixture; not an archived qualification card."""
    card = deepcopy(json.loads(campaign.held_runner.CARD.read_text()))
    card.update(task_id=campaign.factory.TASK,
                parent_rotated_card_sha256=campaign.held_runner.common.sha(campaign.held_runner.CARD),
                orchestration_adapter=campaign.orchestration_manifest(),
                guided_pair={"guidance": 3., "unconditional_source": "fixed mean of the unchanged six parent source vectors"})
    card["sources"] = {str(path.relative_to(campaign.ROOT)): campaign.held_runner.common.sha(path)
                       for path in campaign.SOURCES if path != campaign.CARD}
    card["native_python_source_digest"] = campaign.held_runner.common.native_source_hash()
    card["execution"]["execution_authorized"] = False
    return card


def test_namespaces_reuse_exact_code_objects_without_mutating_held_globals():
    before = {id(module): dict(vars(module)) for module in (campaign.held_runner, campaign.held_reviewer)}
    runner, reviewer = campaign.adapters()
    for module in (campaign.held_runner, campaign.held_reviewer):
        assert set(vars(module)) == set(before[id(module)])
        assert all(vars(module)[name] is value for name, value in before[id(module)].items())
    assert runner.main.__code__ is campaign.held_runner.main.__code__
    assert reviewer.review_run.__code__ is campaign.held_reviewer.review_run.__code__
    assert reviewer.check_state.__code__ is campaign.held_reviewer.check_state.__code__
    assert runner.law.TASK == campaign.factory.TASK
    assert campaign.held_runner.law.TASK != campaign.factory.TASK
    assert runner.base.update is campaign.held_runner.base.update
    assert runner.base.restore is campaign.held_runner.base.restore
    assert runner.base.reachability_witness is campaign.factory.reachability_witness
    assert reviewer.runner is runner
    # Decorator closures must also point at local globals.
    assert inspect.unwrap(reviewer.score_residual).__globals__["base"] is runner.base


def test_complete_source_contract_and_fixed_denominators():
    archived_bytes = campaign.CARD.read_bytes()
    card = software_contract()
    assert card["execution"]["execution_authorized"] is False
    campaign.validate_contract(card)
    runner, reviewer = campaign.adapters()
    assert len(runner.checkpoint_steps()) * len(runner.law.ARMS) == 105
    assert len(runner.curve_steps()) * len(runner.law.ARMS) == 102
    assert len(reviewer.JUDGES) == 4
    assert runner.ENDPOINTS == (5120, 6400)
    assert Path(campaign.held_reviewer.__file__) in campaign.SOURCES
    assert campaign.CARD.read_bytes() == archived_bytes


@pytest.mark.parametrize("change", ["task", "source", "native", "missing_reviewer", "namespace", "guidance", "budget"])
def test_preflight_rejects_source_or_law_drift(change):
    card = software_contract()
    campaign.validate_contract(card)
    if change == "task":
        card["task_id"] = campaign.held_runner.law.TASK
    elif change == "source":
        card["sources"]["examples/e22_routed_convergence_guided_campaign.py"] = "0" * 64
    elif change == "native":
        card["native_python_source_digest"] = "0" * 64
    elif change == "missing_reviewer":
        card["sources"].pop("examples/review_e22_routed_convergence_guided_pair.py")
    elif change == "namespace":
        card["orchestration_adapter"]["held_module_globals_mutated"] = True
    elif change == "guidance":
        card["guided_pair"]["guidance"] = 1.
    else:
        card["execution"]["total_wall_budget_seconds"] = 2701
    with pytest.raises(ValueError):
        campaign.validate_contract(card)


def test_new_task_envelope_preserves_observations_and_rejects_old_task():
    value = {"schema": campaign.HELD_EXECUTION_SCHEMA, "task": campaign.factory.TASK,
             "scores": {"wrong_direction": -3., "positive": 4.}}
    original = deepcopy(value)
    result = campaign.output_envelope(value)
    assert result["schema"] == campaign.EXECUTION_SCHEMA
    assert result["task"] == campaign.factory.TASK
    assert result["scores"] == original["scores"]
    assert value == original
    with pytest.raises(ValueError, match="old task identity"):
        campaign.output_envelope({"schema": campaign.HELD_EXECUTION_SCHEMA, "task": campaign.held_runner.law.TASK})


def test_reviewer_requires_actual_new_schema_and_sha_bound_completion(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    value = {"schema": campaign.EXECUTION_SCHEMA, "task": campaign.factory.TASK,
             "orchestration_adapter": campaign.orchestration_manifest(), "contract": software_contract(), "wall_seconds": 10.}
    receipt_path.write_text(json.dumps(value))
    (tmp_path / "compact-report.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        campaign.reviewer_read(receipt_path)
    completion = {"schema": campaign.COMPLETION_SCHEMA, "task": campaign.factory.TASK, "complete": True, "wall_seconds": 11.,
                  "receipt_sha256": campaign.held_runner.common.sha(receipt_path),
                  "compact_sha256": campaign.held_runner.common.sha(tmp_path / "compact-report.json")}
    (tmp_path / "execution-completion.json").write_text(json.dumps(completion))
    normalized = campaign.reviewer_read(receipt_path)
    assert normalized["schema"] == campaign.HELD_EXECUTION_SCHEMA
    assert normalized["task"] == campaign.factory.TASK
    assert normalized["wall_seconds"] == 11.
    assert json.loads(receipt_path.read_text())["schema"] == campaign.EXECUTION_SCHEMA
    completion["receipt_sha256"] = "0" * 64
    (tmp_path / "execution-completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="bounded-write receipt"):
        campaign.reviewer_read(receipt_path)


def test_factory_binding_is_frozen_and_checkpoint_law_carries_new_manifest(monkeypatch):
    runner, _ = campaign.adapters()
    data = campaign.factory.make_guided_data()
    loop = runner.law.make_rotated_loop(campaign.factory.ARMS[1], data)
    assert loop.law["task"] == campaign.factory.TASK
    assert loop.law["campaign_adapter"] == campaign.orchestration_manifest()
    assert loop.data["guided_pair"]["factory_adapter"] == campaign.factory.factory_binding_manifest()
    original = campaign.factory.factory_binding_manifest
    monkeypatch.setattr(campaign.factory, "factory_binding_manifest", lambda: {**original(), "held_globals_mutated": True})
    with pytest.raises(ValueError, match="changed after namespace creation"):
        runner.law.make_rotated_loop(campaign.factory.ARMS[1], data)


def test_cfg_frozen_owners_cover_fast_and_ema_and_detect_mutation():
    runner, reviewer = campaign.adapters()
    data = campaign.factory.make_guided_data()
    loop = runner.law.make_rotated_loop(campaign.factory.ARMS[1], data)
    frozen = campaign.frozen_values(loop)
    for role in ("generator", "average_generator"):
        for name in campaign.CFG_BUFFERS:
            assert role + "." + name in frozen
    state = runner.base.checkpoint(loop)
    owners = campaign.frozen_owners(state)
    for family in ("models", "averages"):
        for name in campaign.CFG_BUFFERS:
            assert family + "/generator/" + name in owners
    changed = deepcopy(state)
    changed["training"]["averages"]["generator"]["cfg_unconditional_source"][0] += 1.
    assert runner.base.digest(campaign.frozen_owners(changed)) != runner.base.digest(owners)
    assert reviewer.frozen_owners is campaign.frozen_owners


def test_actual_factory_callback_drift_is_rejected(monkeypatch):
    runner, _ = campaign.adapters()
    data = campaign.factory.make_guided_data()
    monkeypatch.setattr(campaign.factory, "make_guided_loop", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="changed after namespace creation"):
        runner.law.make_rotated_loop(campaign.factory.ARMS[1], data)


def test_native_per_context_guard_and_noisy_two_half_controls():
    runner, _ = campaign.adapters()
    data = campaign.factory.make_guided_data()
    loop = runner.law.make_rotated_loop(campaign.factory.ARMS[1], data)
    spec = loop.policy.routed_control.spec
    assert spec.max_context_harm == 0
    assert spec.output_error_guard is False
    assert spec.sites == campaign.factory.SITES
    # Native code dimensions remain the per-half code dimension; the extra
    # half axis is supplied by the public whole-model callback.
    assert loop.policy.table.shape == (128, 4)
    assert loop.policy.recipe.row_policy == "routed_paired"
    assert loop.policy.recipe.row_evidence_gate


def test_matched_native_software_draws_and_checkpoint_restore():
    runner, _ = campaign.adapters()
    data = campaign.factory.make_guided_data()
    loops = [runner.law.make_rotated_loop(arm, data) for arm in campaign.factory.ARMS]
    rows = [runner.base.update(loop) for loop in loops]  # exactly three software updates, no quality campaign
    keys = ("batch_indices", "paired_base_digest", "data_rng", "paired_rng")
    assert all({key: row[key] for key in keys} == {key: rows[0][key] for key in keys} for row in rows)
    for arm, loop in zip(campaign.factory.ARMS, loops):
        state = runner.base.checkpoint(loop)
        with torch.random.fork_rng(devices=[]):
            restored = runner.law.make_rotated_loop(arm, data)
            runner.base.restore(restored, state)
            assert runner.base.digest(runner.base.checkpoint(restored)) == runner.base.digest(state)


def test_zero_code_scoring_is_immutable_and_game_gates_are_denominator_safe():
    runner, _ = campaign.adapters()
    data = campaign.factory.make_guided_data()
    loop = runner.law.make_rotated_loop(campaign.factory.ARMS[1], data)
    panels = runner.base.evaluation_panels(data)
    judge = runner.base.ConditionalCritic(data["scale"]).eval().requires_grad_(False)
    before = runner.base.digest(runner.base.checkpoint(loop))
    runner.immutable_scores(loop, {"software_judge": judge}, panels, code_ablation=True)
    assert runner.base.digest(runner.base.checkpoint(loop)) == before
    scores = {f"{arm}@6400": {name: {"test": {"paired_game": value}} for name in runner.JUDGES}
              for arm, value in zip(campaign.factory.ARMS, (1., .9, .8))}
    witness = {"bridge_still_trainable": True, "bank_still_trainable": True, "router_still_trainable": True,
               "C_norms": {name: 1. for name in campaign.factory.SITES}, "live_bank_updates": 1,
               "live_query_updates": 1, "zero_code_minus_live_test_game": dict.fromkeys(runner.JUDGES, .1)}
    gates = runner.final_gates(scores, dict.fromkeys(campaign.factory.ARMS[1:], witness))
    assert gates["endpoint_gap_reduction"] is None
    assert not gates["support_gate_applicable"]
    assert not gates["remaining_neutral_gap_witness"]
    assert gates["neutral_beats_ordinary_all_four"]


def test_budget_checked_after_io_and_overrun_is_an_error(monkeypatch):
    monkeypatch.setattr(campaign.time, "monotonic", lambda: 13.)
    assert campaign.require_budget(10., 3.) == 3.
    with pytest.raises(TimeoutError, match="serialization"):
        campaign.require_budget(10., 2.999)

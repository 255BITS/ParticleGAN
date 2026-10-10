"""Whole-campaign pinning guards; synthetic certified-row inputs, no training."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pacing_measurement_selection", ROOT / "reports/forge/dualnorm-pacing-v2/select_measurements.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    policy = {"id": "discriminator_stability", "revision": 4, "assignments": []}
    fingerprint = stable_hash(policy)
    protocol = {"id": "screening", "seed": 0}
    protocol_hash = stable_hash(protocol)
    reports, trials, stages, rows = [], [], [], []
    contracts = {name: {"execution": {"task": name, "steps": 80}, "evaluation": {"kind": "unit_fixture"}}
                 for name in (*selection.REQUIRED, selection.CLOCK)}
    contract_ids = {name: stable_hash(contract) for name, contract in contracts.items()}
    for stage, count in zip("ABC", (20, 3, 2)):
        spec = {"id": selection.CAMPAIGN + "-" + stage.lower(), "stage": stage}
        group = []
        for _ in range(count):
            index = len(trials)
            configuration = selection.CONTROL.split("--")[1] if index == 0 else f"{index:064x}"
            name = "bcap-dualnorm--" + configuration
            passed = 3 if index == 0 else 4 if index == 20 else 2
            recipe = {"optimizer_family": "dualnorm", "optimizer_momentum": 0. if index < 23 else .5,
                      "lr": .012 if index >= 20 else .01, "d_lr_mult": 1.5,
                      "prior_lr_mult": 2.5 if index >= 20 else 3.}
            tasks = [{"task": task, "qualification_tier": 1, "importance": "required",
                      "gate_status": "PASS" if i < passed else "FAIL", "attempt_id": f"attempt-{index}-{i}",
                      "compatibility_key": f"key-{index}-{i}"} for i, task in enumerate(selection.REQUIRED)]
            tasks.append({"task": selection.CLOCK, "qualification_tier": 1, "importance": "diagnostic",
                          "gate_status": "PASS", "attempt_id": f"attempt-{index}-clock", "compatibility_key": f"key-{index}-clock"})
            # Real search bindings cover all 28 tasks; the compact campaign
            # projection retains only its seven admitted Tier 1 cells.
            attempts = [task["attempt_id"] for task in tasks]
            tasks.extend({"task": f"deferred-{i}", "qualification_tier": 2, "importance": "required",
                          "gate_status": "UNKNOWN", "attempt_id": None, "compatibility_key": f"key-{index}-deferred-{i}"}
                         for i in range(21))
            trial = {"candidate_id": name, "configuration_id": configuration, "trainer_family": selection.FAMILY,
                "candidate_revision": f"revision-{index}", "resolved_recipe": recipe, "tasks": tasks,
                "source_digest": selection.SOURCE_DIGEST, "runtime_cohort": {"execution_backend": "cuda"},
                "protocol_hash": protocol_hash, "policy_fingerprint": fingerprint, "submission_status": "completed",
                "request_id": f"request-{index}", "attempt_ids": attempts}
            trials.append(trial)
            group.append(trial)
            rows.append({"candidate_id": name, "candidate_revision": trial["candidate_revision"],
                "cohort": f"cohort-{index}", "runtime_cohort": trial["runtime_cohort"], "attempt_ids": trial["attempt_ids"],
                "bindings": {"source_digest": selection.SOURCE_DIGEST, "source_origin_commit": selection.SOURCE_COMMIT,
                    "recipe_sha256": stable_hash(recipe), "protocol_sha256": protocol_hash, "rng_sha256": "rng",
                    "initializer": "public-fixture", "prior": {"kind": "mog", "sigma": .025}, "task_contracts": contract_ids,
                    "task_keys_sha256": stable_hash({task["task"]: task["compatibility_key"] for task in tasks})},
                "tasks": [{"task_id": task["task"], "status": task["gate_status"]} for task in tasks if task["importance"] == "required"],
                "nonrequired_tasks": [{"task_id": selection.CLOCK, "status": "PASS"}]})
        reports.append({"schema_version": 1, "study_id": spec["id"], "spec": spec, "spec_hash": stable_hash(spec),
            "trainer_family": selection.FAMILY, "campaign": {"id": selection.CAMPAIGN},
            "view": "discriminator_stability", "tuning_through_tier": 1, "execution_backend": "cuda",
            "source_digests": [selection.SOURCE_DIGEST], "policy_fingerprint": fingerprint, "trials": group,
            "selection": select_configuration(group, 1)})
        stages.append({"stage": stage, "admitted": True, "execution_complete": True, "spec": spec})
    result = {"status": "completed", "source": {"commit": selection.SOURCE_COMMIT, "digest": selection.SOURCE_DIGEST},
        "admitted_recipes": 25, "running_workers": 0, "all_attempts_supervised_finished": True,
        "automatic_retries": 0, "default_adoption": False, "accounting": {"reserved_seconds": 0},
        "stages": stages, "trials": trials, "selection": select_configuration(trials, 1),
        "control_candidate_id": selection.CONTROL, "control_required_passes": 3}
    snapshot = {"publication_scope": "frozen_source", "policy_fingerprint": fingerprint,
        "frozen_source": {"commit": selection.SOURCE_COMMIT, "source_digests": [selection.SOURCE_DIGEST]},
        "rows": rows, "provenance": {}, "protocol_contracts": {protocol_hash: protocol},
        "task_contracts": {stable_hash(contract): contract for contract in contracts.values()}}
    card = {"schema_version": 1, "view": "discriminator_stability", "policy_fingerprint": fingerprint,
        "default_adoption": False, "selections": [{"trainer_family": "bcap-pure", "candidate_id": "adam-historical"},
            {"trainer_family": selection.FAMILY, "candidate_id": "original-starter", "scientific_row_sha256": "original"}],
        "historical_selections": [{"trainer_family": "atlas", "candidate_id": "exact-archive"}]}
    atomic_json(tmp_path / CURRENT_SELECTION, card)
    validated = []
    monkeypatch.setattr(selection, "_validate_published_row", lambda root, snapshot, row: validated.append(row["candidate_id"]))
    monkeypatch.setattr(selection, "_current_pin", lambda *args, **kwargs: None)
    monkeypatch.setattr(selection, "load_view", lambda *args: policy)
    monkeypatch.setattr(selection, "family_for_candidate", lambda *args: {"id": selection.FAMILY})
    def flush():
        for report in reports:
            report["input_digest"] = stable_hash({key: value for key, value in report.items() if key != "input_digest"})
            atomic_json(tmp_path / "reports/forge/configuration-search" / (report["study_id"] + ".json"), report)
        result["trials"] = [deepcopy(trial) | {"tasks": deepcopy([task for task in trial["tasks"] if task["qualification_tier"] == 1])}
                            for trial in trials]
        atomic_json(tmp_path / selection.REPORT / "results.json", result)
        snapshot["provenance"].pop("input_digest", None)
        snapshot["provenance"]["input_digest"] = stable_hash(snapshot)
    flush()
    return tmp_path, reports, result, snapshot, card, flush, validated


def test_whole_winner_changes_only_dualnorm_measurement_and_preserves_every_old_pin(completed):
    root, reports, result, snapshot, card, _, validated = completed
    original = deepcopy((reports, result, snapshot, card))
    proposed, receipt = selection.proposed_selection(root, snapshot)
    assert proposed["selections"][0] == card["selections"][0]
    assert proposed["historical_selections"] == card["historical_selections"]
    assert proposed["selections"][1]["candidate_id"] == result["selection"]["selected_candidate_id"]
    assert proposed["selections"][1]["selection_kind"] == "current_measurement"
    assert proposed["selections"][1]["measurement_views"] == ["discriminator_stability"]
    assert receipt["required_passes"] == 4 and receipt["control_required_passes"] == 3
    assert receipt["qualified"] is receipt["default_adoption"] is False
    assert receipt["previous_selection"] == card["selections"][1]
    assert len(set(validated)) == 25  # Every rival's independent row is checked.
    assert (reports, result, snapshot, card) == original
    assert read_json(root / CURRENT_SELECTION) == card
    assert receipt["input_digest"] == stable_hash({key: value for key, value in receipt.items() if key != "input_digest"})


def test_manual_tied_candidate_cannot_replace_the_frozen_hash_winner(completed):
    root, reports, result, snapshot, _, flush, _ = completed
    winner, tied = reports[1]["trials"][:2]
    for i in (2, 3):
        tied["tasks"][i]["gate_status"] = "PASS"
        snapshot["rows"][21]["tasks"][i]["status"] = "PASS"
    reports[1]["selection"] = select_configuration(reports[1]["trials"], 1)
    result["selection"] = select_configuration(result["trials"], 1)
    result["selection"].update(selected_candidate_id=tied["candidate_id"], selected_configuration_id=tied["configuration_id"])
    flush()
    with pytest.raises(ValueError, match="PASS/hash objective"):
        selection.proposed_selection(root, snapshot)


@pytest.mark.parametrize("status", ["UNKNOWN", "INCOMPLETE", "INVALID", "BLOCKED"])
def test_missing_or_invalid_losing_peer_prevents_pinning(completed, status):
    root, reports, _, snapshot, _, flush, _ = completed
    reports[0]["trials"][19]["tasks"][6]["gate_status"] = status
    flush()
    with pytest.raises(ValueError, match="complete every actual"):
        selection.proposed_selection(root, snapshot)


@pytest.mark.parametrize("field", ["source_digest", "runtime_cohort", "protocol_hash", "policy_fingerprint"])
def test_mixed_scientific_cohort_cannot_win(completed, field):
    root, reports, _, snapshot, _, flush, _ = completed
    reports[0]["trials"][19][field] = {"execution_backend": "cpu"} if field == "runtime_cohort" else "different"
    flush()
    with pytest.raises(ValueError, match="one frozen"):
        selection.proposed_selection(root, snapshot)


@pytest.mark.parametrize("field", ["candidate_revision", "recipe_sha256", "task_keys_sha256", "source_origin_commit", "task_status"])
def test_independent_snapshot_must_agree_for_losing_rivals_too(completed, field):
    root, _, _, snapshot, _, flush, _ = completed
    row = snapshot["rows"][19]
    if field == "candidate_revision":
        row[field] = "different"
    elif field == "task_status":
        row["tasks"][0]["status"] = "FAIL"
    else:
        row["bindings"][field] = "different"
    flush()
    with pytest.raises(ValueError, match="independently regraded row differs"):
        selection.proposed_selection(root, snapshot)


def test_a_tie_with_current_control_is_not_an_improvement(completed):
    root, reports, result, snapshot, _, flush, _ = completed
    reports[0]["trials"][0]["tasks"][3]["gate_status"] = "PASS"
    snapshot["rows"][0]["tasks"][3]["status"] = "PASS"
    reports[0]["selection"] = select_configuration(reports[0]["trials"], 1)
    result["control_required_passes"] = 4
    result["selection"] = select_configuration([trial for report in reports for trial in report["trials"]], 1)
    flush()
    with pytest.raises(ValueError, match="strict whole-recipe improvement"):
        selection.proposed_selection(root, snapshot)


def test_stale_snapshot_digest_cannot_supply_measurement_credit(completed):
    root, _, _, snapshot, _, _, _ = completed
    snapshot["rows"][0]["candidate_revision"] = "tampered"
    with pytest.raises(ValueError, match="source snapshot input digest mismatch"):
        selection.proposed_selection(root, snapshot)


def test_changed_prior_condition_cannot_hide_behind_matching_source_and_recipe(completed):
    root, _, _, snapshot, _, flush, _ = completed
    snapshot["rows"][19]["bindings"]["prior"]["sigma"] = .1
    flush()
    with pytest.raises(ValueError, match="task, prior, initialization"):
        selection.proposed_selection(root, snapshot)


@pytest.mark.parametrize("replacement", [[], [{"task_id": selection.CLOCK, "status": "FAIL"}]])
def test_missing_or_changed_separate_clock_diagnostic_prevents_pinning(completed, replacement):
    root, _, _, snapshot, _, flush, _ = completed
    snapshot["rows"][19]["nonrequired_tasks"] = replacement
    flush()
    with pytest.raises(ValueError, match="independently regraded row differs"):
        selection.proposed_selection(root, snapshot)


def test_full_task_key_binding_is_required_without_claiming_unknown_higher_tiers(completed):
    root, reports, result, snapshot, _, flush, _ = completed
    assert len(reports[0]["trials"][0]["tasks"]) == 28
    assert len(result["trials"][0]["tasks"]) == 7
    proposed, _ = selection.proposed_selection(root, snapshot)
    assert proposed["selections"][1]["candidate_id"] == result["selection"]["selected_candidate_id"]
    trial = result["trials"][0]
    snapshot["rows"][0]["bindings"]["task_keys_sha256"] = stable_hash(
        {task["task"]: task["compatibility_key"] for task in trial["tasks"]})
    flush()
    with pytest.raises(ValueError, match="independently regraded row differs"):
        selection.proposed_selection(root, snapshot)

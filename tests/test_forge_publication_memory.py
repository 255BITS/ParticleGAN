"""Fresh-checkout recall controls over compact summaries, without training."""
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

import pytest

from experiments.forge import knowledge, publication_memory
from experiments.forge.contracts import atomic_json, file_hash, read_json


@pytest.fixture
def publication(tmp_path):
    path = tmp_path / "reports/forge/configuration-search/study.json"
    report = {"schema_version": 1, "study_id": "study", "view": "stability", "trainer_family": "ka2",
              "selection": {"all_trials_terminal": True, "selection_kind": "best_observed"},
              "default_adoption": False, "trials": [
                  {"candidate_id": "ka2--configuration", "candidate_revision": "revision", "configuration_id": "configuration",
                   "status": "FAIL", "declaration": {"goal": "stability", "mechanism_class": "floor_constant",
                                                          "hypothesis": "Transport preserves component mass"},
                   "source_digest": "frozen-source", "protocol_hash": "frozen-protocol",
                   "resolved_recipe": {"lr": .01}, "runtime_cohort": {"execution_backend": "cuda"},
                   "receipt_bindings": [{"attempt_id": "original", "result_hash": "result", "valid_receipt": True}],
                   "tasks": [{"task": "rare-mass", "gate_status": "FAIL", "metrics": {"min_mass_ratio": .598},
                              "reason": "recorded original failure", "cost": {"wall_seconds": 4.0}, "attempt_id": "original"},
                             {"task": "hold", "gate_status": "UNKNOWN"}]}]}
    atomic_json(path, report)
    return tmp_path, path, report


def test_publications_recall_exact_ids_mechanisms_metrics_and_goals(publication):
    root, path, _ = publication
    for query in ("ka2--configuration", "revision", "min_mass_ratio", "floor_constant", "rare-mass"):
        result = knowledge.recall(root, query, "stability")
        trial = next(row for row in result if row["candidate_id"] == "ka2--configuration")
        assert trial["qualification_input"] is False
        assert trial["source"]["sha256"] == file_hash(path)
        assert trial["source"]["board"] == "reports/forge/technique-inventory.md"
        assert "UNKNOWN" in trial["conclusion"]
    assert knowledge.recall(root, "ka2--configuration", "stability")[0]["exact_identity_match"]
    assert not knowledge.recall(root, "ka2--configuration", "another-goal")


def test_projection_preserves_source_bindings_without_loading_raw_artifacts(publication, monkeypatch):
    root, _, _ = publication
    monkeypatch.setattr(knowledge, "_attempts", lambda *args: pytest.fail("Recall must not hydrate/regrade attempts"))
    records = publication_memory.normalize(root)
    trial = next(record for record in records if record["record_type"] == "published_trial")
    assert trial["provenance"]["source_digest"] == "frozen-source"
    assert trial["provenance"]["receipt_bindings"][0]["result_hash"] == "result"
    assert trial["qualification_reuse"] is False
    assert trial["task_results"][0]["metrics"]["min_mass_ratio"] == .598
    assert knowledge._records(root) == ([], [])  # Qualification never sees projections.
    assert knowledge.recall(root, "configuration")


def test_generated_projection_is_never_recall_authority(publication):
    root, _, _ = publication
    atomic_json(root / publication_memory.OUTPUT, {"records": [{"candidate_id": "invented-winner"}]})
    assert not knowledge.recall(root, "invented-winner")
    assert knowledge.recall(root, "configuration")


def test_live_recall_sees_new_publication_before_compilation_and_marks_stale(publication):
    root, path, report = publication
    knowledge.compile_memory(root)
    assert knowledge.freshness(root)["fresh"]
    report = deepcopy(report)
    report["study_id"] = "second-study"
    report["trials"][0]["candidate_id"] = "ka2--second"
    new = path.with_name("second-study.json")
    atomic_json(new, report)
    assert knowledge.recall(root, "ka2--second")[0]["exact_identity_match"]
    status = knowledge.freshness(root)
    assert status["status"] == "STALE"
    assert str(new.relative_to(root)) in status["added_inputs"]
    assert not status["publication_materialization_current"]


@pytest.mark.parametrize("kind", ["changed", "removed", "projection", "memory", "views", "reducer"])
def test_freshness_detects_changed_and_missing_inputs_or_materializations(publication, kind):
    root, path, report = publication
    knowledge.compile_memory(root)
    assert knowledge.freshness(root)["fresh"]
    if kind == "changed":
        report["trials"][0]["tasks"][0]["metrics"]["min_mass_ratio"] = .4
        atomic_json(path, report)
    elif kind == "removed":
        path.unlink()
    elif kind == "projection":
        (root / publication_memory.OUTPUT).unlink()
    elif kind == "memory":
        (root / "reports/forge/EXPERIMENT_MEMORY.md").write_text("Stale generated memory")
    elif kind == "views":
        atomic_json(root / "configs/forge/views/new.json", {"id": "new"})
    else:
        manifest_path = root / "reports/forge/compilation.json"
        manifest = read_json(manifest_path)
        manifest["reducer_hashes"] = {}
        atomic_json(manifest_path, manifest)
    assert not knowledge.freshness(root)["fresh"]


def test_policy_terminal_gate_and_hold_unknowns_remain_distinct(tmp_path):
    board = {"schema": "policy_family_goal_inventory_v1", "goal": "policy-family-defaults", "cohorts": [
        {"id": "policy--source", "study_id": "policy", "source": {"commit": "frozen-commit"},
         "selection": {"attempts_concluded": True, "outcome": "incomplete_comparison"}, "trials": [
             {"id": "atlas--config", "family": "atlas", "status": "INCOMPLETE", "cases": [
                 {"id": "broad", "original_gate": "PASS", "study_gate": "INCOMPLETE",
                  "final_metrics": {"projection_ks": .04}, "original_failed_bounds": [],
                  "acquisition_hold": {"reason": "two of five hold observations", "status": "INCOMPLETE"},
                  "sampling": {"evaluation": "public selected/served"}},
                 {"id": "quality", "original_gate": None, "study_gate": None, "final_metrics": None,
                  "acquisition_hold": None}]}]}]}
    atomic_json(tmp_path / publication_memory.POLICY_BOARD, board)
    trial = next(record for record in publication_memory.normalize(tmp_path) if record["record_type"] == "published_trial")
    broad, quality = trial["task_results"]
    assert (broad["original_gate"], broad["gate_status"]) == ("PASS", "INCOMPLETE")
    assert quality["gate_status"] == "UNKNOWN" and quality["metrics"] == {}
    assert trial["candidate_revision"] is None  # A source commit is not a candidate revision.
    assert trial["provenance"]["source_commit"] == "frozen-commit"


def test_in_progress_study_is_not_called_concluded(publication):
    root, path, report = publication
    report["selection"]["all_trials_terminal"] = False
    atomic_json(path, report)
    assert not publication_memory.normalize(root)
    knowledge.compile_memory(root)
    assert str(path.relative_to(root)) in read_json(root / "reports/forge/compilation.json")["input_hashes"]


def test_compile_check_cli_is_read_only_and_returns_stale_exit_status(publication, capsys):
    from experiments.forge.__main__ import main
    root, _, _ = publication
    assert main(["--root", str(root), "compile", "--check"]) == 1
    assert not (root / "runs").exists()
    assert not (root / "reports/forge/compilation.json").exists()
    knowledge.compile_memory(root)
    assert main(["--root", str(root), "compile", "--check"]) == 0
    assert '"status": "CURRENT"' in capsys.readouterr().out


def test_catalog_and_unrelated_engineering_changes_do_not_invalidate_recall(publication):
    root, _, _ = publication
    knowledge.compile_memory(root)
    atomic_json(root / "configs/forge/catalog.json", {"files": [], "new_bookkeeping": True})
    (root / "experiments/forge").mkdir(parents=True)
    (root / "experiments/forge/unrelated.py").write_text("# A new engineering feature\n")
    assert knowledge.freshness(root)["fresh"]


def test_summary_refresh_preserves_science_and_costs_when_originals_are_unavailable(publication, monkeypatch):
    root, _, _ = publication
    atomic_json(root / "configs/forge/views/frozen.json", {"id": "frozen"})
    board = root / "reports/forge/leaderboards/frozen.md"
    board.parent.mkdir(parents=True)
    board.write_text("Original qualified result and 24-task denominator\n")
    automation = root / "reports/forge/automation.json"
    original_automation = {"attempt_outcomes": {"completed": 70}, "cost_to_qualify": {"outcomes": 1}}
    atomic_json(automation, original_automation)
    atomic_json(root / "reports/forge/compilation.json",
                {"input_digest": "qualified-source-manifest", "pending_readout": ["unresolved"],
                 "scientific_source_digests": ["original-source"], "operational_lifecycle_digest": "original-lifecycle"})
    monkeypatch.setattr(knowledge, "board", lambda *args: pytest.fail("Cannot replay missing qualified originals"))
    from experiments.forge import telemetry
    monkeypatch.setattr(telemetry, "summarize_automation", lambda *args: pytest.fail("Cannot recompute archived cost ledger"))
    first = knowledge.compile_memory(root, summaries_only=True)
    assert board.read_text() == "Original qualified result and 24-task denominator\n"
    assert read_json(automation) == original_automation
    manifest = read_json(root / "reports/forge/compilation.json")
    assert manifest["scientific_source_digests"] == ["original-source"]
    assert manifest["pending_readout"] == ["unresolved"]
    assert manifest["operational_lifecycle_digest"] == "original-lifecycle"
    assert manifest["qualification_refresh"]["source_manifest_digest"] == "qualified-source-manifest"
    assert knowledge.freshness(root)["fresh"]
    assert knowledge.compile_memory(root, summaries_only=True) == first


def test_summary_refresh_missing_view_is_a_pointer_without_invented_gate_results(publication):
    root, _, _ = publication
    atomic_json(root / "configs/forge/views/new.json", {"id": "new"})
    knowledge.compile_memory(root, summaries_only=True)
    assert "no previously published goal table" in (root / "reports/forge/leaderboards/new.md").read_text()
    assert not (root / "reports/forge/leaderboards/new.json").exists()


def test_current_published_campaign_has_complete_discoverable_trial_coverage():
    root = Path(__file__).resolve().parents[1]
    records = publication_memory.normalize(root)
    trials = [record for record in records if record["record_type"] == "published_trial"]
    # The same card can occur in several frozen studies. Check complete source
    # cohorts from authoritative publications instead of a growing global count.
    expected, studies = {}, {}
    for path in sorted((root / "reports/forge/configuration-search").glob("*.json")):
        report = read_json(path)
        if report["selection"].get("all_trials_terminal") is not True:
            continue
        relative = path.relative_to(root).as_posix()
        studies[relative, report["study_id"], None] = [trial["candidate_id"] for trial in report["trials"]]
        for trial in report["trials"]:
            key = relative, report["study_id"], trial["candidate_id"], trial["candidate_revision"], None
            assert key not in expected
            expected[key] = {field: trial.get(field) for field in ("source_digest", "protocol_hash", "runtime_cohort")}
    policy = read_json(root / publication_memory.POLICY_BOARD)
    for cohort in policy["cohorts"]:
        if cohort["selection"].get("attempts_concluded") is not True:
            continue
        studies[publication_memory.POLICY_BOARD, cohort["study_id"], cohort["id"]] = [
            trial["id"] for trial in cohort["trials"]]
        for trial in cohort["trials"]:
            key = publication_memory.POLICY_BOARD, cohort["study_id"], trial["id"], None, cohort["id"]
            assert key not in expected
            expected[key] = {"source_commit": cohort.get("source", {}).get("commit"),
                             "spec_sha256": cohort.get("spec_sha256"),
                             "runtime_contract": cohort.get("runtime_contract")}

    def identity(record):
        return (record["source"]["path"], record["study_id"], record["candidate_id"],
                record.get("candidate_revision"), record.get("cohort"))

    assert len(trials) == len(expected) and {identity(trial) for trial in trials} == set(expected)
    actual_studies = {(record["source"]["path"], record["study_id"], record.get("cohort")): record["trial_ids"]
                      for record in records if record["record_type"] == "published_study"
                      and record["source"]["path"] != publication_memory.COMPLETION}
    assert actual_studies == studies
    groups = defaultdict(list)
    for trial in trials:
        assert trial["source"]["sha256"] == file_hash(root / trial["source"]["path"])
        assert {field: trial["provenance"].get(field) for field in expected[identity(trial)]} == expected[identity(trial)]
        groups[trial["candidate_id"], trial["goal"]].append(trial)
    for (candidate_id, goal), cohort_trials in groups.items():
        matches = {record["record_id"]: record for record in knowledge.recall(root, candidate_id, goal)}
        for trial in cohort_trials:
            match = matches[trial["record_id"]]
            assert match["candidate_id"] == trial["candidate_id"]
            assert match["candidate_revision"] == trial["candidate_revision"]
            assert match["study_id"] == trial["study_id"] and match["source"] == trial["source"]
            assert match["exact_identity_match"] and match["qualification_input"] is False

    refresh = read_json(root / "configs/forge/rounds/tier1-existing-configs-v1.json")
    refreshed = [trial for trial in trials if trial["study_id"] in refresh["studies"]]
    assert len(refreshed) == len(refresh["configuration_ids"])
    assert {trial["candidate_id"] for trial in refreshed} == set(refresh["configuration_ids"])
    assert {trial["study_id"] for trial in refreshed} == set(refresh["studies"])
    exact_id = "ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f"
    original = read_json(root / "reports/forge/configuration-search/ka2-family-defaults-round1-v1.json")
    original_trial = next(trial for trial in original["trials"] if trial["candidate_id"] == exact_id)
    assert original_trial["source_digest"] == "9f89fa7cc7af552ef7e41405fab415abbd00acf3d3c6e4af8e4435fb2423142e"
    exact = next(record for record in knowledge.recall(root, exact_id, "discriminator_stability")
                 if record["study_id"] == original["study_id"]
                 and record["candidate_revision"] == original_trial["candidate_revision"]
                 and record["source"]["path"] == "reports/forge/configuration-search/ka2-family-defaults-round1-v1.json")
    assert exact["candidate_id"] == exact_id and exact["exact_identity_match"]
    failed = next(task for task in exact["task_results"] if task["task_id"] == "vector_unequal_mass")
    assert failed["gate_status"] == "FAIL"
    assert failed["metrics"]["min_mass_ratio"] == pytest.approx(.59814453125)
    assert any(row["candidate_id"] == exact_id for row in knowledge.recall(root, "min_mass_ratio", "discriminator_stability"))
    assert any(row["study_id"] == "policy-family-defaults-round4-prior-balance-v1"
               for row in knowledge.recall(root, "prior-balance", "policy-family-defaults"))


def test_committed_research_memory_is_fresh_without_raw_log_hydration():
    root = Path(__file__).resolve().parents[1]
    status = knowledge.freshness(root)
    assert status["fresh"], status
    assert status["view_count"] == len(list((root / "configs/forge/views").glob("*.json")))
    assert status["publication_record_count"] == len(publication_memory.normalize(root))

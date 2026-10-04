"""Display exports consume completed compact science without raw hydration."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.views import view_fingerprint


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tier1_refresh_exports", ROOT / "reports/forge/tier1-refresh/regenerate.py")
exports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exports)


@pytest.fixture
def completed(tmp_path):
    round_definition = read_json(ROOT / exports.ROUND)
    relatives = [exports.ROUND, round_definition["campaign"], "configs/forge/legacy-ideas-v1.json",
                 f"configs/forge/views/{round_definition['view']}.json"]
    relatives += [f"configs/forge/{directory}/{name}.json" for directory, key in
                  (("ideas", "idea_ids"), ("configurations", "configuration_ids"))
                  for name in round_definition[key]]
    relatives += [f"configs/forge/searches/{study}.json" for study in round_definition["studies"]]
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    # The completed study is pinned to revision 3, independent of later tasks.
    shutil.copyfile(ROOT / "configs/forge/view-history/discriminator_stability-v3.json",
                    tmp_path / f"configs/forge/views/{round_definition['view']}.json")
    view = read_json(tmp_path / f"configs/forge/views/{round_definition['view']}.json")
    assignments = sorted(view["assignments"], key=lambda row: (row["qualification_tier"], row["order"]))
    requirements = {str(tier): [row["task"] for row in assignments if row["qualification_tier"] == tier]
                    for tier in (1, 2, 3)}
    fingerprint, source = view_fingerprint(view), "a" * 64
    runtime = {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "NVIDIA RTX A6000"}}}
    campaign = read_json(tmp_path / round_definition["campaign"])
    cards = {name: read_json(tmp_path / f"configs/forge/configurations/{name}.json")
             for name in round_definition["configuration_ids"]}

    def tasks():
        return [{"task_id": row["task"], "status": "PASS" if row["qualification_tier"] == 1 and row["order"] < 3 else
                 "FAIL" if row["task"] == "ring16_acquisition" else "UNKNOWN",
                 "gate_status": "PASS" if row["qualification_tier"] == 1 and row["order"] < 3 else
                 "FAIL" if row["task"] == "ring16_acquisition" else "NOT_RUN"} for row in assignments]

    rows = [{"candidate_id": name, "candidate_revision": "revision-" + name,
             "trainer_family": cards[name]["trainer_family"] if name in cards else name,
             "runtime_cohort": deepcopy(runtime), "bindings": {"source_digest": source,
                 "recipe_sha256": stable_hash(cards[name]["resolved_configuration_recipe"]) if name in cards else "unused"},
             "tasks": tasks(), "cost": {"wall_seconds": 1.}, "attempt_ids": [],
             "tiers": {"1": {"passed": 3, "total": 5}, "2": {"passed": 0, "total": 19}, "3": {"passed": 0, "total": 2}}}
            for name in round_definition["candidate_ids"]]
    by_id = {row["candidate_id"]: row for row in rows}
    for study_id in round_definition["studies"]:
        spec = read_json(ROOT / f"configs/forge/searches/{study_id}.json")
        members = sorted((card for card in cards.values() if card["trainer_family"] == spec["trainer_family"]),
                         key=lambda card: card["configuration_id"])
        study = {"study_id": study_id, "trainer_family": spec["trainer_family"], "view": view["id"],
                 "policy_fingerprint": fingerprint, "tuning_through_tier": 1, "spec": spec, "spec_hash": stable_hash(spec),
                 "source_digest": source, "runtime_cohort": runtime, "campaign": campaign,
                 "selection": {"selection_complete": True, "all_trials_terminal": True, "qualified": False,
                               "selection_kind": "best_observed", "selected_candidate_id": members[0]["id"]},
                 "progression": {"comparison_complete": True}, "trials": []}
        for card in members:
            row = by_id[card["id"]]
            study["trials"].append({"candidate_id": card["id"], "configuration_id": card["configuration_id"],
                "candidate_revision": row["candidate_revision"], "source_digest": source, "submission_status": "blocked",
                "resolved_recipe": card["resolved_configuration_recipe"], "tasks": [
                    {**assignment, "gate_status": outcome["status"]} for assignment, outcome in zip(assignments, row["tasks"])]})
        study["input_digest"] = stable_hash(study)
        atomic_json(tmp_path / f"reports/forge/configuration-search/{study_id}.json", study)
    snapshot_path = "reports/forge/technique-evidence/source.json"
    atomic_json(tmp_path / snapshot_path, {"policy_fingerprint": fingerprint,
                                         "frozen_source": {"source_digests": [source]}})
    key = file_hash(tmp_path / snapshot_path)
    entry = {"snapshot": snapshot_path, "json_sha256": key, "source_commit": "commit", "candidates": {}}
    manifest = {"view": view["id"], "view_revision": 3, "policy_fingerprint": fingerprint,
                "tier_requirements": requirements, "cohorts": [entry]}
    atomic_json(tmp_path / exports.MANIFEST, manifest)
    for row in rows:
        row["publication_key"] = key
    publication = {"view": view["id"], "view_revision": 3, "policy_fingerprint": fingerprint,
        "tier_requirements": requirements, "publication_scope": "current_technique_inventory",
        "qualification_input": False, "qualification_reuse": False,
        "provenance": {"evidence_manifest_sha256": stable_hash(manifest)},
        "configuration_rows": rows, "evidence_sources": {key: entry}}
    publication["provenance"]["input_digest"] = stable_hash(publication)
    atomic_json(tmp_path / exports.PUBLICATION, publication)
    return tmp_path


def test_exports_are_idempotent_and_preserve_full_immutable_recipes_without_raw_receipts(completed):
    cards = {path: path.read_bytes() for path in (completed / "configs/forge/configurations").glob("*.json")}
    result = exports.regenerate(completed)
    assert result["changed"] == [exports.READOUT, exports.SELECTIONS]
    readout, selected = read_json(result["readout"]), read_json(result["selections"])
    assert readout["candidate_count"] == 47 and readout["status_counts"] == {"FAIL": 47}
    assert readout["stop_task_counts"] == {"ring16_acquisition": 47}
    assert all(row["first_failed_required_task"] == "ring16_acquisition" for row in readout["candidates"])
    assert len(selected["selections"]) == selected["selected_configuration_count"] == 5
    assert selected["eligible_default_candidate_id"] is None
    for row in selected["selections"]:
        assert row["qualified"] is row["eligible_for_default"] is row["default_adoption"] is False
        assert row["selection_kind"] == "best_observed"
        assert row["resolved_configuration_recipe"] == read_json(completed / row["declaration"])["resolved_configuration_recipe"]
    paths = [Path(result["readout"]), Path(result["selections"])]
    contents, times = [path.read_bytes() for path in paths], [path.stat().st_mtime_ns for path in paths]
    assert exports.regenerate(completed)["changed"] == []
    assert [path.read_bytes() for path in paths] == contents
    assert [path.stat().st_mtime_ns for path in paths] == times
    assert {path: path.read_bytes() for path in cards} == cards
    assert not (completed / "reports/forge/attempts").exists()
    assert not (completed / "runs").exists()
    assert not list((completed / "reports/forge/tier1-refresh").glob("*.md"))


@pytest.mark.parametrize("drift, message", [
    ("source", "47 candidates"), ("policy", "policy"), ("roster", "47 candidates"),
    ("pending", "terminal"), ("card", "immutable"), ("selection", "PASS-count/hash"),
    ("tamper", "publication input digest"),
])
def test_invalid_compact_inputs_are_refused_before_any_output_change(completed, drift, message):
    outputs = [completed / exports.READOUT, completed / exports.SELECTIONS]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("existing result\n")
    publication_path = completed / exports.PUBLICATION
    publication = read_json(publication_path)
    if drift == "source":
        publication["configuration_rows"][0]["bindings"]["source_digest"] = "different"
    elif drift == "policy":
        publication["policy_fingerprint"] = "different"
    elif drift == "roster":
        publication["configuration_rows"][0] = deepcopy(publication["configuration_rows"][1])
    elif drift in {"pending", "selection"}:
        round_definition = read_json(completed / exports.ROUND)
        path = completed / f"reports/forge/configuration-search/{round_definition['studies'][0]}.json"
        study = read_json(path)
        if drift == "pending":
            study["selection"]["all_trials_terminal"] = False
        else:
            study["selection"]["selected_candidate_id"] = study["trials"][1]["candidate_id"]
        study["input_digest"] = stable_hash({key: value for key, value in study.items() if key != "input_digest"})
        atomic_json(path, study)
    elif drift == "card":
        path = next((completed / "configs/forge/configurations").glob("*.json"))
        card = read_json(path)
        card["resolved_configuration_recipe"]["lr"] *= 2
        atomic_json(path, card)
    else:
        publication["configuration_rows"][0]["cost"]["wall_seconds"] = 100
    if drift != "tamper":
        publication["provenance"].pop("input_digest")
        publication["provenance"]["input_digest"] = stable_hash(publication)
    atomic_json(publication_path, publication)
    with pytest.raises(ValueError, match=message):
        exports.regenerate(completed)
    assert [path.read_text() for path in outputs] == ["existing result\n"] * 2

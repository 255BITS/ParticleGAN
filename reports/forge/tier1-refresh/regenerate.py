"""Regenerate display-only Tier 1 readout and selected Recipe exports."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, stable_hash
from experiments.forge.views import view_fingerprint


ROUND = "configs/forge/rounds/tier1-existing-configs-v1.json"
PUBLICATION = "reports/forge/technique-inventory.json"
MANIFEST = "reports/forge/technique-evidence/manifest.json"
READOUT = "reports/forge/tier1-refresh/readout.json"
SELECTIONS = "configs/forge/selections/tier1-existing-configs-v1.json"
NONPASS = {"FAIL", "INVALID", "INCOMPLETE", "BLOCKED", "ERROR", "UNCONFIRMED"}
TERMINAL_SUBMISSIONS = {"completed", "blocked", "terminal", "concluded"}


def _task_outcome(row, requirements):
    tasks = {task["task_id"]: task for task in row["tasks"]}
    if len(tasks) != len(row["tasks"]) or set(tasks) != {task for tier in requirements.values() for task in tier}:
        raise ValueError("publication row must retain the complete required task denominator")
    ordered = [tasks[name] for name in requirements["1"]]
    stop = next((task for task in ordered if task["status"] != "PASS"), None)
    if stop and stop["status"] not in NONPASS:
        raise ValueError("Tier 1 readout requires every candidate to be ordinary terminal")
    if stop:
        remaining = ordered[ordered.index(stop) + 1:]
        if any(task["status"] not in {"UNKNOWN", "BLOCKED"} for task in remaining):
            raise ValueError("required failure must stop remaining ordinary Tier 1 work")
    if any(tasks[name]["status"] not in {"UNKNOWN", "BLOCKED"}
           for tier in ("2", "3") for name in requirements[tier]):
        raise ValueError("this round authorizes no later-tier measurements")
    return ordered, stop


def build(root):
    """Read compact committed inputs only, validating everything before writes."""
    root, inputs = Path(root).resolve(), {}

    def load(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("publication input must stay inside the checkout")
        data = path.read_bytes()
        inputs[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    round_definition, publication, manifest = load(ROUND), load(PUBLICATION), load(MANIFEST)
    policy_path = f"configs/forge/views/{round_definition['view']}.json"
    policy = load(policy_path)
    requirements = {str(tier): [row["task"] for row in sorted(policy["assignments"],
                    key=lambda row: (row["qualification_tier"], row.get("order", 0), row["task"]))
                    if row["importance"] == "required" and row["qualification_tier"] == tier]
                    for tier in (1, 2, 3)}
    fingerprint = view_fingerprint(policy)
    if (round_definition["through_tier"] != 1 or round_definition["view_revision"] != 3 or
            policy["revision"] != 3 or [len(requirements[str(tier)]) for tier in (1, 2, 3)] != [5, 19, 2] or
            round_definition["required_denominator_by_tier"] != [5, 19, 2]):
        raise ValueError("Tier 1 refresh requires the frozen revision-3 5/19/2 policy")
    for document in (publication, manifest):
        if (document["view"] != policy["id"] or document["view_revision"] != 3 or
                document["policy_fingerprint"] != fingerprint or document["tier_requirements"] != requirements):
            raise ValueError("publication/manifest policy differs from the frozen current view")
    if (publication["publication_scope"] != "current_technique_inventory" or
            publication.get("qualification_input") is not False or publication.get("qualification_reuse") is not False or
            publication["provenance"]["evidence_manifest_sha256"] != stable_hash(manifest)):
        raise ValueError("readout requires the current display publication bound to its evidence manifest")
    publication_content = deepcopy(publication)
    recorded_digest = publication_content["provenance"].pop("input_digest")
    if recorded_digest != stable_hash(publication_content):
        raise ValueError("publication input digest mismatch")
    current_ids = sorted(path.stem for directory in ("ideas", "configurations")
                         for path in (root / "configs/forge" / directory).glob("*.json"))
    if (len(round_definition["candidate_ids"]) != 47 or len(set(round_definition["candidate_ids"])) != 47 or
            current_ids != sorted(round_definition["candidate_ids"]) or
            len(round_definition["idea_ids"]) != 15 or len(round_definition["configuration_ids"]) != 32 or
            sorted(round_definition["idea_ids"] + round_definition["configuration_ids"]) != current_ids):
        raise ValueError("current candidate roster differs from the exact frozen 47 declarations")
    studies = {}
    for study_id in round_definition["studies"]:
        study = load(f"reports/forge/configuration-search/{study_id}.json")
        declared_spec = load(f"configs/forge/searches/{study_id}.json")
        if (study["study_id"] != study_id or study["view"] != policy["id"] or
                study["policy_fingerprint"] != fingerprint or study["tuning_through_tier"] != 1):
            raise ValueError("completed study policy differs from the frozen Tier 1 scope")
        if (study["input_digest"] != stable_hash({key: value for key, value in study.items() if key != "input_digest"}) or
                study["spec_hash"] != stable_hash(study["spec"]) or
                study["spec_hash"] != stable_hash(declared_spec) or
                study["trainer_family"] != declared_spec["trainer_family"]):
            raise ValueError("completed study input/spec hash mismatch")
        if (study["selection"]["selection_complete"] is not True or
                study["selection"]["all_trials_terminal"] is not True or
                study["progression"]["comparison_complete"] is not True or
                any(trial["submission_status"].lower() not in TERMINAL_SUBMISSIONS for trial in study["trials"])):
            raise ValueError("all five studies must report complete ordinary terminal comparisons")
        studies[study_id] = study
    if len(studies) != 5 or len({study["trainer_family"] for study in studies.values()}) != 5:
        raise ValueError("selection export requires exactly five refreshed trainer families")
    sources = {study["source_digest"] for study in studies.values()}
    runtimes = {stable_hash(study["runtime_cohort"]) for study in studies.values()}
    if len(sources) != 1 or None in sources or len(runtimes) != 1:
        raise ValueError("completed studies require one exact source/runtime cohort")
    source = next(iter(sources))
    runtime = next(iter(studies.values()))["runtime_cohort"]
    campaign = load(round_definition["campaign"])
    if (any(study["campaign"] != campaign for study in studies.values()) or
            campaign["budget_seconds"] != round_definition["worst_case_campaign_reservation_seconds"] or
            campaign["candidate_budget_seconds"] != round_definition["candidate_reservation_seconds"] or
            runtime["execution_backend"] != round_definition["execution_backend"] or
            runtime["compute_profiles"]["cuda"]["model"] != round_definition["cuda_model"]):
        raise ValueError("completed studies differ from the registered campaign/runtime")
    rows = [row for row in publication["configuration_rows"] if row["bindings"].get("source_digest") == source
            and row["runtime_cohort"] == runtime]
    by_id = {row["candidate_id"]: row for row in rows}
    if len(rows) != 47 or len(by_id) != 47 or sorted(by_id) != current_ids:
        raise ValueError("current publication source cohort must contain exactly the frozen 47 candidates")
    registered = {entry["json_sha256"]: entry for entry in manifest["cohorts"]}
    for key in {row["publication_key"] for row in rows}:
        entry = registered.get(key)
        if not entry or publication["evidence_sources"].get(key) != entry:
            raise ValueError("current row source publication is not registered in the manifest")
        snapshot = load(entry["snapshot"])
        if inputs[entry["snapshot"]] != key or snapshot["policy_fingerprint"] != fingerprint or source not in snapshot["frozen_source"]["source_digests"]:
            raise ValueError("registered source snapshot hash/policy/source mismatch")
    legacy = load("configs/forge/legacy-ideas-v1.json")
    cards = {}
    for candidate_id in round_definition["configuration_ids"]:
        relative = f"configs/forge/configurations/{candidate_id}.json"
        cards[candidate_id] = load(relative)
        if inputs[relative] != legacy["declarations"][relative]:
            raise ValueError("immutable scientific configuration card changed")
    trials = {trial["candidate_id"]: (study, trial) for study in studies.values() for trial in study["trials"]}
    if sum(len(study["trials"]) for study in studies.values()) != 32 or sorted(trials) != sorted(cards):
        raise ValueError("completed studies differ from the exact 32-configuration roster")
    if dict(Counter(card["trainer_family"] for card in cards.values())) != round_definition["configuration_count_by_family"]:
        raise ValueError("immutable cards differ from the frozen per-family roster")
    readout_rows = []
    for candidate_id in sorted(by_id):
        row = by_id[candidate_id]
        ordered, stop = _task_outcome(row, requirements)
        passed = sum(task["status"] == "PASS" for task in ordered)
        if (row["tiers"]["1"]["passed"] != passed or any(row["tiers"][tier]["total"] != len(names)
                                                                  for tier, names in requirements.items())):
            raise ValueError("publication tier counts disagree with complete task outcomes")
        if candidate_id in trials:
            study, trial = trials[candidate_id]
            trial_statuses = {task["task"]: task["gate_status"] for task in trial["tasks"]}
            if (trial["source_digest"] != source or trial["candidate_revision"] != row["candidate_revision"] or
                    study["trainer_family"] != cards[candidate_id]["trainer_family"] or
                    trial["resolved_recipe"] != cards[candidate_id]["resolved_configuration_recipe"] or
                    row["bindings"]["recipe_sha256"] != stable_hash(trial["resolved_recipe"]) or
                    trial_statuses != {task["task_id"]: task["status"] for task in row["tasks"]}):
                raise ValueError("publication configuration differs from its completed study evidence")
        status = stop["status"] if stop else "PASS"
        conclusion = (f"All five Tier 1 requirements passed; later tiers remain unmeasured." if stop is None else
                      f"Stopped at {stop['task_id']} ({status}) after {passed}/5 Tier 1 passes; later work was not eligible.")
        next_action = ("Retain provisional Tier 1 result; later-tier work needs its own authorization." if stop is None else
                       "Resolve the declared compatibility blocker before a new cohort." if status == "BLOCKED" else
                       "Stop this exact revision; inspect the linked numerical failures before any substantive new hypothesis.")
        readout_rows.append({
            "candidate_id": candidate_id, "candidate_revision": row["candidate_revision"],
            "trainer_family": row["trainer_family"], "status": status, "tier1_passed": passed, "tier1_required": 5,
            "first_failed_required_task": stop["task_id"] if stop and status != "BLOCKED" else None,
            "first_stopping_required_task": {"task_id": stop["task_id"], "status": status} if stop else None,
            "tasks": {task["task_id"]: task["status"] for task in ordered},
            "cost": row["cost"], "blockers": row.get("blockers", []),
            "stopping_reasons": publication.get("status_reasons", {}).get(stop.get("reasons_sha256"), []) if stop else [],
            "evidence": {"publication_key": row["publication_key"], "attempt_ids": row.get("attempt_ids", []),
                         "compact_receipts": [f"reports/forge/technique-receipts/{attempt}.json" for attempt in row.get("attempt_ids", [])]},
            "conclusion": conclusion, "next_action": next_action,
        })
    selected = []
    for study_id, study in sorted(studies.items()):
        selection = study["selection"]
        candidate_id = selection["selected_candidate_id"]
        if candidate_id not in cards or candidate_id not in {trial["candidate_id"] for trial in study["trials"]}:
            raise ValueError("completed study selected configuration is missing from its own roster")
        expected = min(study["trials"], key=lambda trial: (
            -sum(task["importance"] == "required" and task["qualification_tier"] == 1 and
                 task["gate_status"] == "PASS" for task in trial["tasks"]), trial["configuration_id"]))
        if expected["candidate_id"] != candidate_id:
            raise ValueError("completed study selection differs from its frozen PASS-count/hash objective")
        card, row = cards[candidate_id], by_id[candidate_id]
        recipe = card["resolved_configuration_recipe"]
        qualified = all(task["status"] == "PASS" for task in _task_outcome(row, requirements)[0])
        if (selection["qualified"] != qualified or selection["selection_kind"] != ("qualified_winner" if qualified else "best_observed") or
                row["bindings"]["recipe_sha256"] != stable_hash(recipe) or
                trials[candidate_id][1]["resolved_recipe"] != recipe):
            raise ValueError("selected configuration recipe/qualification differs from its original card and evidence")
        selected.append({
            "trainer_family": study["trainer_family"], "study_id": study_id,
            "candidate_id": candidate_id, "configuration_id": card["configuration_id"],
            "candidate_revision": row["candidate_revision"], "qualified": qualified,
            "qualification_scope": "provisional Tier 1 only", "selection_kind": selection["selection_kind"],
            "eligible_for_default": False, "default_adoption": False,
            "declaration": f"configs/forge/configurations/{candidate_id}.json",
            "resolved_configuration_recipe": deepcopy(recipe),
            "recipe_scope": "Base candidate Recipe; task-owned resources, prior and schedule bind separately in the referenced contracts.",
            "task_contracts": row["bindings"].get("task_contracts", {}),
            "provenance": {"study_sha256": inputs[f"reports/forge/configuration-search/{study_id}.json"],
                           "study_input_digest": study["input_digest"], "spec_hash": study["spec_hash"],
                           "card_sha256": inputs[f"configs/forge/configurations/{candidate_id}.json"],
                           "source_digest": source, "policy_fingerprint": fingerprint,
                           "source_origin_commit": row["bindings"].get("source_origin_commit"),
                           "publication_key": row["publication_key"]},
        })
    common = {"schema_version": 1, "round": round_definition["id"], "view": policy["id"], "view_revision": 3,
              "policy_fingerprint": fingerprint, "source_digest": source, "runtime_cohort": runtime,
              "required_denominator_by_tier": [5, 19, 2], "through_tier": 1,
              "qualification_input": False, "qualification_reuse": False, "default_adoption": False,
              "provenance": {"input_sha256": dict(sorted(inputs.items())),
                             "publication_input_digest": publication["provenance"]["input_digest"]}}
    readout = {**deepcopy(common), "scope": "compact concluded readout of the exact 47 existing declarations",
               "candidate_count": 47, "status_counts": dict(sorted(Counter(row["status"] for row in readout_rows).items())),
               "stop_task_counts": dict(sorted(Counter(row["first_stopping_required_task"]["task_id"] for row in readout_rows
                                                       if row["first_stopping_required_task"]).items())),
               "candidates": readout_rows}
    exports = {**deepcopy(common), "scope": "display/convenience export; no public default is selected",
               "eligible_default_candidate_id": None, "selected_configuration_count": 5, "selections": selected}
    for document in (readout, exports):
        document["input_digest"] = stable_hash(document)
    return readout, exports


def regenerate(root):
    root = Path(root).resolve()
    readout, exports = build(root)  # All validation completes before either output changes.
    changed = []
    for relative, document in ((READOUT, readout), (SELECTIONS, exports)):
        path = root / relative
        if not path.exists() or json.loads(path.read_text()) != document:
            atomic_json(path, document)
            changed.append(relative)
    return {"readout": str(root / READOUT), "selections": str(root / SELECTIONS), "changed": changed,
            "candidates": 47, "selected_configurations": 5, "training_launched": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(regenerate(args.root), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

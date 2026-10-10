"""Propose the completed pacing winner's exact whole-row measurement pin.

Register the executed source with regenerate_technique_inventory.py first.
This helper validates every admitted trial against that independent numerical
snapshot, prints a proposed selection card, and optionally writes a compact
selection receipt. It never modifies the card, trains, or adopts a default.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import (
    CURRENT_SELECTION, _current_pin, comparison_cohort, family_for_candidate, family_row_pin,
)
from experiments.forge.views import load_view, view_fingerprint
from reports.forge.regenerate_technique_inventory import _validate_published_row

CAMPAIGN = "bcap-dualnorm-pacing-v2"
FAMILY = "bcap-dualnorm"
SOURCE_COMMIT = "a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be"
SOURCE_DIGEST = "f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226"
CONTROL = "bcap-dualnorm--78aa66ae5c32297882274c11f191638f2cd1d79a9be4963349db5ed3d2b9a249"
REQUIRED = ("gaussian1d_acquisition", "two_pole", "unused_token_hold", "ae_gan_hold",
            "ring16_acquisition", "five_word_joint_acquisition")
CLOCK = "clockfree_audit_measurement_v1"
REPORT = Path("reports/forge/dualnorm-pacing-v2")
SELECTION_KEYS = ("objective", "selection_complete", "selected_candidate_id", "selected_configuration_id",
                  "required_pass_count", "required_total", "qualified", "default_adoption")


def _signed(value, label, *, nested=False):
    unsigned = deepcopy(value)
    target = unsigned.get("provenance", {}) if nested else unsigned
    claimed = target.pop("input_digest", None)
    if claimed != stable_hash(unsigned):
        raise ValueError(label + " input digest mismatch")


def _tasks(trial):
    tier = [task for task in trial["tasks"] if task["qualification_tier"] == 1]
    tasks = {task["task"]: task for task in tier}
    if (len(tier) != 7 or set(tasks) != {*REQUIRED, CLOCK}
            or any(tasks[name]["importance"] != "required" for name in REQUIRED)
            or tasks[CLOCK]["importance"] != "diagnostic"
            or trial.get("submission_status", "").lower() not in {"completed", "blocked", "terminal", "concluded"}
            or any(not task.get("attempt_id") or task["gate_status"] not in {"PASS", "FAIL"}
                   for task in tier)):
        raise ValueError("complete every actual required and diagnostic attempt before selecting a measurement")
    return tasks


def completed_trials(root, result, snapshot):
    """Verify the complete 20+3+2 pool and its unmodified whole-row objective."""
    root = Path(root)
    if (result.get("status") != "completed" or result.get("admitted_recipes") != 25
            or result.get("source") != {"commit": SOURCE_COMMIT, "digest": SOURCE_DIGEST}
            or result.get("running_workers") != 0 or result.get("all_attempts_supervised_finished") is not True
            or result.get("automatic_retries") != 0 or result.get("default_adoption") is not False
            or result.get("control_candidate_id") != CONTROL
            or result.get("accounting", {}).get("reserved_seconds") != 0):
        raise ValueError("selection requires the completed exact executed campaign with no active work")
    if (snapshot.get("publication_scope") != "frozen_source"
            or snapshot.get("frozen_source", {}).get("commit") != SOURCE_COMMIT
            or set(snapshot.get("frozen_source", {}).get("source_digests", [])) != {SOURCE_DIGEST}):
        raise ValueError("selection requires the independently regraded exact executed source snapshot")
    _signed(snapshot, "source snapshot", nested=True)
    stages = result.get("stages", [])
    if ([stage.get("stage") for stage in stages] != ["A", "B", "C"]
            or any(stage.get("admitted") is not True or stage.get("execution_complete") is not True for stage in stages)):
        raise ValueError("the completed finite campaign must retain its exact three admitted stages")
    trials, inputs = [], {}
    for stage, count in zip(stages, (20, 3, 2)):
        relative = Path("reports/forge/configuration-search") / (stage["spec"]["id"] + ".json")
        report = read_json(root / relative)
        _signed(report, "search report")
        if (len(report["trials"]) != count or report.get("spec") != stage["spec"]
                or report.get("spec_hash") != stable_hash(stage["spec"])
                or report.get("trainer_family") != FAMILY or report.get("campaign", {}).get("id") != CAMPAIGN
                or report.get("view") != "discriminator_stability" or report.get("tuning_through_tier") != 1
                or report.get("execution_backend") != "cuda"
                or set(report.get("source_digests", [])) != {SOURCE_DIGEST}
                or report.get("policy_fingerprint") != snapshot.get("policy_fingerprint")):
            raise ValueError("search report differs from the finite executed campaign")
        for trial in report["trials"]:
            _tasks(trial)
        expected = select_configuration(report["trials"], 1)
        if any(report.get("selection", {}).get(key) != expected[key] for key in SELECTION_KEYS):
            raise ValueError("recorded search selection differs from the frozen whole-row objective")
        trials.extend(report["trials"])
        inputs[relative.as_posix()] = file_hash(root / relative)
    if len({trial["configuration_id"] for trial in trials}) != 25:
        raise ValueError("the whole comparison contains duplicate configurations")
    for key in ("source_digest", "runtime_cohort", "protocol_hash", "policy_fingerprint"):
        if any(not trial.get(key) for trial in trials) or len({stable_hash(trial[key]) for trial in trials}) != 1:
            raise ValueError("whole selection requires one frozen source/runtime/protocol/view cohort")
    if trials[0]["source_digest"] != SOURCE_DIGEST:
        raise ValueError("trial source differs from the executed source")
    recorded = {trial["candidate_id"]: trial for trial in result["trials"]}
    if len(recorded) != 25 or set(recorded) != {trial["candidate_id"] for trial in trials}:
        raise ValueError("campaign results omit or duplicate an admitted whole recipe")
    projected = ("candidate_id", "configuration_id", "candidate_revision", "resolved_recipe", "source_digest",
                 "runtime_cohort", "protocol_hash", "policy_fingerprint", "submission_status", "request_id", "attempt_ids")
    rows = {}
    for trial in trials:
        saved = recorded[trial["candidate_id"]]
        if (any(saved.get(key) != trial.get(key) for key in projected)
                or saved["tasks"] != [task for task in trial["tasks"] if task["qualification_tier"] == 1]):
            raise ValueError("campaign trial differs from its frozen search evidence")
        matches = [row for row in snapshot["rows"] if row.get("candidate_id") == trial["candidate_id"]
                   and row.get("bindings", {}).get("source_digest") == SOURCE_DIGEST
                   and row.get("runtime_cohort", {}).get("execution_backend") == "cuda"]
        if len(matches) != 1:
            raise ValueError("every admitted recipe needs one exact independently regraded CUDA row")
        row = deepcopy(matches[0])
        _validate_published_row(root, snapshot, row)
        bindings = row.get("bindings", {})
        statuses = {task["task_id"]: task["status"]
                    for task in [*row["tasks"], *row.get("nonrequired_tasks", [])]}
        if (row.get("candidate_revision") != trial["candidate_revision"] or row.get("runtime_cohort") != trial["runtime_cohort"]
                or bindings.get("source_origin_commit") != SOURCE_COMMIT
                or bindings.get("recipe_sha256") != stable_hash(trial["resolved_recipe"])
                or bindings.get("protocol_sha256") != trial["protocol_hash"]
                or bindings.get("task_keys_sha256") != stable_hash({task["task"]: task["compatibility_key"] for task in trial["tasks"]})
                or set(row.get("attempt_ids", [])) != set(trial["attempt_ids"])
                or any(statuses.get(name) != task["gate_status"] for name, task in _tasks(trial).items())):
            raise ValueError("independently regraded row differs from its exact frozen search evidence")
        row["trainer_family"] = FAMILY
        rows[trial["candidate_id"]] = row
    if len({comparison_cohort(row, snapshot, task_ids=[*REQUIRED, CLOCK]) for row in rows.values()}) != 1:
        raise ValueError("admitted rows differ in task, prior, initialization, sampling or protocol conditions")
    protocol = snapshot.get("protocol_contracts", {}).get(trials[0]["protocol_hash"])
    if protocol is None or stable_hash(protocol) != trials[0]["protocol_hash"] or protocol.get("seed") != 0:
        raise ValueError("the independently graded campaign must retain protocol seed zero")
    expected = select_configuration(trials, 1)
    if any(result.get("selection", {}).get(key) != expected[key] for key in SELECTION_KEYS):
        raise ValueError("campaign winner differs from the frozen whole-row PASS/hash objective")
    return trials, rows, expected, inputs


def proposed_selection(root, snapshot):
    root = Path(root)
    result = read_json(root / REPORT / "results.json")
    trials, rows, selection, inputs = completed_trials(root, result, snapshot)
    policy = view_fingerprint(load_view(root, "discriminator_stability"))
    card = read_json(root / CURRENT_SELECTION)
    if snapshot.get("policy_fingerprint") != policy or card.get("policy_fingerprint") != policy:
        raise ValueError("source snapshot or retained selection card differs from the current view policy")
    control = next((trial for trial in trials if trial["candidate_id"] == result["control_candidate_id"]), None)
    recipe = (control or {}).get("resolved_recipe", {})
    if (recipe.get("optimizer_family") != "dualnorm" or recipe.get("optimizer_momentum") != 0
            or recipe.get("lr") != .01 or recipe.get("d_lr_mult") != 1.5 or recipe.get("prior_lr_mult") != 3):
        raise ValueError("the matched current-source control must retain the exact starter recipe")
    control_passes = sum(_tasks(control)[name]["gate_status"] == "PASS" for name in REQUIRED)
    if result.get("control_required_passes") != control_passes or selection["required_pass_count"] <= control_passes:
        raise ValueError("a new measurement requires a strict whole-recipe improvement over the matched current-source control")
    candidate = selection["selected_candidate_id"]
    row = rows[candidate]
    if family_for_candidate(root, candidate, {"trainer_family": FAMILY})["id"] != FAMILY:
        raise ValueError("selected candidate differs from the registered full-dualnorm family")
    reason = ("Completed optimizer-only BCAP pacing comparison: select one whole recipe by required Tier 1 PASS count "
              "descending, then configuration hash ascending, across all 25 complete current-source recipes. "
              f"The selected recipe passes {selection['required_pass_count']}/6 versus the matched starter's {control_passes}/6. "
              "This pins a provisional current measurement; calibration, confirmation and default adoption remain separate.")
    pin = family_row_pin(row, selection_kind="current_measurement", reason=reason,
                         measurement_views=["discriminator_stability"])
    _current_pin(root, FAMILY, [row], pin, view_id="discriminator_stability", catalogs=snapshot)
    matches = [index for index, previous in enumerate(card["selections"]) if previous["trainer_family"] == FAMILY]
    if len(matches) != 1 or card.get("default_adoption") is not False:
        raise ValueError("retained card must contain one experimental dualnorm measurement")
    proposed = deepcopy(card)
    previous = deepcopy(proposed["selections"][matches[0]])
    proposed["selections"][matches[0]] = pin
    receipt = {"schema_version": 1, "campaign": CAMPAIGN, "qualification_input": False,
        "selection_kind": "current_measurement", "source_commit": SOURCE_COMMIT, "source_digest": SOURCE_DIGEST,
        "source_snapshot_input_digest": snapshot["provenance"]["input_digest"],
        "results_sha256": file_hash(root / REPORT / "results.json"), "search_report_sha256": inputs,
        "selected_candidate_id": candidate, "selected_configuration_id": selection["selected_configuration_id"],
        "required_passes": selection["required_pass_count"], "required_total": 6,
        "control_candidate_id": control["candidate_id"], "control_required_passes": control_passes,
        "objective": selection["objective"], "qualified": selection["qualified"], "calibration_status": "provisional",
        "independent_confirmation": "not_performed", "default_adoption": False,
        "previous_selection": previous, "new_selection": pin, "updated_family_only": FAMILY,
        "preserved_other_selections_sha256": stable_hash([item for item in card["selections"] if item["trainer_family"] != FAMILY]),
        "preserved_historical_selections_sha256": stable_hash(card.get("historical_selections", []))}
    receipt["input_digest"] = stable_hash(receipt)
    return proposed, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, help="optional compact measurement-selection.json; never edits the selection card")
    args = parser.parse_args()
    args.root = args.root.resolve()
    path = args.snapshot if args.snapshot.is_absolute() else args.root / args.snapshot
    proposed, receipt = proposed_selection(args.root, read_json(path))
    if args.receipt:
        target = args.receipt if args.receipt.is_absolute() else args.root / args.receipt
        receipt.update(source_snapshot=path.relative_to(args.root).as_posix(), source_snapshot_sha256=file_hash(path))
        receipt["input_digest"] = stable_hash({key: value for key, value in receipt.items() if key != "input_digest"})
        atomic_json(target, receipt)
    print(json.dumps(proposed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

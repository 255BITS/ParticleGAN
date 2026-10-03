"""Recall projections of compact publications, never qualification evidence.

Read only committed summary schemas. Do not load checkpoints, original execution
envelopes, per-update streams, or paths on the machine that ran the study.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .contracts import file_hash, read_json, stable_hash

VERSION = "forge-publication-memory-v1"
OUTPUT = "reports/forge/publication-records.json"
POLICY_BOARD = "reports/forge/policy-family-inventory.json"
COMPLETION = "reports/forge/family-winner-round1/campaign-completion.json"


def input_paths(root: Path) -> list[Path]:
    """Authoritative compact reports and bindings, excluding our projections."""
    paths = set()
    for directory in ("configuration-search", "technique-evidence", "technique-receipts", "family-winner-round1"):
        paths.update((root / "reports/forge" / directory).rglob("*.json"))
    for name in ("technique-inventory.json", "technique-inventory.md", "policy-family-inventory.json",
                 "policy-family-inventory.md"):
        path = root / "reports/forge" / name
        if path.is_file():
            paths.add(path)
    return sorted(paths)


def _source(root, path, board):
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path), "board": board}


def _base(source, study_id, candidate_id, identity, goal, record_type):
    return {"schema_version": 1, "record_id": "publication-" + stable_hash(identity)[:24],
            "record_type": record_type, "candidate_id": candidate_id, "study_id": study_id,
            "goal": goal, "evidence_scope": "published_summary", "lifecycle": "concluded",
            "qualification_input": False, "qualification_reuse": False,
            "source": source, "materialization": OUTPUT, "task_results": [],
            "next_action": "Read the source-bound study and original evidence before another bounded hypothesis. "
                           "Unknown requirements remain unknown; this projection grants no qualification or adoption."}


def _conclusion(status, rows):
    counts = dict(sorted(Counter(row["gate_status"] for row in rows).items()))
    failed = [row["task_id"] for row in rows if row["gate_status"] in {"FAIL", "INVALID", "BLOCKED"}]
    return f"Recorded {status}; required outcomes {counts}." + (f" Failed requirements: {', '.join(failed)}." if failed else "")


def _search_records(root, path, report):
    if report.get("schema_version") != 1 or not report.get("study_id") or not isinstance(report.get("trials"), list):
        raise ValueError(f"unsupported compact search publication: {path}")
    # An in-progress publication participates in freshness, but cannot be called
    # a concluded study. Terminal task failures and unknowns are not conflated.
    if report.get("selection", {}).get("all_trials_terminal") is not True:
        return []
    study_id = report["study_id"]
    source = _source(root, path, "reports/forge/technique-inventory.md")
    records = []
    for trial in report["trials"]:
        declaration = trial.get("declaration", {})
        record = _base(source, study_id, trial["candidate_id"],
                       {"publication": source["path"], "study": study_id, "candidate": trial["candidate_id"],
                        "revision": trial["candidate_revision"]},
                       declaration.get("goal", report.get("view")), "published_trial")
        record.update(candidate_revision=trial["candidate_revision"], configuration_id=trial["configuration_id"],
                      hypothesis=declaration.get("hypothesis", report.get("spec", {}).get("hypothesis")),
                      mechanism_class=declaration.get("mechanism_class", "floor_constant"),
                      trainer_family=trial.get("trainer_family", report.get("trainer_family")),
                      prior=declaration.get("prior"), claim_contract=declaration.get("claim_contract"),
                      settings=trial.get("settings"), status=trial["status"],
                      recorded_cost=trial.get("cost"), recorded_qualification=trial.get("qualification"),
                      provenance={"source_digest": trial.get("source_digest"), "protocol_hash": trial.get("protocol_hash"),
                                  "runtime_cohort": trial.get("runtime_cohort"),
                                  "recipe_sha256": stable_hash(trial.get("resolved_recipe")),
                                  "receipt_bindings": trial.get("receipt_bindings", [])})
        rows = []
        for task in trial["tasks"]:
            if task["gate_status"] in {"UNKNOWN", "NOT_RUN"} and not task.get("attempt_id"):
                rows.append({"task_id": task["task"], "gate_status": task["gate_status"],
                             "qualification_tier": task.get("qualification_tier"), "metrics": {}})
                continue
            rows.append({"task_id": task["task"], "gate_status": task["gate_status"],
                         "metrics": task.get("metrics", {}), "reason": task.get("reason"),
                         "cost": {"wall_seconds": task.get("cost", {}).get("wall_seconds")},
                         "compatibility_key": task.get("compatibility_key"),
                         "_attempt_id": task.get("attempt_id"),
                         "qualification_tier": task.get("qualification_tier"),
                         "failed_bounds": task.get("failed_bounds", [])})
        record.update(task_results=rows, conclusion=_conclusion(trial["status"], rows))
        records.append(record)
    study = _base(source, study_id, study_id, {"publication": source["path"], "study": study_id},
                  report.get("view"), "published_study")
    study.update(hypothesis=report.get("spec", {}).get("hypothesis"), mechanism_class="bounded_configuration_search",
                 trainer_family=report.get("trainer_family"), trial_ids=[r["candidate_id"] for r in records],
                 conclusion=f"Concluded {len(records)} whole configurations. Recorded selection: "
                            f"{report['selection'].get('selection_kind')}; default adoption {report.get('default_adoption')}.",
                 provenance={"source_digests": report.get("source_digests"), "spec_hash": report.get("spec_hash"),
                             "protocol_hash": report.get("protocol_hash"), "runtime_cohort": report.get("runtime_cohort")},
                 recorded_cost=report.get("cost"))
    return [study, *records]


def _policy_records(root, path, board):
    if board.get("schema") != "policy_family_goal_inventory_v1" or not isinstance(board.get("cohorts"), list):
        raise ValueError(f"unsupported policy publication: {path}")
    records = []
    source = _source(root, path, "reports/forge/policy-family-inventory.md")
    for cohort in board["cohorts"]:
        if cohort.get("selection", {}).get("attempts_concluded") is not True:
            continue
        study_id, cohort_id = cohort["study_id"], cohort["id"]
        trials = []
        for trial in cohort["trials"]:
            record = _base(source, study_id, trial["id"],
                           {"study": study_id, "cohort": cohort_id, "candidate": trial["id"]},
                           board["goal"], "published_trial")
            record.update(configuration_id=trial["id"].split("--", 1)[-1],
                          candidate_revision=None, trainer_family=trial["family"],
                          mechanism_class="policy_configuration_search", settings=trial.get("recipe_overrides"),
                          hypothesis="One complete public selected/served policy configuration across the frozen required hosts.",
                          status=trial["status"], cohort=cohort_id,
                          recorded_cost={"wall_seconds": trial.get("paid_wall_seconds")},
                          provenance={"source_commit": cohort.get("source", {}).get("commit"),
                                      "runtime_contract": cohort.get("runtime_contract"), "spec_sha256": cohort.get("spec_sha256"),
                                      "combined_archive": cohort.get("combined_archive"),
                                      "family_archives": cohort.get("family_archives")})
            rows = []
            for case in trial["cases"]:
                if (not case.get("study_gate") and not case.get("receipt_sha256")
                        and not case.get("original_gate") and not case.get("paid_wall_seconds")):
                    rows.append({"task_id": case["id"], "gate_status": "UNKNOWN", "metrics": {},
                                 "qualification_tier": case.get("tier")})
                    continue
                rows.append({"task_id": case["id"], "gate_status": case.get("study_gate") or "UNKNOWN",
                             "original_gate": case.get("original_gate"), "metrics": case.get("final_metrics") or {},
                             "failed_bounds": case.get("original_failed_bounds") or [],
                             "reason": case.get("reason") or (case.get("acquisition_hold") or {}).get("reason"),
                             "cost": {"wall_seconds": case.get("paid_wall_seconds")},
                             "qualification_tier": case.get("tier"), "acquisition_hold": case.get("acquisition_hold"),
                             "sampling": case.get("sampling"), "case_sha256": case.get("case_sha256"),
                             "resolved_recipe_sha256": case.get("resolved_recipe_sha256"),
                             "receipt": {"path": case.get("receipt_path"), "sha256": case.get("receipt_sha256")},
                             "raw_artifacts": case.get("raw_artifacts", {})})
            record.update(task_results=rows, conclusion=_conclusion(trial["status"], rows))
            trials.append(record)
        study = _base(source, study_id, study_id, {"study": study_id, "cohort": cohort_id}, board["goal"], "published_study")
        study.update(cohort=cohort_id, mechanism_class="policy_configuration_search",
                     trial_ids=[record["candidate_id"] for record in trials],
                     hypothesis="Source/spec/runtime-bound policy acquisition and uninterrupted hold across required original API hosts.",
                     conclusion=f"Concluded {len(trials)} configurations; recorded outcome {cohort['selection'].get('outcome')}. "
                                "Original terminal gates and study persistence gates remain distinct.",
                     provenance={"source_commit": cohort.get("source", {}).get("commit"),
                                 "spec_sha256": cohort.get("spec_sha256"), "combined_archive": cohort.get("combined_archive")},
                     recorded_cost={"wall_seconds": cohort.get("measured_paid_seconds")})
        records.extend([study, *trials])
    return records


def normalize(root: Path) -> list[dict]:
    """Produce stable recall records from available, recognized publications."""
    root = Path(root)
    records = []
    for path in sorted((root / "reports/forge/configuration-search").glob("*.json")):
        records.extend(_search_records(root, path, read_json(path)))
    path = root / POLICY_BOARD
    if path.is_file():
        records.extend(_policy_records(root, path, read_json(path)))
    path = root / COMPLETION
    if path.is_file():
        completion = read_json(path)
        if completion.get("schema") != "particlegan_bounded_family_screen_completion_v1":
            raise ValueError(f"unsupported campaign completion publication: {path}")
        record = _base(_source(root, path, "reports/forge/family-winner-round1/README.md"),
                       "family-winner-round1", "family-winner-round1",
                       {"publication": COMPLETION}, completion.get("goal"), "published_study")
        record.update(mechanism_class="bounded_family_screen", hypothesis=completion.get("comparison_scope"),
                      conclusion=f"{completion.get('whole_configurations')} configurations; {completion.get('status')}. "
                                 f"{completion.get('default_limit')}",
                      recorded_cost={"wall_seconds": completion.get("summed_paid_child_and_setup_seconds")},
                      provenance={"mog_screen": completion.get("mog_screen"), "policy_cohorts":
                                  completion.get("policy_screen", {}).get("cohorts"),
                                  "cost_scope": completion.get("cost_scope")})
        records.append(record)
    identities = [record["record_id"] for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate publication recall identity")
    return sorted(records, key=lambda record: record["record_id"])

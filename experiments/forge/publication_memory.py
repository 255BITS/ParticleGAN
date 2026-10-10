"""Recall projections of compact publications, never qualification evidence.

Read only committed summary schemas. Do not load checkpoints, original execution
envelopes, per-update streams, or paths on the machine that ran the study.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

from .contracts import file_hash, read_json, stable_hash

VERSION = "forge-publication-memory-v1"
OUTPUT = "reports/forge/publication-records.json"
POLICY_BOARD = "reports/forge/policy-family-inventory.json"
COMPLETION = "reports/forge/family-winner-round1/campaign-completion.json"
PASSIVE_REGISTRY = "reports/forge/passive-publications.json"
PR223_SCHEMA = "pg_pr223_full19_passive_publication_v1"
PR223_PROTOCOL = "reports/forge/pr223-original-full-retest-20261004/protocol.json"
PR223_PROTOCOL_SHA = "8a5f0f63839e613b051562f030486388e8fcf962b6b6987fe5aaa27ea1f59786"


def _require(condition, message):
    if not condition:
        raise ValueError("passive publication: " + message)


def _equal_cost(actual, expected):
    _require(type(actual) in (int, float) and math.isfinite(actual) and actual >= 0,
             "finite nonnegative cost required")
    _require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8), "cost arithmetic changed")


def _same_json(actual, expected):
    return stable_hash(actual) == stable_hash(expected)


def _public_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            _require(key not in {"token", "attempt_token", "lease_fd", "lease_fds", "credential", "credentials"},
                     "private execution fields cannot enter recall")
            _public_fields(child)
    elif isinstance(value, list):
        for child in value:
            _public_fields(child)


def _pr223_cut(root, entry, report, final, proof):
    """Project recorded gates and final costs; never open an execution path."""
    from .completed_studies import _json, _read_pin

    _require(entry["protocol"]["path"] == PR223_PROTOCOL
             and entry["protocol"]["sha256"] == PR223_PROTOCOL_SHA, "original protocol identity changed")
    protocol = _json(_read_pin(root, entry["protocol"]))
    _public_fields(report)
    _public_fields(final)
    _require(report.get("family") == "atlas" and report.get("required") == 19
             and type(report.get("required")) is int, "all 19 original questions required")
    rows = report.get("rows", [])
    _require(len(rows) == 19 and [r.get("id") for r in rows] == [r["id"] for r in protocol["rows"]],
             "original ordered question denominator changed")
    source = report.get("source", {})
    from .completed_studies import HEX
    import re
    _require(re.fullmatch(r"[0-9a-f]{40}", source.get("origin_commit", ""))
             and HEX.fullmatch(source.get("digest", "")), "exact execution source required")
    _require(_same_json(source, entry.get("source")), "registered execution source changed")
    claims = report.get("claims", {})
    for key in ("historical_passes_are_current_credit", "current26_qualification", "default_adoption",
                "speed_ranking", "named10500_cost_pooling"):
        _require(claims.get(key) is False, "original retest cannot grant pooled/default/speed credit")
    _require(report.get("accepted_full_retest_status") == "PENDING_PUBLICATION_COST_FINALIZATION",
             "immutable cut must retain its pre-finalization status")
    statuses = {"PASS", "FAIL", "INVALID", "INCOMPLETE", "BUDGET_EXCEEDED", "BLOCKED", "NOT_RUN", "UNKNOWN"}
    cells, media, reached_end = [], [], False
    for row, declared in zip(rows, protocol["rows"]):
        definition = declared["original_definition"]
        for key in ("group", "task"):
            _require(row.get(key) == declared[key], "original host identity changed")
        for key in ("original_requirements", "observation_steps", "original_options", "sampling"):
            _require(_same_json(row.get(key), definition[key]), "original gate/cadence/sampling changed")
        _require(_same_json(row.get("required_host"), definition["original_host"])
                 and _same_json(row.get("complete_recipe"), declared["resolved_recipe"]), "full original host/Recipe changed")
        _require(isinstance(row.get("case_sha256"), str) and HEX.fullmatch(row["case_sha256"]),
                 "case fingerprint required")
        status = row.get("execution_status")
        _require(status in statuses and type(row.get("full_protocol_complete")) is bool, "unknown execution status")
        accepted = status in {"PASS", "FAIL"}
        _require(row.get("accepted_original_gate") == (status if accepted else None)
                 and row["full_protocol_complete"] is accepted, "raw/accepted gate distinction changed")
        if not accepted:
            reached_end = True
        elif accepted:
            _require(not reached_end and row.get("raw_reported_status") == status,
                     "accepted result after an unreached slot or changed raw verdict")
            _require(row.get("completed_steps") == definition["original_host"]["steps"]
                     and row.get("metric_observations") == len(definition["observation_steps"]),
                     "accepted gate lacks full original execution")
            if row["group"] == "native":
                noisy = row.get("native_gates", {}).get("noisy", {})
                _require(set(noisy) == {"coverage", "accuracy"}
                         and set(noisy.values()) <= {"PASS", "FAIL"}
                         and (all(v == "PASS" for v in noisy.values())) is (status == "PASS"),
                         "native primary coverage/accuracy joint grade changed")
        value = row.get("media")
        if accepted:
            _require(isinstance(value, dict) and value.get("original_gate") == status
                     and value.get("actual_steps") == declared["media_steps"]
                     and value.get("frames") == len(declared["media_steps"]), "original goal media scope changed")
            directory = str(Path(entry["result"]["path"]).parent)
            media.append({"path": directory + "/" + value["file"], "sha256": value["sha256"], "bytes": value["bytes"]})
        else:
            _require(value is None, "unaccepted gate cannot acquire goal media credit")
        cost = row.get("cost", {})
        allowance = declared["proposed_inclusive_allowance_seconds"]
        _equal_cost(cost.get("allowance_seconds"), allowance)
        for key in ("paid_wall_seconds", "reserved_seconds", "charged_seconds", "overrun_seconds"):
            _equal_cost(cost.get(key), cost.get(key, -1))
        paid, reserve = cost["paid_wall_seconds"], cost["reserved_seconds"]
        _equal_cost(cost["charged_seconds"], paid + reserve)
        _equal_cost(cost.get("unmeasured_interrupt_reserved_seconds"), reserve)
        _equal_cost(cost["overrun_seconds"], max(0, paid - allowance))
        terminal = cost.get("terminal_status")
        _require(terminal in {"completed", "missing", "error", "timeout", "cancelled"}
                 and cost.get("completed_terminal") is (terminal == "completed")
                 and cost.get("certified") is accepted, "durable completion/certification distinction changed")
        if status in {"NOT_RUN", "UNKNOWN", "BLOCKED"}:
            _equal_cost(paid + reserve, 0)
        elif terminal == "completed":
            _equal_cost(reserve, 0)
        else:
            _equal_cost(reserve, max(0, allowance - paid))
        if accepted:
            _require(terminal == "completed", "accepted gate requires a completed durable attempt")
        cells.append({"task_id": row["id"], "gate_status": status,
                      "original_gate": row["accepted_original_gate"], "raw_status": row.get("raw_reported_status"),
                      "metrics": deepcopy(row.get("final_metrics", {})), "cost": deepcopy(cost),
                      "case_sha256": row["case_sha256"], "recipe_sha256": stable_hash(row["complete_recipe"]),
                      "sampling": row["sampling"], "original_requirements": deepcopy(row["original_requirements"]),
                      "media": deepcopy(value)})
    counts = dict(sorted(Counter(row["execution_status"] for row in rows).items()))
    _require(type(report.get("counts")) is dict and set(report["counts"]) == statuses
             and all(type(n) is int and n >= 0 for n in report["counts"].values())
             and counts == {k: v for k, v in report["counts"].items() if v}, "execution count drift")
    completed = sum(row["full_protocol_complete"] for row in rows)
    accepted_counts = {"PASS": counts.get("PASS", 0), "FAIL": counts.get("FAIL", 0), "UNAVAILABLE": 19 - completed}
    _require(type(report.get("completed")) is int and report["completed"] == completed
             and all(type(n) is int for n in report.get("accepted_original_gate_counts", {}).values())
             and report.get("accepted_original_gate_counts") == accepted_counts
             and report.get("all_required_case_evidence_complete") is (completed == 19), "accepted count drift")
    raw_gate = ("PASS" if counts.get("PASS") == 19 else "FAIL") if completed == 19 else "UNAVAILABLE"
    _require(report.get("raw_full_protocol_gate") == raw_gate, "partial cut cannot supply full19 grade")
    _require(entry["media"] == media, "all accepted original GIFs must be pinned")
    for pin in media:
        _require(_read_pin(root, pin)[:6] in (b"GIF89a", b"GIF87a"), "original goal GIF required")
    card = report.get("trusted_terminal_card", {})
    _require(isinstance(card.get("sha256"), str) and HEX.fullmatch(card["sha256"]), "trusted terminal-card pin required")
    _require(card["sha256"] == entry.get("terminal_card_sha256"), "registered terminal-card identity changed")
    _require(final.get("schema") == "pg_pr223_final_publication_cost_v1"
             and final.get("original_terminal_cut_results_sha256") == entry["result"]["sha256"]
             and final.get("trusted_terminal_card_sha256") == card["sha256"]
             and final.get("original_case_verdicts_unchanged") is True
             and final.get("no_old_cost_or_qualification_pooling") is True
             and final.get("raw_full_protocol_gate") == raw_gate, "FINAL_COST/result/source-card join changed")
    costs = deepcopy(final["costs"])
    for key, expected in (("aggregate_cap_seconds", 10800), ("metadata_cap_seconds", 180),
                          ("case_caps_sum_seconds", 9810), ("export_grace_seconds", 0), ("retries", 0)):
        _equal_cost(costs.get(key), expected)
    paid = math.fsum(row["cost"]["paid_wall_seconds"] for row in rows)
    reserve = math.fsum(row["cost"]["reserved_seconds"] for row in rows)
    for key, expected in (("case_paid_wall_seconds", paid), ("case_reserved_seconds", reserve),
                          ("case_charged_seconds", paid + reserve)):
        _equal_cost(costs.get(key), expected)
    metadata = costs["metadata"]
    _require(type(metadata.get("blocked")) is bool and type(costs.get("halt_required")) is bool,
             "explicit final budget status required")
    for key in ("paid_wall_seconds", "reserved_seconds", "overrun_seconds", "charged_seconds"):
        _equal_cost(metadata.get(key), metadata.get(key, -1))
    _equal_cost(metadata["charged_seconds"], metadata["paid_wall_seconds"] + metadata["reserved_seconds"])
    _equal_cost(metadata["overrun_seconds"], max(0, metadata["paid_wall_seconds"] - 180))
    _equal_cost(metadata["reserved_seconds"], max(0, 180 - metadata["paid_wall_seconds"]) if metadata["blocked"] else 0)
    _equal_cost(costs.get("charged_seconds"), paid + reserve + metadata["charged_seconds"])
    _require(type(costs.get("remaining_seconds")) in (int, float)
             and math.isclose(costs["remaining_seconds"], 10800 - costs["charged_seconds"], abs_tol=1e-8), "final remaining budget changed")
    halt = (metadata["blocked"] or metadata["overrun_seconds"] > 0 or costs["charged_seconds"] > 10800
            or any(row["cost"]["overrun_seconds"] > 0 for row in rows))
    _require(costs["halt_required"] is halt, "final budget halt changed")
    status = "INCOMPLETE" if halt or completed != 19 else raw_gate
    _require(final.get("status") == status, "final accepted status changed")
    snapshot = report["cost_snapshot_before_publication"]
    _require(metadata["charged_seconds"] >= snapshot["metadata"]["charged_seconds"]
             and type(final.get("metadata_phase_count")) is int
             and type(report.get("metadata_phase_count_before_publication")) is int
             and final["metadata_phase_count"] > report["metadata_phase_count_before_publication"], "final metadata cost/history rewound")
    _require(proof.get("schema") == "pg_pr223_passive_verification_v1"
             and proof.get("results_sha256") == entry["result"]["sha256"]
             and proof.get("trusted_card_sha256") == card["sha256"]
             and proof.get("counts") == report["counts"] and proof.get("required") == 19
             and proof.get("accepted_original_gifs") == completed, "passive certification join changed")
    for key in ("models", "draws", "scorer_calls", "grader_calls", "rendered_frames"):
        _require(type(proof.get(key)) is int and proof[key] == 0, "projection cannot perform scientific work")
    ledger = final.get("authoritative_final_metadata_ledger", {})
    _require(isinstance(ledger.get("sha256"), str) and HEX.fullmatch(ledger["sha256"])
             and type(ledger.get("bytes")) is int and ledger["bytes"] >= 0
             and ledger.get("path") == report.get("authoritative_metadata_ledger_path"), "final closed-ledger pin required")
    return {"id": entry["id"], "schema": report["schema"], "source": deepcopy(source),
            "required": 19, "completed": completed, "counts": counts, "accepted_counts": accepted_counts,
            "status": status, "raw_full_protocol_gate": raw_gate, "cost": costs, "cells": cells,
            "runtime_declaration": deepcopy(protocol["runtime"]),
            "actual_runtime_in_compact_result": False, "recipe": deepcopy(protocol["common_resolved_recipe"]),
            "publication": deepcopy(entry["result"]), "final_cost": deepcopy(entry["final_cost"]),
            "terminal_card_sha256": card["sha256"], "readout": entry["readout"]["path"],
            "qualification_input": False, "qualification_reuse": False, "default_adoption": False,
            "speed_ranking": False, "cost_scope": "Cumulative within this retest only; never sum overlapping cuts."}


# Add adapters here for reviewed passive schemas; paths and latest-cut pointers
# belong to the optional committed registry, not a hard-coded report date.
from .native3_publication_memory import SCHEMA as NATIVE3_SCHEMA, project_native3
from .native3_repaired_publication_memory import SCHEMA as NATIVE3_REPAIRED_SCHEMA, project_native3_repaired

PASSIVE_ADAPTERS = {PR223_SCHEMA: _pr223_cut, NATIVE3_SCHEMA: project_native3,
                    NATIVE3_REPAIRED_SCHEMA: project_native3_repaired}


def passive_publication_entry(root, directory):
    """Create a byte-bound registration; caller owns reviewing and writing it."""
    from .completed_studies import _json, _pin
    root = Path(root).resolve()
    entry = {"id": Path(directory).name}
    for role, filename in (("result", "results.json"), ("final_cost", "FINAL_COST.json"),
                           ("verification", "verification.json"), ("readout", "README.md")):
        entry[role] = _pin(root, directory + "/" + filename)
    report = _json((root / entry["result"]["path"]).read_bytes())
    entry["schema"] = report["schema"]
    entry["source"] = deepcopy(report["source"])
    entry["terminal_card_sha256"] = report["trusted_terminal_card"]["sha256"]
    _require(entry["schema"] in PASSIVE_ADAPTERS, "unsupported passive schema")
    entry["protocol"] = _pin(root, PR223_PROTOCOL)
    entry["media"] = [{"path": directory + "/" + row["media"]["file"],
                       "sha256": row["media"]["sha256"], "bytes": row["media"]["bytes"]}
                      for row in report["rows"] if row.get("media")]
    if entry["schema"] in {NATIVE3_SCHEMA, NATIVE3_REPAIRED_SCHEMA}:
        from .native3_publication_memory import PARENT_METADATA_PIN_PATH
        entry["parent_metadata_anchor"] = _pin(root, PARENT_METADATA_PIN_PATH)
    if entry["schema"] == NATIVE3_REPAIRED_SCHEMA:
        from .native3_repaired_publication_memory import PREDECESSOR_ANCHOR_PATH, PREDECESSOR_METADATA_PATH
        entry["predecessor_anchor"] = _pin(root, PREDECESSOR_ANCHOR_PATH)
        entry["predecessor_metadata_anchor"] = _pin(root, PREDECESSOR_METADATA_PATH)
    return entry


def load_passive_publications(root):
    """Read committed summaries only; raw/source snapshot paths stay inert."""
    from .completed_studies import _file, _json, _read_pin
    root = Path(root).resolve()
    if not (root / PASSIVE_REGISTRY).exists() and not (root / PASSIVE_REGISTRY).is_symlink():
        return {}
    registry = _json(_file(root, PASSIVE_REGISTRY).read_bytes())
    _require(registry.get("schema") == "forge_passive_publications_registry_v1"
             and all(registry.get(k) is False for k in ("qualification_input", "reuse", "cross_cohort_pooling")),
             "registry scope changed")
    entries = registry.get("publications", [])
    _require(isinstance(entries, list) and entries and len({e["id"] for e in entries}) == len(entries),
             "distinct passive cuts required")
    cuts, pins = [], []
    for entry in entries:
        _require(entry["schema"] in PASSIVE_ADAPTERS, "unsupported passive schema")
        directory = str(Path(entry["result"]["path"]).parent)
        _require(directory.startswith("reports/forge/") and entry["id"] == Path(directory).name, "committed report identity required")
        values = {}
        for role, suffix in (("result", "results.json"), ("final_cost", "FINAL_COST.json"),
                             ("verification", "verification.json"), ("readout", "README.md")):
            _require(entry[role]["path"] == directory + "/" + suffix, "cohort-local report input required")
            data = _read_pin(root, entry[role]);pins.append(deepcopy(entry[role]))
            if role != "readout": values[role] = _json(data)
        _require(values["result"].get("schema") == entry["schema"], "registered/report schema mismatch")
        cut = PASSIVE_ADAPTERS[entry["schema"]](root, entry, values["result"], values["final_cost"], values["verification"])
        cuts.append(cut);pins += [deepcopy(entry["protocol"]), *deepcopy(entry["media"]), *deepcopy(cut.get("inputs", []))]
    latest = registry.get("latest", {})
    _require(set(latest) == {cut["schema"] for cut in cuts}, "one explicit latest cut per schema required")
    selected = {}
    for schema, identifier in latest.items():
        matching = [cut for cut in cuts if cut["schema"] == schema and cut["id"] == identifier]
        _require(len(matching) == 1, "latest cut missing or substituted")
        chosen = matching[0]
        for earlier in [c for c in cuts if c["schema"] == schema]:
            _require(earlier["source"] == chosen["source"], "one retest source per latest-cut history")
            _require(earlier["completed"] <= chosen["completed"], "latest cut rewinds accepted coverage")
            _require(earlier["cost"]["charged_seconds"] <= chosen["cost"]["charged_seconds"], "latest cut resets cumulative costs")
            for old, new in zip(earlier["cells"], chosen["cells"]):
                if old["gate_status"] in {"PASS", "FAIL"}:
                    _require(old == new, "later cut rewrites an accepted historical case")
        selected[schema] = deepcopy(chosen)
    return {"registry": PASSIVE_REGISTRY, "cuts": cuts, "latest": selected, "inputs": pins,
            "qualification_input": False, "qualification_reuse": False, "cross_cohort_pooling": False}


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
    passive = load_passive_publications(root)
    if passive:
        paths.add(root / PASSIVE_REGISTRY)
        paths.update(root / pin["path"] for pin in passive["inputs"])
        if any(cut["schema"] == NATIVE3_SCHEMA for cut in passive["cuts"]):
            paths.add(root / "experiments/forge/native3_publication_memory.py")
        if any(cut["schema"] == NATIVE3_REPAIRED_SCHEMA for cut in passive["cuts"]):
            paths.add(root / "experiments/forge/native3_publication_memory.py")
            paths.add(root / "experiments/forge/native3_repaired_publication_memory.py")
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
    for cut in load_passive_publications(root).get("cuts", []):
        if cut["schema"] == NATIVE3_REPAIRED_SCHEMA:
            from .native3_repaired_publication_memory import recall_record
            records.append(recall_record(root, cut))
            continue
        if cut["schema"] == NATIVE3_SCHEMA:
            from .native3_publication_memory import recall_record
            records.append(recall_record(root, cut))
            continue
        source = {**cut["publication"], "board": "reports/forge/technique-inventory.md"}
        record = _base(source, "pr223-original-full19-retest", "atlas",
                       {"publication": cut["publication"], "source": cut["source"]},
                       "discriminator_stability", "published_study")
        record.update(lifecycle="concluded" if cut["completed"] == 19 else "closed_partial_cut",
                      status=cut["status"], trainer_family="atlas", cohort=cut["source"]["digest"],
                      mechanism_class="faithful_original_full19_retest", configuration_id=cut["id"],
                      settings={"lr": .00425, "prior_lr_mult": 2., "d_lr_mult": 1.},
                      task_results=cut["cells"], recorded_cost=cut["cost"],
                      provenance={"source_commit": cut["source"]["origin_commit"],
                                  "execution_digest": cut["source"]["digest"], "final_cost": cut["final_cost"],
                                  "terminal_card_sha256": cut["terminal_card_sha256"]},
                      next_action="Read the latest closed cut and authoritative FINAL_COST. Original19 results grant "
                                  "no current26 qualification, default or speed credit; continuation requires its own retained evidence.",
                      conclusion=f"Closed original19 cut: {cut['accepted_counts']['PASS']}/19 PASS, "
                                 f"{cut['accepted_counts']['FAIL']} FAIL, {cut['accepted_counts']['UNAVAILABLE']} unavailable. "
                                 f"Final accepted status {cut['status']}; required execution counts {cut['counts']}. "
                                 "Historical positives and overlapping cut costs are not pooled.")
        records.append(record)
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

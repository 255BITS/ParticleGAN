"""Passive repaired-native3 recall, versioned separately from V1 and full19.

Read only registry-pinned committed JSON and original GIF bytes. The old16,
first native3 invalid, and all source/ledger/raw paths remain inert provenance.
No scorer, state loader, scientific module, queue or renderer is imported.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import PurePosixPath
import re

from .contracts import stable_hash
from .native3_publication_memory import (
    TASKS, IDS, PARENT_SOURCE, PARENT_FILES, PARENT_DIRECTORY, PARENT_CASE_CHARGE,
    PARENT_METADATA_PAID, PARENT_METADATA_PATH, PARENT_METADATA_PIN_PATH,
    PARENT_METADATA_SHA, CANONICAL_LEDGER, STATUS_KEYS, CLAIMS,
    _require, _same, _number, _cost, _privacy, _parent, _row_cost, _phases,
)

SCHEMA = "pg_pr223_native3_passive_publication_v2"
FINAL_SCHEMA = "pg_pr223_native3_final_publication_cost_v2"
VERIFICATION_SCHEMA = "pg_pr223_native3_passive_verification_v2"
PREDECESSOR_SOURCE = {
    "origin_commit": "0fa92c5b599da8ab2c891e544aafc09cf802a70d",
    "digest": "b2d0e98000d06a9731a4a40e52e226fedece194c097a1f5d939f501a9360aaa6",
}
PRETRAINING_INVALID_CHARGE = 2.0496059330180287
COMBINED_PRIOR_CASE_CHARGE = math.fsum((PARENT_CASE_CHARGE, PRETRAINING_INVALID_CHARGE))
PREDECESSOR_ANCHOR_PATH = "reports/forge/pr223-native3-continuation-20261004/fixtures/first-invalid-debit-anchor.json"
PREDECESSOR_ANCHOR_SHA = "301b8036304c14f0c1a352af280e4c0dd2c8267f675294e62fadf3259fcd5ecb"
PREDECESSOR_METADATA_PATH = "reports/forge/pr223-native3-continuation-20261004/fixtures/closed-metadata-22.json"
PREDECESSOR_METADATA_SHA = "1cb2f30350c37bedec8e012eb674bda05641093248b9f1658db0bbab8026aaf2"
PREDECESSOR_FILES = {
    PREDECESSOR_ANCHOR_PATH: {"sha256": PREDECESSOR_ANCHOR_SHA, "bytes": 1948},
    PREDECESSOR_METADATA_PATH: {"sha256": PREDECESSOR_METADATA_SHA, "bytes": 4644},
}
PREDECESSOR_COST = {
    "schema": "pg_pr223_native3_predecessor_cost_v1", "source": PREDECESSOR_SOURCE,
    "files": PREDECESSOR_FILES, "original19_case_charged_seconds": PARENT_CASE_CHARGE,
    "pretraining_invalid_case_charged_seconds": PRETRAINING_INVALID_CHARGE,
    "combined_prior_case_charged_seconds": COMBINED_PRIOR_CASE_CHARGE,
    "canonical_metadata_ledger": CANONICAL_LEDGER, "required_closed_prefix_phases": 22,
    "execution_counts": {"INVALID": 1, "NOT_RUN": 2},
    "accepted_original_gate_counts": {"PASS": 0, "FAIL": 0, "UNAVAILABLE": 3},
    "training_started": False, "numerical_credit": False, "old_grades_are_current_credit": False,
}


def _predecessor(root, entry, report, parent):
    """Bind cost-only frozen JSON; never follow its runtime provenance paths."""
    from .completed_studies import _json, _read_pin

    _require(_same(report.get("predecessor_cost"), PREDECESSOR_COST),
             "cost-only pretraining predecessor changed or acquired numerical credit")
    for role, relative in (("predecessor_anchor", PREDECESSOR_ANCHOR_PATH),
                           ("predecessor_metadata_anchor", PREDECESSOR_METADATA_PATH)):
        _require(entry.get(role) == {"path": relative, **PREDECESSOR_FILES[relative]},
                 "immutable predecessor anchor pin changed")
    anchor = _json(_read_pin(root, entry["predecessor_anchor"]))
    closed = _json(_read_pin(root, entry["predecessor_metadata_anchor"]))
    _require(anchor.get("schema") == "pg_pr223_native3_first_invalid_debit_anchor_v1"
             and anchor.get("source") == PREDECESSOR_SOURCE
             and all(anchor.get(key) is False for key in ("training_started", "numerical_credit",
                 "historical_grades_transferred", "old_grades_are_current_credit"))
             and anchor.get("metadata_is_same_cumulative_180_not_extra_debit") is True,
             "immutable predecessor source/scope changed")
    _require(_same(anchor.get("execution_counts"), {"INVALID": 1, "NOT_RUN": 2})
             and _same(anchor.get("accepted_original_gate_counts"), {"PASS": 0, "FAIL": 0, "UNAVAILABLE": 3}),
             "pretraining invalid cannot transfer a grade")
    _cost(anchor.get("old19_case_charged_seconds"), PARENT_CASE_CHARGE)
    _cost(anchor.get("retained_total_case_charge_seconds"), COMBINED_PRIOR_CASE_CHARGE)
    for key, expected in (("paid_wall_seconds", PRETRAINING_INVALID_CHARGE),
                          ("charged_seconds", PRETRAINING_INVALID_CHARGE),
                          ("reserved_seconds", 0), ("overrun_seconds", 0)):
        _cost(anchor.get("case_cost", {}).get(key), expected)
    boundary = anchor.get("closed_metadata_before_sealing", {})
    _require(boundary.get("canonical_path") == CANONICAL_LEDGER
             and boundary.get("sha256") == PREDECESSOR_METADATA_SHA
             and boundary.get("bytes") == 4644 and boundary.get("phase_count") == 22,
             "predecessor canonical same-ledger boundary changed")
    _require(closed.get("schema") == "pg_pr223_original_full_retest_metadata_budget_v1"
             and closed.get("cap_seconds") == 180 and closed.get("aggregate_cap_seconds") == 10800
             and closed.get("current_phase") is None and closed.get("blocked") is False
             and closed.get("reason") is None and len(closed.get("phases", [])) == 22,
             "exact closed22 metadata boundary required")
    _phases(closed["phases"], parent)
    _cost(closed.get("paid_wall_seconds"), math.fsum(p["paid_wall_seconds"] for p in closed["phases"]))
    _cost(closed.get("charged_seconds"), closed["paid_wall_seconds"])
    _cost(boundary.get("charged_seconds"), closed["charged_seconds"])
    for key in ("reserved_seconds", "unmeasured_interrupt_reserved_seconds", "overrun_seconds"):
        _cost(closed.get(key), 0)
    _require(_same(report.get("predecessor_metadata_anchor"), {
        "sha256": PREDECESSOR_METADATA_SHA, "bytes": 4644, "phases": 22,
        "snapshot_relative": PREDECESSOR_METADATA_PATH,
        "first_invalid_anchor_sha256": PREDECESSOR_ANCHOR_SHA, "first_invalid_anchor_bytes": 1948,
        "historical_numerical_credit": False}), "reported predecessor22 identity changed")
    return closed


def _phases_v2(history, parent, predecessor):
    _phases(history, parent)
    _require(len(history) >= 22 and _same(history[:22], predecessor["phases"]),
             "same-ledger complete predecessor22 prefix changed")


def _accounting_v2(costs, rows, history, anchor, predecessor):
    for key, expected in (("aggregate_cap_seconds", 10800), ("metadata_cap_seconds", 180),
                          ("case_caps_sum_seconds", 4290), ("export_grace_seconds", 0), ("retries", 0),
                          ("prior_case_charged_seconds", PARENT_CASE_CHARGE),
                          ("pretraining_invalid_case_charged_seconds", PRETRAINING_INVALID_CHARGE),
                          ("combined_prior_case_charged_seconds", COMBINED_PRIOR_CASE_CHARGE)):
        _cost(costs.get(key), expected)
    paid = math.fsum(row["cost"]["paid_wall_seconds"] for row in rows)
    reserved = math.fsum(row["cost"]["reserved_seconds"] for row in rows)
    for key, expected in (("current_case_paid_wall_seconds", paid), ("current_case_reserved_seconds", reserved),
                          ("current_case_charged_seconds", paid + reserved)):
        _cost(costs.get(key), expected)
    metadata = costs.get("metadata", {})
    _require(type(metadata.get("blocked")) is bool and type(costs.get("halt_required")) is bool,
             "explicit budget status required")
    for key in ("paid_wall_seconds", "reserved_seconds", "charged_seconds", "overrun_seconds"):
        _number(metadata.get(key))
    _cost(metadata["charged_seconds"], metadata["paid_wall_seconds"] + metadata["reserved_seconds"])
    _cost(metadata["reserved_seconds"], max(0, 180 - metadata["paid_wall_seconds"]) if metadata["blocked"] else 0)
    _cost(metadata["overrun_seconds"], max(0, metadata["paid_wall_seconds"] - 180))
    _require(metadata["paid_wall_seconds"] >= predecessor["paid_wall_seconds"], "cumulative metadata reset")
    _phases_v2(history, anchor, predecessor)
    _cost(metadata["paid_wall_seconds"], math.fsum(phase["paid_wall_seconds"] for phase in history))
    charge = math.fsum((PARENT_CASE_CHARGE, PRETRAINING_INVALID_CHARGE, paid, reserved, metadata["charged_seconds"]))
    _cost(costs.get("charged_seconds"), charge)
    _cost(costs.get("overrun_seconds"), math.fsum(row["cost"]["overrun_seconds"] for row in rows)
          + metadata["overrun_seconds"])
    _cost(costs.get("aggregate_overrun_seconds"), max(0, charge - 10800))
    remaining = costs.get("remaining_seconds")
    _require(type(remaining) in (int, float) and math.isfinite(remaining)
             and math.isclose(remaining, 10800 - charge, rel_tol=0, abs_tol=1e-8), "inclusive remaining budget changed")
    halt = (metadata["blocked"] or metadata["overrun_seconds"] > 0 or charge > 10800
            or any(row["cost"]["overrun_seconds"] > 0 for row in rows))
    _require(costs["halt_required"] is halt, "inclusive budget halt changed")
    return halt


def project_native3_repaired(root, entry, report, final, proof):
    """Project V2 recorded flags; old19 and first-invalid are cost-only history."""
    from .completed_studies import HEX, _json, _read_pin
    from .publication_memory import PR223_PROTOCOL, PR223_PROTOCOL_SHA, _public_fields

    _require(entry["protocol"] == {"path": PR223_PROTOCOL, "sha256": PR223_PROTOCOL_SHA, "bytes": 157526},
             "complete original19 protocol pin changed")
    protocol = _json(_read_pin(root, entry["protocol"]))
    definitions = [row for row in protocol["rows"] if row["group"] == "native"]
    _require(len(definitions) == 3 and tuple(row["task"] for row in definitions) == TASKS, "original native subset changed")
    for value in (entry, report, final, proof):
        _public_fields(value)
        _privacy(value)
    _require(report.get("schema") == SCHEMA and report.get("family") == "atlas"
             and report.get("publication_scope") == "native3_repaired"
             and type(report.get("required")) is int and report["required"] == 3
             and type(report.get("original_required")) is int and report["original_required"] == 19,
             "separate3/original19 scope changed")
    source = report.get("source", {})
    _require(set(source) == {"origin_commit", "digest"}
             and isinstance(source.get("origin_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", source["origin_commit"])
             and isinstance(source.get("digest"), str) and HEX.fullmatch(source["digest"])
             and source == entry.get("source") and source["origin_commit"] != PARENT_SOURCE["origin_commit"]
             and source["digest"] != PARENT_SOURCE["digest"]
             and source["origin_commit"] != PREDECESSOR_SOURCE["origin_commit"]
             and source["digest"] != PREDECESSOR_SOURCE["digest"], "new-source native3 identity changed or borrowed")
    _require(all(report.get("claims", {}).get(key) is False for key in CLAIMS), "native3 grants forbidden pooled/default/speed credit")
    _require(report.get("accepted_native3_status") == "PENDING_PUBLICATION_COST_FINALIZATION",
             "immutable native3 cut status changed")
    anchor = _parent(report, entry, root)
    predecessor = _predecessor(root, entry, report, anchor)
    rows = report.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == 3 and tuple(row.get("id") for row in rows) == IDS,
             "three exact fresh ordered case IDs required")
    cells, pins, accepted_values, recorded_values = [], [], [], []
    reached_stop = False
    for row, declared in zip(rows, definitions):
        definition = declared["original_definition"]
        _require(row.get("group") == "native" and row.get("task") == declared["task"]
                 and row.get("parent_full19_retest_id") == declared["id"]
                 and row.get("parent_original_id") == definition["id"], "original native host mapping changed")
        for key in ("original_requirements", "observation_steps"):
            _require(_same(row.get(key), definition[key]), "native gates or34-clock schedule changed")
        _require(_same(row.get("required_host"), definition["original_host"])
                 and _same(row.get("complete_recipe"), declared["resolved_recipe"])
                 and _same(row.get("observer"), declared["observer"]), "complete original Recipe/host/observer changed")
        status = row.get("execution_status")
        _require(status in STATUS_KEYS and type(row.get("full_protocol_complete")) is bool
                 and type(row.get("recorded_full_protocol_complete")) is bool, "unknown execution/recorded status")
        gate = row.get("accepted_original_gate")
        accepted = gate in {"PASS", "FAIL"}
        _require((gate is None or accepted) and row["full_protocol_complete"] is accepted,
                 "accepted numerical scope changed")
        if accepted:
            _require(not reached_stop and status == gate and row.get("acceptance_status") == "ACCEPTED"
                     and row["recorded_full_protocol_complete"] is True, "accepted result after stopped slot or partial evidence")
            _require(type(row.get("completed_steps")) is int and row["completed_steps"] == 7000
                     and type(row.get("metric_observations")) is int and row["metric_observations"] == 34
                     and _same(row.get("terminal_reads"), [6000, 6250, 6500, 6750, 7000])
                     and type(row.get("independent_holdout_samples")) is int
                     and row["independent_holdout_samples"] == 100000, "accepted native gate lacks original full reads/holdout")
        else:
            _require(row.get("acceptance_status") in {"UNAVAILABLE", "UNAVAILABLE_BUDGET_EXCEEDED"},
                     "unaccepted gate must remain unavailable")
            reached_stop = True
        gates = row.get("native_gates")
        if gates is None and not accepted:
            gates = {}
        _require(isinstance(gates, dict), "native gates must be a mapping or an unavailable null")
        if accepted:
            _require(isinstance(gates, dict) and set(gates) == {"noisy", "clean"}
                     and all(isinstance(value, dict) and set(value) == {"coverage", "accuracy"}
                             and set(value.values()) <= {"PASS", "FAIL"} for value in gates.values()),
                     "accepted full native protocol requires both noisy and clean receipts")
        noisy = gates.get("noisy", {})
        _require(isinstance(noisy, dict), "native noisy gates must be a mapping")
        joint = None
        if set(noisy) == {"coverage", "accuracy"} and set(noisy.values()) <= {"PASS", "FAIL"}:
            joint = "PASS" if all(value == "PASS" for value in noisy.values()) else "FAIL"
        if accepted:
            _require(joint == gate and row.get("raw_reported_status") == noisy["accuracy"],
                     "accuracy-only raw status versus accepted joint gate changed")
            accepted_values.append(gate)
        if row["recorded_full_protocol_complete"] and joint is not None:
            recorded_values.append(joint)
        media = row.get("media")
        if accepted:
            _require(isinstance(media, dict) and media.get("file") == "gifs/native-" + declared["task"] + ".gif"
                     and media.get("original_gate") == gate and type(media.get("frames")) is int
                     and media["frames"] == 9 and _same(media.get("actual_steps"), declared["media_steps"]),
                     "accepted original native goal media scope changed")
            directory = entry["result"]["path"].rsplit("/", 1)[0]
            pin = {"path": directory + "/" + media["file"], "sha256": media["sha256"], "bytes": media["bytes"]}
            _require(_read_pin(root, pin)[:6] in {b"GIF87a", b"GIF89a"}, "original goal GIF bytes required")
            pins.append(pin)
        else:
            _require(media is None, "unavailable gate cannot acquire accepted media credit")
        row_cost = _row_cost(row, declared["proposed_inclusive_allowance_seconds"], accepted)
        cells.append({"task_id": row["id"], "parent_original_id": row["parent_original_id"],
                      "gate_status": gate if accepted else (status if status not in {"PASS", "FAIL"} else "INCOMPLETE"),
                      "execution_status": status, "original_gate": gate, "raw_status": row.get("raw_reported_status"),
                      "metrics": deepcopy(row.get("final_metrics", {})), "cost": row_cost,
                      "recipe_sha256": stable_hash(row["complete_recipe"]),
                      "original_requirements": deepcopy(row["original_requirements"]),
                      "sampling": deepcopy(row["observer"]), "media": deepcopy(media)})
    counts = dict(sorted(Counter(row["execution_status"] for row in rows).items()))
    _require(type(report.get("counts")) is dict and set(report["counts"]) == STATUS_KEYS
             and all(type(n) is int and n >= 0 for n in report["counts"].values())
             and counts == {key: val for key, val in report["counts"].items() if val}, "three-cell execution counts changed")
    completed = len(accepted_values)
    accepted_counts = {"PASS": accepted_values.count("PASS"), "FAIL": accepted_values.count("FAIL"), "UNAVAILABLE": 3 - completed}
    _require(type(report.get("completed")) is int and report["completed"] == completed
             and report.get("accepted_original_gate_counts") == accepted_counts
             and all(type(n) is int for n in report["accepted_original_gate_counts"].values())
             and report.get("all_required_case_evidence_complete") is (completed == 3), "accepted three-case count changed")
    raw_gate = ("PASS" if all(value == "PASS" for value in recorded_values) else "FAIL") if len(recorded_values) == 3 else "UNAVAILABLE"
    _require(report.get("raw_native3_protocol_gate") == raw_gate, "raw versus accepted native3 grade changed")
    _require(entry["media"] == pins, "all accepted native goal GIFs must be pinned")
    card = report.get("trusted_terminal_card", {})
    _require(isinstance(card.get("sha256"), str) and HEX.fullmatch(card["sha256"])
             and card["sha256"] == entry.get("terminal_card_sha256"), "trusted native3 terminal card changed")
    _require(final.get("schema") == FINAL_SCHEMA and final.get("original_terminal_cut_results_sha256") == entry["result"]["sha256"]
             and final.get("trusted_terminal_card_sha256") == card["sha256"]
             and final.get("raw_native3_protocol_gate") == raw_gate
             and all(final.get(key) is True for key in ("original_case_verdicts_unchanged", "old_case_debit_once", "pretraining_invalid_debit_once", "cumulative_metadata_once"))
             and all(final.get(key) is False for key in ("old_grades_are_current_credit", "full_new_source_original19_credit")),
             "native3 final cost/result/card/scope joins changed")
    _require(_same(final.get("predecessor_cost"), PREDECESSOR_COST)
             and final.get("first_invalid_anchor_sha256") == PREDECESSOR_ANCHOR_SHA
             and final.get("predecessor_closed_metadata_sha256") == PREDECESSOR_METADATA_SHA
             and type(final.get("preserved_metadata_prefix_phases")) is int
             and final["preserved_metadata_prefix_phases"] == 22,
             "final cost-only predecessor or complete22 history changed")
    before_history = report.get("metadata_phase_history_before_publication")
    final_history = final.get("metadata_phase_history")
    _phases_v2(before_history, anchor, predecessor)
    _phases_v2(final_history, anchor, predecessor)
    _require(type(report.get("metadata_phase_count_before_publication")) is int
             and report["metadata_phase_count_before_publication"] == len(before_history)
             and type(final.get("metadata_phase_count")) is int and final["metadata_phase_count"] == len(final_history)
             and len(final_history) > len(before_history)
             and _same(final_history[:len(before_history)], before_history), "final cumulative metadata history rewound")
    _accounting_v2(report["cost_snapshot_before_publication"], rows, before_history, anchor, predecessor)
    costs = deepcopy(final["costs"])
    halt = _accounting_v2(costs, rows, final_history, anchor, predecessor)
    status = ("PASS" if accepted_counts["PASS"] == 3 else "FAIL") if completed == 3 and not halt else "INCOMPLETE"
    _require(final.get("status") == status, "final native3 accepted status changed")
    ledger = final.get("authoritative_final_metadata_ledger", {})
    _require(isinstance(ledger.get("sha256"), str) and HEX.fullmatch(ledger["sha256"])
             and type(ledger.get("bytes")) is int and ledger["bytes"] >= 0
             and ledger.get("path") == report.get("authoritative_metadata_ledger_path") == CANONICAL_LEDGER,
             "same final cumulative ledger identity changed")
    closed = final.get("consumed_closed_metadata_ledger", {})
    closed_path = closed.get("path")
    _require(set(closed) == {"path", "sha256", "bytes"} and isinstance(closed_path, str)
             and PurePosixPath(closed_path).is_absolute() and ".." not in PurePosixPath(closed_path).parts
             and closed["sha256"] == ledger["sha256"] and type(closed["bytes"]) is int
             and closed["bytes"] == ledger["bytes"]
             and closed_path != CANONICAL_LEDGER and final.get("ledger_input_is_copy") is True,
             "trusted closed ledger input versus canonical identity changed")
    _require(proof.get("schema") == VERIFICATION_SCHEMA and proof.get("status") == "VERIFIED_PRE_PUBLICATION_COST_CUT"
             and proof.get("results_sha256") == entry["result"]["sha256"] and proof.get("trusted_card_sha256") == card["sha256"]
             and _same(proof.get("counts"), report["counts"]) and _same(proof.get("accepted_original_gate_counts"), accepted_counts)
             and proof.get("required") == 3 and proof.get("original_required") == 19
             and proof.get("accepted_original_gifs") == completed and proof.get("final_cost_pending") is True,
             "passive native3 verification join changed")
    _require(proof.get("source") == source
             and proof.get("first_invalid_anchor_sha256") == PREDECESSOR_ANCHOR_SHA
             and proof.get("predecessor_closed_metadata_sha256") == PREDECESSOR_METADATA_SHA
             and type(proof.get("preserved_metadata_prefix_phases")) is int
             and proof["preserved_metadata_prefix_phases"] == 22
             and proof.get("predecessor_cost_only_verified") is True
             and proof.get("request_boundary_control_verified") is True,
             "V2 source/predecessor/current-request verification joins changed")
    _require(all(type(proof.get(key)) is int for key in ("required", "original_required", "accepted_original_gifs")),
             "typed passive count required")
    for key in ("models", "draws", "scorer_calls", "grader_calls", "rendered_frames"):
        _require(type(proof.get(key)) is int and proof[key] == 0, "passive projection cannot perform scientific work")
    producer = report.get("producer", {})
    _require(isinstance(producer.get("sha256"), str) and HEX.fullmatch(producer["sha256"])
             and proof.get("producer_sha256") == producer["sha256"], "frozen passive producer binding changed")
    for key in ("models", "draws", "scorer_calls", "regrades", "rendered_frames"):
        _require(type(producer.get(key)) is int and producer[key] == 0, "passive producer cannot perform scientific work")
    _require(proof.get("private_nonce_values_copied") is False and proof.get("bulk_state_arrays_logs_copied") is False,
             "public proof privacy/copy boundary changed")
    return {"id": entry["id"], "schema": SCHEMA, "source": deepcopy(source), "required": 3, "original_required": 19,
            "completed": completed, "counts": counts, "accepted_counts": accepted_counts, "status": status,
            "raw_native3_protocol_gate": raw_gate, "cost": costs, "cells": cells,
            "runtime_declaration": deepcopy(protocol["runtime"]), "actual_runtime_in_compact_result": False,
            "recipe": deepcopy(protocol["common_resolved_recipe"]), "prior_reference": deepcopy(report["prior_reference"]),
            "predecessor_cost": deepcopy(report["predecessor_cost"]),
            "publication": deepcopy(entry["result"]), "final_cost": deepcopy(entry["final_cost"]),
            "authoritative_final_metadata_ledger": deepcopy(ledger),
            "consumed_closed_metadata_ledger": deepcopy(closed),
            "terminal_card_sha256": card["sha256"], "readout": entry["readout"]["path"],
            "inputs": [deepcopy(entry[key]) for key in ("parent_metadata_anchor", "predecessor_anchor", "predecessor_metadata_anchor")],
            "qualification_input": False, "qualification_reuse": False, "default_adoption": False, "speed_ranking": False,
            "cost_scope": "Original19 and pretraining-invalid case debits once, plus new native3 costs and SAME cumulative metadata once; never sum overlapping cuts."}


def recall_record(root, cut):
    """One repaired-source recall; V1 and full19 remain separate evidence."""
    from .publication_memory import _base

    source = {**cut["publication"], "board": "reports/forge/technique-inventory.md"}
    record = _base(source, "pr223-native3-repaired-continuation", "atlas",
                   {"publication": cut["publication"], "source": cut["source"]},
                   "discriminator_stability", "published_study")
    record.update(
        lifecycle="concluded" if cut["completed"] == 3 else "closed_partial_cut",
        status=cut["status"], trainer_family="atlas", cohort=cut["source"]["digest"],
        mechanism_class="faithful_original_native3_repaired_continuation", configuration_id=cut["id"],
        required=3, original_required=19,
        settings={"lr": .00425, "prior_lr_mult": 2., "d_lr_mult": 1.},
        task_results=deepcopy(cut["cells"]), recorded_cost=deepcopy(cut["cost"]),
        prior_reference=deepcopy(cut["prior_reference"]),
        predecessor_cost=deepcopy(cut["predecessor_cost"]),
        provenance={"source_commit": cut["source"]["origin_commit"], "execution_digest": cut["source"]["digest"],
                    "final_cost": deepcopy(cut["final_cost"]), "terminal_card_sha256": cut["terminal_card_sha256"]},
        next_action="Read the repaired three-case cut and authoritative SAME-ledger FINAL_COST. The original 16 "
                    "remain old-source evidence; no fresh single-source 19, current 26, default or speed credit.",
        conclusion=f"Repaired native3 continuation: {cut['accepted_counts']['PASS']}/3 PASS, "
                   f"{cut['accepted_counts']['FAIL']} FAIL, {cut['accepted_counts']['UNAVAILABLE']} unavailable; "
                   f"final accepted {cut['status']}. Original19 parent remains 16 PASS/1 INVALID/2 NOT_RUN "
                   "under its own source. Current cost includes original19 and pretraining-invalid debits separately once and SAME cumulative metadata once.",
    )
    return record


"""Passive native3 recall, separate from the immutable original19 cuts.

Prospective contract: pg_pr223_native3_passive_publication_v1. Only committed
JSON and GIF bytes are consumed; execution/source/ledger paths stay inert.
No scientific module, original receipt, checkpoint or evaluator is imported.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import PurePosixPath
import re

from .contracts import stable_hash

SCHEMA = "pg_pr223_native3_passive_publication_v1"
FINAL_SCHEMA = "pg_pr223_native3_final_publication_cost_v1"
VERIFICATION_SCHEMA = "pg_pr223_native3_passive_verification_v1"
TASKS = ("grid100", "rotated100", "staggered100")
IDS = tuple("pr223-native3-continuation-v1-native-" + task for task in TASKS)
PARENT_SOURCE = {
    "origin_commit": "2068a661331a45e0e283b0362dd58b7d92263c6f",
    "digest": "00cadbfd06c770e16c35045b42387932b14ee3751954e5320f9e2c32799f7a2f",
}
PARENT_DIRECTORY = "reports/forge/pr223-original-full-retest-stopped17-20261004"
PARENT_FILES = {
    PARENT_DIRECTORY + "/results.json": "4cca29c719ffe295712775be27167d7465090c40865d3b096aea247e61a3d4f3",
    PARENT_DIRECTORY + "/FINAL_COST.json": "2335c0da59f1155a329f72390078d1480f253ce1e17a5879a52138d53e3661d3",
    PARENT_DIRECTORY + "/verification.json": "a74262e6af196c7686f9275a0723ace989067f65552fb8b554a2c35948c26eb5",
}
PARENT_CASE_CHARGE = 3165.841891122982
PARENT_METADATA_PAID = 44.37232269323431
PARENT_METADATA_PATH = "reports/forge/pr223-native3-continuation-20261004/inputs/closed-parent-metadata.json"
PARENT_METADATA_PIN_PATH = "reports/forge/pr223-native3-continuation-20261004/fixtures/closed-parent-metadata.json"
PARENT_METADATA_SHA = "ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369"
CANONICAL_LEDGER = "/ml2/hypergan/.pg-pr223-full-original-retest-20261004.pr223-full19-metadata-cost.json"
STATUS_KEYS = {"PASS", "FAIL", "INVALID", "INCOMPLETE", "BUDGET_EXCEEDED", "BLOCKED", "NOT_RUN", "UNKNOWN"}
CLAIMS = ("full_new_source_original19_credit", "old_grades_are_current_credit", "current26_qualification",
          "default_adoption", "speed_ranking", "named10500_cost_pooling", "clean_or_forced_ema_primary_credit")


def _require(condition, message):
    if not condition:
        raise ValueError("native3 passive publication: " + message)


def _same(actual, expected):
    return stable_hash(actual) == stable_hash(expected)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "finite nonnegative paid/reserved cost required")
    return value


def _cost(actual, expected):
    _require(math.isclose(_number(actual), expected, rel_tol=0, abs_tol=1e-8), "cost arithmetic changed")


def _privacy(value):
    if isinstance(value, dict):
        for key, child in value.items():
            _require(key not in {"token", "attempt_token", "nonce", "attempt_nonce", "lease_fd", "lease_fds",
                                 "credential", "credentials", "access_token", "api_key", "private_nonce_value"},
                     "private execution field cannot enter native3 recall")
            if key in {"qualification_input", "qualification_reuse", "default_adoption", "speed_ranking",
                       "current26_qualification", "full_new_source_original19_credit", "old_grades_are_current_credit",
                       "winner", "shipping_default", "fastest"}:
                _require(child is False, "native3 cannot acquire qualification or old-source credit")
            if key in {"token_sha256", "nonce_sha256"}:
                _require(isinstance(child, str) and re.fullmatch(r"[0-9a-f]{64}", child), "hashed nonce identity required")
            _privacy(child)
    elif isinstance(value, list):
        for child in value:
            _privacy(child)


def _phases(history, anchor):
    _require(isinstance(history, list) and len(history) >= 12
             and _same(history[:12], anchor["phases"]), "same-ledger parent12 phase prefix changed")
    for index, phase in enumerate(history):
        _require(isinstance(phase, dict) and set(phase) == {
            "index", "name", "status", "paid_wall_seconds", "paused_wall_seconds"}, "metadata phase shape changed")
        _require(type(phase["index"]) is int and phase["index"] == index
                 and isinstance(phase["name"], str) and phase["name"].strip()
                 and phase["status"] in {"COMPLETE", "ERROR", "INTERRUPTED", "BUDGET_EXCEEDED"},
                 "closed metadata phase identity changed")
        _number(phase["paid_wall_seconds"])
        _number(phase["paused_wall_seconds"])


def _parent(report, entry, root):
    from .completed_studies import _json, _read_pin

    prior = report.get("prior_reference", {})
    _require(prior.get("source") == PARENT_SOURCE and type(prior.get("required")) is int
             and prior["required"] == 19 and prior.get("files_sha256") == PARENT_FILES,
             "immutable stopped17 parent source/report pins changed")
    _require(_same(prior.get("execution_counts"), {"PASS": 16, "INVALID": 1, "NOT_RUN": 2})
             and _same(prior.get("accepted_original_gate_counts"), {"PASS": 16, "FAIL": 0, "UNAVAILABLE": 3})
             and prior.get("old_grades_are_current_credit") is False
             and prior.get("original_evidence_unchanged") is True, "old16 or failed-grid evidence borrowed or rewritten")
    _cost(prior.get("case_charged_seconds"), PARENT_CASE_CHARGE)
    pin = entry.get("parent_metadata_anchor")
    _require(pin == {"path": PARENT_METADATA_PIN_PATH, "sha256": PARENT_METADATA_SHA, "bytes": 2675},
             "immutable parent metadata anchor changed")
    anchor = _json(_read_pin(root, pin))
    _require(anchor.get("schema") == "pg_pr223_original_full_retest_metadata_budget_v1"
             and anchor.get("cap_seconds") == 180 and anchor.get("aggregate_cap_seconds") == 10800
             and anchor.get("current_phase") is None and anchor.get("blocked") is False
             and len(anchor.get("phases", [])) == 12, "closed parent metadata boundary changed")
    _cost(anchor.get("paid_wall_seconds"), PARENT_METADATA_PAID)
    _cost(anchor.get("charged_seconds"), PARENT_METADATA_PAID)
    _cost(anchor.get("reserved_seconds"), 0)
    _cost(anchor.get("overrun_seconds"), 0)
    declared = report.get("parent_metadata_anchor", {})
    _require(declared.get("sha256") == PARENT_METADATA_SHA and declared.get("bytes") == 2675
             and declared.get("phases") == 12 and declared.get("snapshot_relative") == PARENT_METADATA_PATH,
             "reported parent metadata identity changed")
    return anchor


def _row_cost(row, allowance, accepted):
    value = row.get("cost", {})
    _cost(value.get("allowance_seconds"), allowance)
    for key in ("paid_wall_seconds", "reserved_seconds", "charged_seconds", "overrun_seconds"):
        _number(value.get(key))
    paid, reserved = value["paid_wall_seconds"], value["reserved_seconds"]
    _cost(value["charged_seconds"], paid + reserved)
    _cost(value.get("unmeasured_interrupt_reserved_seconds"), reserved)
    _cost(value["overrun_seconds"], max(0, paid - allowance))
    terminal = value.get("terminal_status")
    _require(terminal in {"completed", "missing", "error", "timeout", "cancelled"}
             and value.get("completed_terminal") is (terminal == "completed")
             and type(value.get("certified")) is bool, "durable completion identity changed")
    _require(not value["certified"] or terminal == "completed", "interrupted terminal cannot be certified")
    if row["execution_status"] in {"NOT_RUN", "UNKNOWN", "BLOCKED"}:
        _cost(paid + reserved, 0)
        _require(terminal == "missing" and value["certified"] is False, "unattempted cell cannot claim a terminal")
    elif terminal == "completed":
        _cost(reserved, 0)
    else:
        _cost(reserved, max(0, allowance - paid))
    if accepted:
        _require(terminal == "completed" and value["certified"] is True and paid > 0,
                 "accepted gate lacks completed certified durable cost")
        _cost(value["overrun_seconds"], 0)
    return deepcopy(value)


def _accounting(costs, rows, history, anchor):
    for key, expected in (("aggregate_cap_seconds", 10800), ("metadata_cap_seconds", 180),
                          ("case_caps_sum_seconds", 4290), ("export_grace_seconds", 0), ("retries", 0),
                          ("prior_case_charged_seconds", PARENT_CASE_CHARGE)):
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
    _require(metadata["paid_wall_seconds"] >= PARENT_METADATA_PAID, "cumulative metadata reset")
    _phases(history, anchor)
    _cost(metadata["paid_wall_seconds"], math.fsum(phase["paid_wall_seconds"] for phase in history))
    charge = PARENT_CASE_CHARGE + paid + reserved + metadata["charged_seconds"]
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


def project_native3(root, entry, report, final, proof):
    """Project recorded native3 flags; never borrow old16 or score samples."""
    from .completed_studies import HEX, _json, _read_pin
    from .publication_memory import PR223_PROTOCOL, PR223_PROTOCOL_SHA, _public_fields

    _require(entry["protocol"] == {"path": PR223_PROTOCOL, "sha256": PR223_PROTOCOL_SHA, "bytes": 157526},
             "complete original19 protocol pin changed")
    protocol = _json(_read_pin(root, entry["protocol"]))
    definitions = [row for row in protocol["rows"] if row["group"] == "native"]
    _require(len(definitions) == 3 and tuple(row["task"] for row in definitions) == TASKS, "original native subset changed")
    for value in (report, final, proof):
        _public_fields(value)
        _privacy(value)
    _require(report.get("schema") == SCHEMA and report.get("family") == "atlas"
             and report.get("publication_scope") == "native3"
             and type(report.get("required")) is int and report["required"] == 3
             and type(report.get("original_required")) is int and report["original_required"] == 19,
             "separate3/original19 scope changed")
    source = report.get("source", {})
    _require(set(source) == {"origin_commit", "digest"}
             and isinstance(source.get("origin_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", source["origin_commit"])
             and isinstance(source.get("digest"), str) and HEX.fullmatch(source["digest"])
             and source == entry.get("source") and source["origin_commit"] != PARENT_SOURCE["origin_commit"]
             and source["digest"] != PARENT_SOURCE["digest"], "new-source native3 identity changed or borrowed")
    _require(all(report.get("claims", {}).get(key) is False for key in CLAIMS), "native3 grants forbidden pooled/default/speed credit")
    _require(report.get("accepted_native3_status") == "PENDING_PUBLICATION_COST_FINALIZATION",
             "immutable native3 cut status changed")
    anchor = _parent(report, entry, root)
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
             and all(final.get(key) is True for key in ("original_case_verdicts_unchanged", "old_case_debit_once", "cumulative_metadata_once"))
             and all(final.get(key) is False for key in ("old_grades_are_current_credit", "full_new_source_original19_credit")),
             "native3 final cost/result/card/scope joins changed")
    before_history = report.get("metadata_phase_history_before_publication")
    final_history = final.get("metadata_phase_history")
    _phases(before_history, anchor)
    _phases(final_history, anchor)
    _require(type(report.get("metadata_phase_count_before_publication")) is int
             and report["metadata_phase_count_before_publication"] == len(before_history)
             and type(final.get("metadata_phase_count")) is int and final["metadata_phase_count"] == len(final_history)
             and len(final_history) > len(before_history)
             and _same(final_history[:len(before_history)], before_history), "final cumulative metadata history rewound")
    _accounting(report["cost_snapshot_before_publication"], rows, before_history, anchor)
    costs = deepcopy(final["costs"])
    halt = _accounting(costs, rows, final_history, anchor)
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
             and final.get("ledger_input_is_copy") is (closed_path != CANONICAL_LEDGER),
             "trusted closed ledger input versus canonical identity changed")
    _require(proof.get("schema") == VERIFICATION_SCHEMA and proof.get("status") == "VERIFIED_PRE_PUBLICATION_COST_CUT"
             and proof.get("results_sha256") == entry["result"]["sha256"] and proof.get("trusted_card_sha256") == card["sha256"]
             and _same(proof.get("counts"), report["counts"]) and _same(proof.get("accepted_original_gate_counts"), accepted_counts)
             and proof.get("required") == 3 and proof.get("original_required") == 19
             and proof.get("accepted_original_gifs") == completed and proof.get("final_cost_pending") is True,
             "passive native3 verification join changed")
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
            "publication": deepcopy(entry["result"]), "final_cost": deepcopy(entry["final_cost"]),
            "authoritative_final_metadata_ledger": deepcopy(ledger),
            "consumed_closed_metadata_ledger": deepcopy(closed),
            "terminal_card_sha256": card["sha256"], "readout": entry["readout"]["path"],
            "inputs": [deepcopy(entry["parent_metadata_anchor"])],
            "qualification_input": False, "qualification_reuse": False, "default_adoption": False, "speed_ranking": False,
            "cost_scope": "Old case debit once plus new native3 costs and SAME cumulative metadata once; do not sum overlapping cuts."}


def recall_record(root, cut):
    """One three-case recall record; no trial or nineteen-case substitution."""
    from .publication_memory import _base

    source = {**cut["publication"], "board": "reports/forge/technique-inventory.md"}
    record = _base(source, "pr223-native3-continuation", "atlas",
                   {"publication": cut["publication"], "source": cut["source"]},
                   "discriminator_stability", "published_study")
    record.update(
        lifecycle="concluded" if cut["completed"] == 3 else "closed_partial_cut",
        status=cut["status"], trainer_family="atlas", cohort=cut["source"]["digest"],
        mechanism_class="faithful_original_native3_continuation", configuration_id=cut["id"],
        required=3, original_required=19,
        settings={"lr": .00425, "prior_lr_mult": 2., "d_lr_mult": 1.},
        task_results=deepcopy(cut["cells"]), recorded_cost=deepcopy(cut["cost"]),
        prior_reference=deepcopy(cut["prior_reference"]),
        provenance={"source_commit": cut["source"]["origin_commit"], "execution_digest": cut["source"]["digest"],
                    "final_cost": deepcopy(cut["final_cost"]), "terminal_card_sha256": cut["terminal_card_sha256"]},
        next_action="Read the three-case continuation and authoritative SAME-ledger FINAL_COST. The original 16 "
                    "remain old-source evidence; no fresh single-source 19, current 26, default or speed credit.",
        conclusion=f"Native3 continuation: {cut['accepted_counts']['PASS']}/3 PASS, "
                   f"{cut['accepted_counts']['FAIL']} FAIL, {cut['accepted_counts']['UNAVAILABLE']} unavailable; "
                   f"final accepted {cut['status']}. Original19 parent remains 16 PASS/1 INVALID/2 NOT_RUN "
                   "under its own source. Current cost includes prior case debit once and cumulative metadata once.",
    )
    return record

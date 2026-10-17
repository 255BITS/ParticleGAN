"""Private repaired-native3 metadata; never actual reports, scoring or science."""
from collections import Counter
from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

from experiments.forge import native3_publication_memory as v1
from experiments.forge import native3_repaired_publication_memory as repaired
from experiments.forge import publication_memory as memory
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
CONTRACT_SHA = "100c6a237728a4e02d1d8d5607baccb11f64de283a585a6474b52a76a3d63831"
INTERFACE_SHA = "e8050beb00dae6b86013af50b3e652ed155b0cd88740f45f2a832a9ac6663e39"
# Nine inert one-pixel GIF frames, never a model illustration or metric input.
SYNTHETIC_GIF = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff"
    + b"\x21\xf9\x04\x00\x01\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00" * 9
    + b"\x3b")


def fixture(name, sha):
    data = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha
    return json.loads(data)


def original_protocol():
    return fixture("pr223_native3_repaired_interface.json", INTERFACE_SHA)["protocol"]["original19_catalog"]


def function_hash(path, name):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256("".join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno]).encode()).hexdigest()


class PrivateCut:
    def __init__(self, root, *, completed=3, coverage_failure=None, invalid=None, timeout=False,
                 version="v2", name=None):
        self.root = root
        self.adapter = repaired if version == "v2" else v1
        self.version = version
        self.directory = "reports/forge/" + (name or "synthetic-native3-" + version)
        protocol = original_protocol()
        atomic_json(root / memory.PR223_PROTOCOL, protocol)
        assert file_hash(root / memory.PR223_PROTOCOL) == memory.PR223_PROTOCOL_SHA
        parent_bytes = (FIXTURES / "pr223_native3_parent_metadata.json").read_bytes()
        assert hashlib.sha256(parent_bytes).hexdigest() == v1.PARENT_METADATA_SHA
        target = root / v1.PARENT_METADATA_PIN_PATH
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(parent_bytes)
        parent = json.loads(parent_bytes)
        predecessor = fixture("pr223_native3_predecessor_metadata.json", repaired.PREDECESSOR_METADATA_SHA)
        for relative, source, sha in (
            (repaired.PREDECESSOR_ANCHOR_PATH, "pr223_native3_first_invalid_anchor.json", repaired.PREDECESSOR_ANCHOR_SHA),
            (repaired.PREDECESSOR_METADATA_PATH, "pr223_native3_predecessor_metadata.json", repaired.PREDECESSOR_METADATA_SHA),
        ):
            data = (FIXTURES / source).read_bytes(); assert hashlib.sha256(data).hexdigest() == sha
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        rows = []
        for index, declared in enumerate(row for row in protocol["rows"] if row["group"] == "native"):
            definition = declared["original_definition"]
            accepted = index < completed
            status = ("FAIL" if index == coverage_failure else "PASS") if accepted else "NOT_RUN"
            if index == invalid: accepted, status = False, "INCOMPLETE" if timeout else "INVALID"
            paid = 1. if accepted or index == invalid else 0.
            allowance = declared["proposed_inclusive_allowance_seconds"]
            terminal = ("timeout" if timeout else "completed") if index == invalid else ("completed" if accepted else "missing")
            reserved = allowance - paid if index == invalid and timeout else 0.
            row = {
                "id": repaired.IDS[index], "task": declared["task"], "group": "native",
                "parent_full19_retest_id": declared["id"], "parent_original_id": definition["id"],
                "required_host": deepcopy(definition["original_host"]), "complete_recipe": deepcopy(declared["resolved_recipe"]),
                "observer": deepcopy(declared["observer"]), "original_requirements": deepcopy(definition["original_requirements"]),
                "observation_steps": deepcopy(definition["observation_steps"]), "execution_status": status,
                "acceptance_status": "ACCEPTED" if accepted else "UNAVAILABLE",
                "accepted_original_gate": status if accepted else None,
                "raw_reported_status": "PASS" if accepted else ("ERROR" if index == invalid else None),
                "full_protocol_complete": accepted, "recorded_full_protocol_complete": accepted,
                "native_gates": None, "media": None,
                "cost": {"allowance_seconds": allowance, "paid_wall_seconds": paid,
                    "reserved_seconds": reserved, "unmeasured_interrupt_reserved_seconds": reserved,
                    "charged_seconds": paid + reserved, "overrun_seconds": 0.,
                    "terminal_status": terminal, "completed_terminal": terminal == "completed", "certified": accepted},
            }
            if accepted:
                row.update(completed_steps=7000, metric_observations=34,
                    terminal_reads=[6000, 6250, 6500, 6750, 7000], independent_holdout_samples=100000,
                    final_metrics={"synthetic_flag": status},
                    native_gates={"noisy": {"coverage": "FAIL" if index == coverage_failure else "PASS", "accuracy": "PASS"},
                                  "clean": {"coverage": "FAIL", "accuracy": "FAIL"}})
                filename = "gifs/native-" + declared["task"] + ".gif"
                media = root / self.directory / filename
                media.parent.mkdir(parents=True, exist_ok=True); media.write_bytes(SYNTHETIC_GIF)
                row["media"] = {"file": filename, "sha256": file_hash(media), "bytes": len(SYNTHETIC_GIF),
                                "frames": 9, "actual_steps": deepcopy(declared["media_steps"]), "original_gate": status}
            rows.append(row)
        counts = {key: Counter(row["execution_status"] for row in rows).get(key, 0) for key in v1.STATUS_KEYS}
        completed = sum(row["full_protocol_complete"] for row in rows)
        accepted_counts = {"PASS": sum(row["accepted_original_gate"] == "PASS" for row in rows),
                           "FAIL": sum(row["accepted_original_gate"] == "FAIL" for row in rows), "UNAVAILABLE": 3 - completed}
        raw = ("FAIL" if coverage_failure is not None else "PASS") if completed == 3 else "UNAVAILABLE"
        before = deepcopy(predecessor["phases"] if version == "v2" else parent["phases"])
        before.append({"index": len(before), "name": "synthetic_metadata_cut", "paid_wall_seconds": 1.,
                       "paused_wall_seconds": 0., "status": "COMPLETE"})
        after = deepcopy(before) + [{"index": len(before), "name": "synthetic_publication", "paid_wall_seconds": 1.,
                                    "paused_wall_seconds": 0., "status": "COMPLETE"}]
        source = {"origin_commit": ("f" if version == "v2" else "a") * 40,
                  "digest": ("a" if version == "v2" else "b") * 64}
        self.report = {
            "schema": self.adapter.SCHEMA, "publication_scope": "native3_repaired" if version == "v2" else "native3",
            "family": "atlas", "required": 3, "original_required": 19, "completed": completed, "source": source,
            "rows": rows, "counts": counts, "accepted_original_gate_counts": accepted_counts,
            "all_required_case_evidence_complete": completed == 3, "raw_native3_protocol_gate": raw,
            "accepted_native3_status": "PENDING_PUBLICATION_COST_FINALIZATION", "synthetic_only": True,
            "trusted_terminal_card": {"sha256": "c" * 64, "bytes": 100, "path": "/not-consumed/raw/card.json"},
            "cost_snapshot_before_publication": {}, "metadata_phase_history_before_publication": before,
            "metadata_phase_count_before_publication": len(before), "authoritative_metadata_ledger_path": v1.CANONICAL_LEDGER,
            "parent_metadata_anchor": {"sha256": v1.PARENT_METADATA_SHA, "bytes": 2675, "phases": 12,
                                       "snapshot_relative": v1.PARENT_METADATA_PATH},
            "prior_reference": {"source": deepcopy(v1.PARENT_SOURCE), "required": 19,
                "files_sha256": deepcopy(v1.PARENT_FILES), "case_charged_seconds": v1.PARENT_CASE_CHARGE,
                "execution_counts": {"PASS": 16, "INVALID": 1, "NOT_RUN": 2},
                "accepted_original_gate_counts": {"PASS": 16, "FAIL": 0, "UNAVAILABLE": 3},
                "old_grades_are_current_credit": False, "original_evidence_unchanged": True},
            "producer": {"sha256": "e" * 64, **{key: 0 for key in ("models", "draws", "scorer_calls", "regrades", "rendered_frames")}},
            "claims": {key: False for key in v1.CLAIMS},
        }
        self.final = {
            "schema": self.adapter.FINAL_SCHEMA, "costs": {}, "raw_native3_protocol_gate": raw,
            "status": raw if completed == 3 else "INCOMPLETE", "metadata_phase_count": len(after), "metadata_phase_history": after,
            "trusted_terminal_card_sha256": "c" * 64,
            "authoritative_final_metadata_ledger": {"path": v1.CANONICAL_LEDGER, "sha256": "d" * 64, "bytes": 100},
            "consumed_closed_metadata_ledger": {"path": "/not-consumed/closed-ledger.json", "sha256": "d" * 64, "bytes": 100},
            "ledger_input_is_copy": True, "original_case_verdicts_unchanged": True, "old_case_debit_once": True,
            "cumulative_metadata_once": True, "old_grades_are_current_credit": False, "full_new_source_original19_credit": False,
        }
        self.proof = {
            "schema": self.adapter.VERIFICATION_SCHEMA, "status": "VERIFIED_PRE_PUBLICATION_COST_CUT",
            "required": 3, "original_required": 19, "counts": deepcopy(counts), "accepted_original_gate_counts": deepcopy(accepted_counts),
            "accepted_original_gifs": completed, "trusted_card_sha256": "c" * 64, "final_cost_pending": True,
            "producer_sha256": "e" * 64, "private_nonce_values_copied": False, "bulk_state_arrays_logs_copied": False,
            **{key: 0 for key in ("models", "draws", "scorer_calls", "grader_calls", "rendered_frames")},
        }
        if version == "v2":
            contract = fixture("pr223_native3_repaired_schema_contract.json", CONTRACT_SHA)
            self.report.update(predecessor_cost=deepcopy(contract["predecessor_cost"]),
                               predecessor_metadata_anchor=deepcopy(contract["predecessor_metadata_anchor"]))
            self.final.update(predecessor_cost=deepcopy(contract["predecessor_cost"]),
                first_invalid_anchor_sha256=repaired.PREDECESSOR_ANCHOR_SHA,
                predecessor_closed_metadata_sha256=repaired.PREDECESSOR_METADATA_SHA,
                preserved_metadata_prefix_phases=22, pretraining_invalid_debit_once=True)
            self.proof.update(source=deepcopy(source), first_invalid_anchor_sha256=repaired.PREDECESSOR_ANCHOR_SHA,
                predecessor_closed_metadata_sha256=repaired.PREDECESSOR_METADATA_SHA,
                preserved_metadata_prefix_phases=22, predecessor_cost_only_verified=True, request_boundary_control_verified=True)
        self.account(); self.refresh()

    def cost(self, history):
        rows = self.report["rows"]
        paid = math.fsum(row["cost"]["paid_wall_seconds"] for row in rows)
        reserved = math.fsum(row["cost"]["reserved_seconds"] for row in rows)
        metadata_paid = math.fsum(phase["paid_wall_seconds"] for phase in history)
        added = repaired.PRETRAINING_INVALID_CHARGE if self.version == "v2" else 0.
        charge = math.fsum((v1.PARENT_CASE_CHARGE, added, paid, reserved, metadata_paid))
        costs = {"aggregate_cap_seconds": 10800, "metadata_cap_seconds": 180, "case_caps_sum_seconds": 4290,
            "export_grace_seconds": 0, "retries": 0, "prior_case_charged_seconds": v1.PARENT_CASE_CHARGE,
            "current_case_paid_wall_seconds": paid, "current_case_reserved_seconds": reserved, "current_case_charged_seconds": paid + reserved,
            "metadata": {"paid_wall_seconds": metadata_paid, "reserved_seconds": 0., "charged_seconds": metadata_paid,
                         "overrun_seconds": max(0, metadata_paid - 180), "blocked": metadata_paid >= 180},
            "charged_seconds": charge, "remaining_seconds": 10800 - charge,
            "overrun_seconds": math.fsum(row["cost"]["overrun_seconds"] for row in rows) + max(0, metadata_paid - 180),
            "aggregate_overrun_seconds": max(0, charge - 10800),
            "halt_required": metadata_paid >= 180 or charge > 10800 or any(row["cost"]["overrun_seconds"] > 0 for row in rows)}
        if self.version == "v2": costs.update(pretraining_invalid_case_charged_seconds=added,
            combined_prior_case_charged_seconds=repaired.COMBINED_PRIOR_CASE_CHARGE)
        return costs

    def account(self):
        self.report["cost_snapshot_before_publication"] = self.cost(self.report["metadata_phase_history_before_publication"])
        self.final["costs"] = self.cost(self.final["metadata_phase_history"])
        self.final["status"] = ("PASS" if self.report["accepted_original_gate_counts"]["PASS"] == 3 else "FAIL") if self.report["completed"] == 3 and not self.final["costs"]["halt_required"] else "INCOMPLETE"

    def refresh(self):
        root = self.root; directory = self.directory
        atomic_json(root / directory / "results.json", self.report)
        sha = file_hash(root / directory / "results.json")
        self.final["original_terminal_cut_results_sha256"] = sha; self.proof["results_sha256"] = sha
        atomic_json(root / directory / "FINAL_COST.json", self.final)
        atomic_json(root / directory / "verification.json", self.proof)
        (root / directory / "README.md").write_text("Private synthetic metadata only; no numerical evidence.\n")
        self.entry = memory.passive_publication_entry(root, directory)
        registry = root / memory.PASSIVE_REGISTRY
        current = read_json(registry) if registry.exists() else {
            "schema": "forge_passive_publications_registry_v1", "qualification_input": False,
            "reuse": False, "cross_cohort_pooling": False, "publications": [], "latest": {}}
        current["publications"] = [e for e in current["publications"] if e["id"] != self.entry["id"]] + [self.entry]
        current["latest"][self.adapter.SCHEMA] = self.entry["id"]
        atomic_json(registry, current)
        return self.entry

    def latest(self): return memory.load_passive_publications(self.root)["latest"][self.adapter.SCHEMA]


def test_frozen_contract_and_protected_source_unchanged():
    contract = fixture("pr223_native3_repaired_schema_contract.json", CONTRACT_SHA)
    assert contract["status"] == "PROSPECTIVE_SCHEMA_ONLY_NOT_RESULTS"
    assert contract["schema"] == repaired.SCHEMA and contract["final_cost"]["schema"] == repaired.FINAL_SCHEMA
    assert contract["verification"]["schema"] == repaired.VERIFICATION_SCHEMA
    assert contract["ordered_ids"] == list(repaired.IDS)
    assert contract["predecessor_cost"] == repaired.PREDECESSOR_COST
    assert hashlib.sha256((ROOT / "experiments/forge/native3_publication_memory.py").read_bytes()).hexdigest() == "a1fbc7c3f1677c27b747233b677adb925a37109feea8ea78aeb01bcce0715314"
    assert function_hash(ROOT / "experiments/forge/publication_memory.py", "_pr223_cut") == "b931c69e1b0a35d94f4e5a85c8b358eb97f6b43bafe50c3e27319ce14a6914be"


def test_optional_absence_retains_existing_empty_projection(tmp_path):
    assert memory.load_passive_publications(tmp_path) == {}
    assert memory.normalize(tmp_path) == []


def test_three_current_native_cells_carry_both_prior_debits_once(tmp_path):
    cut = PrivateCut(tmp_path); result = cut.latest()
    assert result["required"] == 3 and result["original_required"] == 19
    assert result["status"] == "PASS" and result["accepted_counts"] == {"PASS": 3, "FAIL": 0, "UNAVAILABLE": 0}
    assert result["predecessor_cost"] == repaired.PREDECESSOR_COST
    cost = result["cost"]
    assert cost["prior_case_charged_seconds"] == 3165.841891122982
    assert cost["pretraining_invalid_case_charged_seconds"] == 2.0496059330180287
    assert cost["combined_prior_case_charged_seconds"] == 3167.891497056
    assert cost["charged_seconds"] == math.fsum((3165.841891122982, 2.0496059330180287, 3., cost["metadata"]["charged_seconds"]))
    records = memory.normalize(tmp_path)
    assert len(records) == 1 and records[0]["study_id"] == "pr223-native3-repaired-continuation"
    assert records[0]["predecessor_cost"]["numerical_credit"] is False
    assert records[0]["qualification_input"] is records[0]["qualification_reuse"] is False


def test_joint_fail_with_raw_accuracy_pass_is_real_recorded_fail(tmp_path):
    cut = PrivateCut(tmp_path, coverage_failure=1)
    result = cut.latest()
    assert result["accepted_counts"] == {"PASS": 2, "FAIL": 1, "UNAVAILABLE": 0} and result["status"] == "FAIL"
    assert result["cells"][1]["original_gate"] == "FAIL" and result["cells"][1]["raw_status"] == "PASS"


def test_unavailable_null_gates_and_invalid_error_have_no_accepted_media(tmp_path):
    cut = PrivateCut(tmp_path, completed=0, invalid=0)
    result = cut.latest()
    assert result["accepted_counts"] == {"PASS": 0, "FAIL": 0, "UNAVAILABLE": 3}
    assert [cell["gate_status"] for cell in result["cells"]] == ["INVALID", "NOT_RUN", "NOT_RUN"]
    assert result["cells"][0]["raw_status"] == "ERROR"
    assert all(cell["media"] is None for cell in result["cells"])
    assert result["cost"]["current_case_paid_wall_seconds"] == 1.


def test_timeout_retains_residual_full_allowance_once(tmp_path):
    cut = PrivateCut(tmp_path, completed=0, invalid=0, timeout=True)
    result = cut.latest()
    assert result["cells"][0]["cost"]["charged_seconds"] == 1470
    assert result["cost"]["current_case_reserved_seconds"] == 1469
    assert result["status"] == "INCOMPLETE"


def test_case_overrun_keeps_raw_pass_and_measured_cost_without_numeric_credit(tmp_path):
    cut = PrivateCut(tmp_path, completed=1)
    row = cut.report["rows"][0]
    row.update(accepted_original_gate=None, acceptance_status="UNAVAILABLE_BUDGET_EXCEEDED",
               full_protocol_complete=False, media=None)
    row["cost"].update(paid_wall_seconds=1471., charged_seconds=1471., overrun_seconds=1.)
    cut.report.update(completed=0, accepted_original_gate_counts={"PASS": 0, "FAIL": 0, "UNAVAILABLE": 3},
                      all_required_case_evidence_complete=False)
    cut.proof.update(accepted_original_gate_counts={"PASS": 0, "FAIL": 0, "UNAVAILABLE": 3}, accepted_original_gifs=0)
    cut.account(); cut.refresh()
    result = cut.latest()
    cell = result["cells"][0]
    assert result["status"] == "INCOMPLETE" and result["completed"] == 0
    assert cell["execution_status"] == cell["raw_status"] == "PASS"
    assert cell["original_gate"] is None and cell["gate_status"] == "INCOMPLETE" and cell["media"] is None
    assert cell["cost"]["charged_seconds"] == 1471. and cell["cost"]["overrun_seconds"] == 1.
    assert result["cost"]["charged_seconds"] == math.fsum((repaired.COMBINED_PRIOR_CASE_CHARGE, 1471.,
                                                          result["cost"]["metadata"]["charged_seconds"]))


def test_final_metadata_overrun_keeps_completed_raw_cells_and_halts_aggregate(tmp_path):
    cut = PrivateCut(tmp_path)
    history = cut.final["metadata_phase_history"]
    history[-1]["paid_wall_seconds"] = 181.25 - math.fsum(p["paid_wall_seconds"] for p in history[:-1])
    history[-1]["status"] = "BUDGET_EXCEEDED"
    cut.account(); cut.refresh()
    result = cut.latest()
    assert result["status"] == "INCOMPLETE" and result["raw_native3_protocol_gate"] == "PASS"
    assert result["accepted_counts"] == {"PASS": 3, "FAIL": 0, "UNAVAILABLE": 0}
    assert result["cost"]["metadata"]["charged_seconds"] == 181.25
    assert result["cost"]["overrun_seconds"] == 1.25 and result["cost"]["halt_required"] is True
    assert result["qualification_input"] is result["default_adoption"] is False


def test_v1_v2_coexist_with_separate_sources_latest_and_unchanged_v1(tmp_path):
    old = PrivateCut(tmp_path, completed=0, invalid=0, version="v1")
    before = deepcopy(old.latest()); old_bytes = (tmp_path / old.directory / "results.json").read_bytes()
    new = PrivateCut(tmp_path, completed=1)
    projection = memory.load_passive_publications(tmp_path)
    assert projection["latest"][v1.SCHEMA] == before
    assert projection["latest"][repaired.SCHEMA]["source"] != before["source"]
    assert (tmp_path / old.directory / "results.json").read_bytes() == old_bytes
    assert projection["cross_cohort_pooling"] is False
    records = memory.normalize(tmp_path)
    assert len(records) == 2 and {r["study_id"] for r in records} == {"pr223-native3-continuation", "pr223-native3-repaired-continuation"}
    assert records[0]["recorded_cost"] != records[1]["recorded_cost"]


def test_full19_adapter_and_old_records_remain_unchanged(tmp_path):
    # Reuse only the old author fixture source; ROOT is redirected to private
    # source templates BEFORE make_cut. No committed original report is read.
    templates = tmp_path / "source-templates"
    atomic_json(templates / memory.PR223_PROTOCOL, original_protocol())
    path = ROOT / "tests/test_forge_passive_publications.py"
    source = path.read_text()
    # Extract the two unchanged pure author-fixture functions, avoiding the
    # unrelated knowledge module import and every existing author test.
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in {"make_cut", "refresh"}]
    assert {n.name for n in nodes} == {"make_cut", "refresh"}
    namespace = dict(ROOT=templates, memory=memory, Counter=Counter, deepcopy=deepcopy,
                     atomic_json=atomic_json, file_hash=file_hash, read_json=read_json, stable_hash=stable_hash)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["make_cut"](tmp_path)
    before = deepcopy(memory.load_passive_publications(tmp_path)["latest"][memory.PR223_SCHEMA])
    records = deepcopy(memory.normalize(tmp_path))
    PrivateCut(tmp_path)
    assert memory.load_passive_publications(tmp_path)["latest"][memory.PR223_SCHEMA] == before
    assert [r for r in memory.normalize(tmp_path) if r["study_id"] == "pr223-original-full19-retest"] == records


def test_later_v2_prefix_preserves_previous_cells_and_cost_history(tmp_path):
    first = PrivateCut(tmp_path, completed=1, name="synthetic-native3-v2-first")
    old = deepcopy(first.latest())
    second = PrivateCut(tmp_path, completed=2, name="synthetic-native3-v2-second")
    result = second.latest()
    assert result["completed"] == 2 and result["cells"][0] == old["cells"][0]
    assert result["cost"]["charged_seconds"] > old["cost"]["charged_seconds"]
    assert len(memory.load_passive_publications(tmp_path)["cuts"]) == 2


@pytest.mark.parametrize("change", ("rewind", "forked_source", "accepted_cell"))
def test_latest_cannot_rewind_fork_or_rewrite_accepted_cells(tmp_path, change):
    PrivateCut(tmp_path, completed=2, name="synthetic-native3-v2-first")
    second = PrivateCut(tmp_path, completed=1 if change == "rewind" else 2,
                        name="synthetic-native3-v2-second")
    if change == "forked_source":
        second.report["source"] = {"origin_commit": "b" * 40, "digest": "c" * 64}
        second.proof["source"] = deepcopy(second.report["source"])
    elif change == "accepted_cell":
        second.report["rows"][0]["final_metrics"]["synthetic_flag"] = "changed recorded metric"
    second.refresh()
    with pytest.raises(ValueError): second.latest()


def test_recall_freshness_inputs_include_versioned_adapter_and_fixed_anchors(tmp_path):
    PrivateCut(tmp_path)
    paths = set(memory.input_paths(tmp_path))
    assert tmp_path / "experiments/forge/native3_repaired_publication_memory.py" in paths
    assert tmp_path / "experiments/forge/native3_publication_memory.py" in paths
    assert tmp_path / repaired.PREDECESSOR_ANCHOR_PATH in paths
    assert tmp_path / repaired.PREDECESSOR_METADATA_PATH in paths


def test_goal_bytes_and_all_nine_original_clocks_are_pinned(tmp_path):
    cut = PrivateCut(tmp_path)
    for row in cut.latest()["cells"]:
        media = row["media"]
        assert media["frames"] == 9 and media["actual_steps"] == [0, 50, 750, 1750, 2750, 3750, 4750, 5750, 7000]
        assert (tmp_path / cut.directory / media["file"]).read_bytes() == SYNTHETIC_GIF
    media = cut.report["rows"][0]["media"]
    (tmp_path / cut.directory / media["file"]).write_bytes(SYNTHETIC_GIF + b"tamper")
    with pytest.raises(ValueError): cut.latest()


def tamper(cut, change):
    report, final, proof = cut.report, cut.final, cut.proof
    row = report["rows"][0]
    if change == "missing_startup": report["predecessor_cost"].pop("pretraining_invalid_case_charged_seconds")
    elif change == "startup_zero": report["predecessor_cost"]["pretraining_invalid_case_charged_seconds"] = 0.
    elif change == "startup_grade": report["predecessor_cost"]["numerical_credit"] = True
    elif change == "startup_source": report["predecessor_cost"]["source"]["digest"] = "f" * 64
    elif change == "cost_startup_omitted": final["costs"].pop("pretraining_invalid_case_charged_seconds")
    elif change == "cost_startup_double": final["costs"]["charged_seconds"] += repaired.PRETRAINING_INVALID_CHARGE
    elif change == "cost_combined_wrong": final["costs"]["combined_prior_case_charged_seconds"] = v1.PARENT_CASE_CHARGE
    elif change == "metadata_double": final["costs"]["charged_seconds"] += 58.63964644144289
    elif change == "metadata_reset": final["costs"]["metadata"]["paid_wall_seconds"] = 0.
    elif change == "parent22_changed": final["metadata_phase_history"][13]["name"] = "forgotten error"
    elif change == "parent22_missing": final["metadata_phase_history"] = final["metadata_phase_history"][:12]
    elif change == "phase_rewind": final["metadata_phase_history"] = deepcopy(report["metadata_phase_history_before_publication"])
    elif change == "startup_anchor": final["first_invalid_anchor_sha256"] = "f" * 64
    elif change == "metadata_anchor": report["predecessor_metadata_anchor"]["sha256"] = "f" * 64
    elif change == "final_old_source": report["source"] = deepcopy(repaired.PREDECESSOR_SOURCE); proof["source"] = deepcopy(report["source"])
    elif change == "old19_source": report["source"] = deepcopy(v1.PARENT_SOURCE); proof["source"] = deepcopy(report["source"])
    elif change == "full19_credit": report["claims"]["full_new_source_original19_credit"] = True
    elif change == "old_grade": report["prior_reference"]["execution_counts"]["PASS"] = 19
    elif change == "proof_source": proof["source"]["origin_commit"] = "a" * 40
    elif change == "proof_request_missing": proof.pop("request_boundary_control_verified")
    elif change == "proof_anchor": proof["first_invalid_anchor_sha256"] = "f" * 64
    elif change == "proof_work": proof["scorer_calls"] = 1
    elif change == "producer_work": report["producer"]["rendered_frames"] = 1
    elif change == "cap_reset": final["costs"]["aggregate_cap_seconds"] = 21600
    elif change == "metadata180_reset": final["costs"]["metadata_cap_seconds"] = 360
    elif change == "raw_joint": row["raw_reported_status"] = "FAIL"
    elif change == "noisy_fail_credit": row["native_gates"]["noisy"]["coverage"] = "FAIL"
    elif change == "recipe": row["complete_recipe"]["lr"] = .0053125
    elif change == "width": row["original_requirements"][0][2] = 95.
    elif change == "seed": row["required_host"]["seed"] = 0
    elif change == "read24": row["metric_observations"] = 24
    elif change == "steps6999": row["completed_steps"] = 6999
    elif change == "holdout20k": row["independent_holdout_samples"] = 20000
    elif change == "terminal5": row["terminal_reads"] = [7000]
    elif change == "noise_law": row["observer"]["primary_law"] = "clean"
    elif change == "clock9": row["media"]["actual_steps"][1] = 51
    elif change == "frame8": row["media"]["frames"] = 8
    elif change == "plain_nonce": row["nonce"] = "synthetic-not-a-secret"
    elif change == "unknown_nonce": report["private_nonce_value"] = "synthetic-private"
    elif change == "default_credit": report["claims"]["default_adoption"] = True
    elif change == "live_provenance": final["authoritative_final_metadata_ledger"]["path"] = "/foreign/ledger.json"
    elif change == "closed_not_copy": final["consumed_closed_metadata_ledger"]["path"] = v1.CANONICAL_LEDGER; final["ledger_input_is_copy"] = False
    elif change == "overrun_accepted": row["cost"].update(paid_wall_seconds=1471., charged_seconds=1471., overrun_seconds=1.); cut.account()
    elif change == "reserve_clip": row["cost"]["reserved_seconds"] = row["cost"]["unmeasured_interrupt_reserved_seconds"] = 0.
    elif change == "subset": report["rows"].pop()
    elif change == "row_order": report["rows"].reverse()
    elif change == "denominator": report["required"] = 19
    elif change == "schema_v1": report["schema"] = v1.SCHEMA
    elif change == "malformed_gates": row["native_gates"] = []
    else: raise AssertionError(change)


@pytest.mark.parametrize("change", (
    "missing_startup", "startup_zero", "startup_grade", "startup_source", "cost_startup_omitted", "cost_startup_double",
    "cost_combined_wrong", "metadata_double", "metadata_reset", "parent22_changed", "parent22_missing", "phase_rewind",
    "startup_anchor", "metadata_anchor", "final_old_source", "old19_source", "full19_credit", "old_grade", "proof_source",
    "proof_request_missing", "proof_anchor", "proof_work", "producer_work", "cap_reset", "metadata180_reset", "raw_joint",
    "noisy_fail_credit", "recipe", "width", "seed", "read24", "steps6999", "holdout20k", "terminal5", "noise_law", "clock9",
    "frame8", "plain_nonce", "unknown_nonce", "default_credit", "live_provenance", "closed_not_copy", "overrun_accepted",
    "reserve_clip", "subset", "row_order", "denominator", "schema_v1", "malformed_gates",
))
def test_coherent_metadata_repins_cannot_waive_contract(tmp_path, change):
    cut = PrivateCut(tmp_path, completed=0, invalid=0, timeout=True) if change == "reserve_clip" else PrivateCut(tmp_path)
    tamper(cut, change); cut.refresh()
    with pytest.raises(ValueError): cut.latest()


@pytest.mark.parametrize("role", ("result", "final_cost", "verification", "readout", "protocol", "parent_metadata_anchor",
                                  "predecessor_anchor", "predecessor_metadata_anchor", "media"))
def test_missing_or_changed_committed_bytes_refused(tmp_path, role):
    cut = PrivateCut(tmp_path)
    pin = cut.entry[role][0] if role == "media" else cut.entry[role]
    (tmp_path / pin["path"]).write_bytes(b"tampered private bytes")
    with pytest.raises(ValueError): cut.latest()


def test_coherently_repinned_anchor_cannot_appoint_new_cost_history(tmp_path):
    cut = PrivateCut(tmp_path)
    data = read_json(tmp_path / repaired.PREDECESSOR_ANCHOR_PATH); data["case_cost"]["charged_seconds"] = 0
    atomic_json(tmp_path / repaired.PREDECESSOR_ANCHOR_PATH, data)
    cut.refresh()
    with pytest.raises(ValueError): cut.latest()


def test_coherently_repinned_closed22_cannot_appoint_new_phase_history(tmp_path):
    cut = PrivateCut(tmp_path)
    data = read_json(tmp_path / repaired.PREDECESSOR_METADATA_PATH)
    data["phases"][13]["name"] = "replacement private history"
    atomic_json(tmp_path / repaired.PREDECESSOR_METADATA_PATH, data)
    cut.refresh()
    with pytest.raises(ValueError): cut.latest()


def test_open_final_metadata_phase_is_refused_after_coherent_byte_repin(tmp_path):
    cut = PrivateCut(tmp_path)
    cut.final["metadata_phase_history"][-1]["status"] = "ACTIVE"
    cut.refresh()
    with pytest.raises(ValueError): cut.latest()


def test_unknown_schema_and_unsafe_registered_paths_refused(tmp_path):
    cut = PrivateCut(tmp_path)
    registry = read_json(tmp_path / memory.PASSIVE_REGISTRY)
    registry["publications"][0]["schema"] = "pg_pr223_native3_passive_publication_v999"
    atomic_json(tmp_path / memory.PASSIVE_REGISTRY, registry)
    with pytest.raises(ValueError): cut.latest()
    cut.refresh(); registry = read_json(tmp_path / memory.PASSIVE_REGISTRY)
    registry["publications"][0]["media"][0]["path"] = "../not-consumed.gif"
    atomic_json(tmp_path / memory.PASSIVE_REGISTRY, registry)
    with pytest.raises(ValueError): cut.latest()


def test_nonfinite_json_is_not_projected_even_after_byte_repin(tmp_path):
    cut = PrivateCut(tmp_path)
    value = deepcopy(cut.report); value["rows"][0]["final_metrics"]["bad"] = float("nan")
    path = tmp_path / cut.directory / "results.json"; path.write_text(json.dumps(value))
    registry = read_json(tmp_path / memory.PASSIVE_REGISTRY)
    registry["publications"][0]["result"].update(sha256=file_hash(path), bytes=path.stat().st_size)
    atomic_json(tmp_path / memory.PASSIVE_REGISTRY, registry)
    with pytest.raises(ValueError): cut.latest()


def test_no_raw_or_live_provenance_is_consumed_or_science_imported(tmp_path):
    cut = PrivateCut(tmp_path)
    modules = set(sys.modules)
    result = cut.latest()
    assert result["authoritative_final_metadata_ledger"]["path"] == v1.CANONICAL_LEDGER
    assert result["consumed_closed_metadata_ledger"]["path"] == "/not-consumed/closed-ledger.json"
    assert not any(name.split(".", 1)[0] in {"torch", "numpy", "particlegan", "benchmarks", "lib"} for name in set(sys.modules) - modules)

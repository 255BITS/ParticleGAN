"""Private recorded metadata controls, not models, scores or real publications."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.forge import native3_publication_memory as native3
from experiments.forge import publication_memory as memory
from experiments.forge.contracts import atomic_json, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/pr223_native3_parent_metadata.json"


def make_cut(root, *, completed=3, coverage_failure=None, invalid=None, timeout=False):
    protocol_path = root / memory.PR223_PROTOCOL
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes((ROOT / memory.PR223_PROTOCOL).read_bytes())
    protocol = read_json(protocol_path)
    anchor_path = root / native3.PARENT_METADATA_PIN_PATH
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_path.write_bytes(FIXTURE.read_bytes())
    anchor = read_json(anchor_path)
    assert hashlib.sha256(anchor_path.read_bytes()).hexdigest() == native3.PARENT_METADATA_SHA
    rows = []
    for index, declared in enumerate(row for row in protocol["rows"] if row["group"] == "native"):
        definition = declared["original_definition"]
        accepted = index < completed
        status = ("FAIL" if index == coverage_failure else "PASS") if accepted else "NOT_RUN"
        if index == invalid:
            accepted, status = False, "INCOMPLETE" if timeout else "INVALID"
        paid = 1. if accepted or index == invalid else 0.
        allowance = declared["proposed_inclusive_allowance_seconds"]
        terminal = ("timeout" if timeout else "completed") if index == invalid else ("completed" if accepted else "missing")
        reserved = allowance - paid if index == invalid and timeout else 0.
        row = {
            "id": native3.IDS[index], "task": declared["task"], "group": "native",
            "parent_full19_retest_id": declared["id"], "parent_original_id": definition["id"],
            "required_host": definition["original_host"], "complete_recipe": declared["resolved_recipe"],
            "observer": declared["observer"], "original_requirements": definition["original_requirements"],
            "observation_steps": definition["observation_steps"], "execution_status": status,
            "acceptance_status": "ACCEPTED" if accepted else "UNAVAILABLE",
            "accepted_original_gate": status if accepted else None,
            "raw_reported_status": "PASS" if accepted else ("ERROR" if index == invalid else None),
            "full_protocol_complete": accepted, "recorded_full_protocol_complete": accepted,
            "cost": {"allowance_seconds": allowance, "paid_wall_seconds": paid,
                     "reserved_seconds": reserved, "unmeasured_interrupt_reserved_seconds": reserved,
                     "charged_seconds": paid + reserved, "overrun_seconds": 0.,
                     "terminal_status": terminal, "completed_terminal": terminal == "completed", "certified": accepted},
            "media": None,
        }
        if accepted:
            row.update(completed_steps=7000, metric_observations=34,
                       terminal_reads=[6000, 6250, 6500, 6750, 7000], independent_holdout_samples=100000,
                       final_metrics={"recorded_synthetic_flag": status},
                       native_gates={"noisy": {"coverage": "FAIL" if index == coverage_failure else "PASS", "accuracy": "PASS"},
                                     "clean": {"coverage": "FAIL", "accuracy": "FAIL"}})
            media_file = "gifs/native-" + declared["task"] + ".gif"
            media_path = root / "reports/forge/synthetic-native3" / media_file
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(b"GIF89a-private-byte-fixture")
            row["media"] = {"file": media_file, "sha256": file_hash(media_path), "bytes": media_path.stat().st_size,
                            "frames": 9, "actual_steps": declared["media_steps"], "original_gate": status}
        rows.append(row)
    counts = {key: Counter(row["execution_status"] for row in rows).get(key, 0) for key in native3.STATUS_KEYS}
    completed = sum(row["full_protocol_complete"] for row in rows)
    accepted_counts = {"PASS": sum(row["accepted_original_gate"] == "PASS" for row in rows),
                       "FAIL": sum(row["accepted_original_gate"] == "FAIL" for row in rows), "UNAVAILABLE": 3 - completed}
    raw_gate = ("FAIL" if coverage_failure is not None else "PASS") if completed == 3 else "UNAVAILABLE"
    before = deepcopy(anchor["phases"]) + [{"index": 12, "name": "synthetic_native3_metadata",
        "paid_wall_seconds": 1., "paused_wall_seconds": 0., "status": "COMPLETE"}]
    after = deepcopy(before) + [{"index": 13, "name": "synthetic_passive_publication",
        "paid_wall_seconds": 1., "paused_wall_seconds": 0., "status": "COMPLETE"}]

    def costs(history):
        paid = math.fsum(row["cost"]["paid_wall_seconds"] for row in rows)
        reserved = math.fsum(row["cost"]["reserved_seconds"] for row in rows)
        metadata_paid = math.fsum(phase["paid_wall_seconds"] for phase in history)
        charge = native3.PARENT_CASE_CHARGE + paid + reserved + metadata_paid
        return {"aggregate_cap_seconds": 10800, "metadata_cap_seconds": 180, "case_caps_sum_seconds": 4290,
            "export_grace_seconds": 0, "retries": 0, "prior_case_charged_seconds": native3.PARENT_CASE_CHARGE,
            "current_case_paid_wall_seconds": paid, "current_case_reserved_seconds": reserved,
            "current_case_charged_seconds": paid + reserved,
            "metadata": {"paid_wall_seconds": metadata_paid, "reserved_seconds": 0., "charged_seconds": metadata_paid,
                         "overrun_seconds": 0., "blocked": False},
            "charged_seconds": charge, "remaining_seconds": 10800 - charge, "overrun_seconds": 0.,
            "aggregate_overrun_seconds": max(0, charge - 10800), "halt_required": False}

    report = {"schema": native3.SCHEMA, "publication_scope": "native3", "family": "atlas", "required": 3,
        "original_required": 19, "completed": completed, "source": {"origin_commit": "a" * 40, "digest": "b" * 64},
        "rows": rows, "counts": counts, "accepted_original_gate_counts": accepted_counts,
        "all_required_case_evidence_complete": completed == 3, "raw_native3_protocol_gate": raw_gate,
        "accepted_native3_status": "PENDING_PUBLICATION_COST_FINALIZATION",
        "trusted_terminal_card": {"sha256": "c" * 64, "bytes": 100},
        "cost_snapshot_before_publication": costs(before),
        "metadata_phase_history_before_publication": before, "metadata_phase_count_before_publication": len(before),
        "authoritative_metadata_ledger_path": native3.CANONICAL_LEDGER,
        "parent_metadata_anchor": {"sha256": native3.PARENT_METADATA_SHA, "bytes": 2675, "phases": 12,
                                   "snapshot_relative": native3.PARENT_METADATA_PATH},
        "prior_reference": {"source": deepcopy(native3.PARENT_SOURCE), "required": 19,
            "files_sha256": deepcopy(native3.PARENT_FILES), "case_charged_seconds": native3.PARENT_CASE_CHARGE,
            "execution_counts": {"PASS": 16, "INVALID": 1, "NOT_RUN": 2},
            "accepted_original_gate_counts": {"PASS": 16, "FAIL": 0, "UNAVAILABLE": 3},
            "old_grades_are_current_credit": False, "original_evidence_unchanged": True},
        "producer": {"sha256": "e" * 64, **{key: 0 for key in ("models", "draws", "scorer_calls", "regrades", "rendered_frames")}},
        "claims": {key: False for key in native3.CLAIMS},
    }
    final = {"schema": native3.FINAL_SCHEMA, "costs": costs(after), "raw_native3_protocol_gate": raw_gate,
        "status": raw_gate if completed == 3 else "INCOMPLETE", "metadata_phase_count": len(after),
        "metadata_phase_history": after, "trusted_terminal_card_sha256": "c" * 64,
        "authoritative_final_metadata_ledger": {"path": native3.CANONICAL_LEDGER, "sha256": "d" * 64, "bytes": 100},
        "consumed_closed_metadata_ledger": {"path": native3.CANONICAL_LEDGER, "sha256": "d" * 64, "bytes": 100},
        "ledger_input_is_copy": False,
        "original_case_verdicts_unchanged": True, "old_case_debit_once": True, "cumulative_metadata_once": True,
        "old_grades_are_current_credit": False, "full_new_source_original19_credit": False}
    proof = {"schema": native3.VERIFICATION_SCHEMA, "status": "VERIFIED_PRE_PUBLICATION_COST_CUT",
        "required": 3, "original_required": 19, "counts": counts, "accepted_original_gate_counts": accepted_counts,
        "accepted_original_gifs": completed, "trusted_card_sha256": "c" * 64, "final_cost_pending": True,
        "producer_sha256": "e" * 64, "private_nonce_values_copied": False, "bulk_state_arrays_logs_copied": False,
        **{key: 0 for key in ("models", "draws", "scorer_calls", "grader_calls", "rendered_frames")}}
    cut = {"root": root, "directory": "reports/forge/synthetic-native3", "report": report, "final": final, "proof": proof}
    refresh(cut)
    return cut


def refresh(cut):
    root, directory = cut["root"], cut["directory"]
    atomic_json(root / directory / "results.json", cut["report"])
    result_sha = file_hash(root / directory / "results.json")
    cut["final"]["original_terminal_cut_results_sha256"] = result_sha
    cut["proof"]["results_sha256"] = result_sha
    for filename, value in (("FINAL_COST.json", cut["final"]), ("verification.json", cut["proof"])):
        atomic_json(root / directory / filename, value)
    (root / directory / "README.md").write_text("Private synthetic recorded metadata only.\n")
    entry = memory.passive_publication_entry(root, directory)
    atomic_json(root / memory.PASSIVE_REGISTRY, {"schema": "forge_passive_publications_registry_v1",
        "qualification_input": False, "reuse": False, "cross_cohort_pooling": False,
        "publications": [entry], "latest": {native3.SCHEMA: entry["id"]}})
    return entry


class Native3PassivePublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def latest(self):
        return memory.load_passive_publications(self.root)["latest"][native3.SCHEMA]

    def test_optional_absence_keeps_previous_empty_behavior(self):
        self.assertEqual(memory.load_passive_publications(self.root), {})
        self.assertEqual(memory.normalize(self.root), [])

    def test_prospective_contract_scope_and_source_pin(self):
        data = (ROOT / "tests/fixtures/pr223_native3_schema_contract.json").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), "58a16973e8fcb270ec7f121538e25b288340ca788f69236cac3ffba27772be5e")
        contract = read_json(ROOT / "tests/fixtures/pr223_native3_schema_contract.json")
        self.assertEqual(contract["schema"], native3.SCHEMA)
        self.assertEqual(contract["ordered_ids"], list(native3.IDS))
        self.assertEqual(contract["status"], "PROSPECTIVE_SCHEMA_ONLY_NOT_RESULTS")

    def test_native3_record_has_only_three_current_cells_and_parent_provenance(self):
        make_cut(self.root)
        cut = self.latest()
        self.assertEqual(cut["required"], 3)
        self.assertEqual(cut["original_required"], 19)
        self.assertEqual(cut["accepted_counts"], {"PASS": 3, "FAIL": 0, "UNAVAILABLE": 0})
        self.assertEqual(cut["status"], "PASS")
        self.assertEqual(len(cut["cells"]), 3)
        record = memory.normalize(self.root)[0]
        self.assertEqual(record["study_id"], "pr223-native3-continuation")
        self.assertNotIn("trial_ids", record)
        self.assertFalse(record["qualification_input"])
        self.assertFalse(record["qualification_reuse"])
        self.assertEqual(len(record["task_results"]), 3)
        self.assertIn("no fresh single-source 19", record["next_action"])

    def test_raw_accuracy_pass_with_noisy_coverage_fail_is_accepted_fail(self):
        make_cut(self.root, coverage_failure=0)
        cut = self.latest()
        self.assertEqual(cut["status"], "FAIL")
        self.assertEqual(cut["accepted_counts"], {"PASS": 2, "FAIL": 1, "UNAVAILABLE": 0})
        self.assertEqual(cut["cells"][0]["raw_status"], "PASS")
        self.assertEqual(cut["cells"][0]["original_gate"], "FAIL")

    def test_invalid_complete_terminal_is_paid_only_and_numerically_unavailable(self):
        make_cut(self.root, completed=0, invalid=0)
        cut = self.latest()
        self.assertEqual(cut["status"], "INCOMPLETE")
        self.assertEqual(cut["counts"], {"INVALID": 1, "NOT_RUN": 2})
        self.assertEqual(cut["accepted_counts"]["UNAVAILABLE"], 3)
        self.assertEqual(cut["cost"]["current_case_paid_wall_seconds"], 1.)
        self.assertEqual(cut["cost"]["current_case_reserved_seconds"], 0.)
        self.assertIsNone(cut["cells"][0]["media"])

    def test_pretraining_invalid_and_unrun_null_gates_remain_unavailable(self):
        cut = make_cut(self.root, completed=0, invalid=0)
        for row in cut["report"]["rows"]:
            row.update(native_gates=None, final_metrics=None, raw_reported_status=None)
        refresh(cut)
        projected = self.latest()
        self.assertEqual(projected["counts"], {"INVALID": 1, "NOT_RUN": 2})
        self.assertEqual(projected["accepted_counts"], {"PASS": 0, "FAIL": 0, "UNAVAILABLE": 3})
        self.assertEqual(projected["status"], "INCOMPLETE")
        self.assertEqual(projected["cost"]["current_case_paid_wall_seconds"], 1.)
        self.assertTrue(all(row["original_gate"] is None and row["raw_status"] is None
                            and row["media"] is None for row in projected["cells"]))

    def test_null_gates_do_not_supply_an_accepted_score(self):
        cut = make_cut(self.root)
        cut["report"]["rows"][0]["native_gates"] = None
        refresh(cut)
        with self.assertRaises(ValueError):
            self.latest()

    def test_unavailable_gates_reject_malformed_nonmapping_values(self):
        cut = make_cut(self.root, completed=0, invalid=0)
        for value in ([], ["PASS"], "", "PASS", 0, False, {"noisy": None}):
            with self.subTest(value=value):
                cut["report"]["rows"][0]["native_gates"] = value
                refresh(cut)
                with self.assertRaises(ValueError):
                    self.latest()
    def test_timeout_reserves_residual_full_allowance_and_no_additional_metadata_debit(self):
        make_cut(self.root, completed=0, invalid=0, timeout=True)
        cut = self.latest()
        self.assertEqual(cut["cost"]["current_case_paid_wall_seconds"], 1.)
        self.assertEqual(cut["cost"]["current_case_reserved_seconds"], 1469.)
        self.assertAlmostEqual(cut["cost"]["charged_seconds"],
            native3.PARENT_CASE_CHARGE + 1470. + cut["cost"]["metadata"]["charged_seconds"])

    def test_later_closed_error_phase_preserves_prefix_without_new180(self):
        cut = make_cut(self.root)
        cut["final"]["metadata_phase_history"][-1]["status"] = "ERROR"
        refresh(cut)
        self.assertEqual(self.latest()["cost"]["metadata_cap_seconds"], 180)

    def test_source_and_byte_identity_rejected_after_registry_pin(self):
        cut = make_cut(self.root)
        registry = read_json(self.root / memory.PASSIVE_REGISTRY)
        cut["report"]["source"]["origin_commit"] = "f" * 40
        atomic_json(self.root / cut["directory"] / "results.json", cut["report"])
        new_sha = file_hash(self.root / cut["directory"] / "results.json")
        registry["publications"][0]["result"]["sha256"] = new_sha
        registry["publications"][0]["result"]["bytes"] = (self.root / cut["directory"] / "results.json").stat().st_size
        atomic_json(self.root / memory.PASSIVE_REGISTRY, registry)
        with self.assertRaises(ValueError): self.latest()

    def test_material_coherent_repin_tampering_rejected(self):
        changes = ("scope19", "old_source", "subset", "old_id", "recipe", "typed_recipe", "host", "gate",
                   "cadence", "observer", "partial_steps", "partial_reads", "no_holdout", "terminal_five",
                   "clean_credit", "raw_joint_confusion", "parent_cost_reset", "parent_pin", "parent_count_type",
                   "new_metadata180", "double_metadata", "phase_prefix", "phase_rewind", "ledger_identity",
                   "cap_reset", "short_allowance", "reserve_clip", "status_credit", "count_credit", "proof_work",
                   "producer_work", "old_credit", "default", "full19_credit", "plaintext_nonce", "media_gate",
                   "missing_clean", "winner", "unattempted_terminal", "closed_ledger_pin", "copy_flag",
                   "missing_copy_provenance", "aggregate_overrun")
        for change in changes:
            with self.subTest(change=change):
                cut = make_cut(self.root, completed=0, invalid=0, timeout=True) if change == "reserve_clip" else make_cut(self.root)
                report, final, proof = cut["report"], cut["final"], cut["proof"]
                row = report["rows"][0]
                if change == "scope19": report["required"] = 19
                elif change == "old_source": report["source"] = deepcopy(native3.PARENT_SOURCE)
                elif change == "subset": report["rows"][0]["task"] = "stationary"
                elif change == "old_id": row["id"] = row["parent_full19_retest_id"]
                elif change == "recipe": row["complete_recipe"]["lr"] = .0053125
                elif change == "typed_recipe": row["complete_recipe"]["prior_reg"] = False
                elif change == "host": row["required_host"]["seed"] = 0
                elif change == "gate": row["original_requirements"][0][2] = 95
                elif change == "cadence": row["observation_steps"] = [0, 7000]
                elif change == "observer": row["observer"]["primary_law"] = "clean live"
                elif change == "partial_steps": row["completed_steps"] = 6999
                elif change == "partial_reads": row["metric_observations"] = 24
                elif change == "no_holdout": row["independent_holdout_samples"] = 20000
                elif change == "terminal_five": row["terminal_reads"] = [7000]
                elif change == "clean_credit": row["native_gates"]["noisy"]["coverage"] = "FAIL"
                elif change == "raw_joint_confusion": row["raw_reported_status"] = "FAIL"
                elif change == "parent_cost_reset": report["prior_reference"]["case_charged_seconds"] = 0.
                elif change == "parent_pin": report["prior_reference"]["files_sha256"][native3.PARENT_DIRECTORY + "/results.json"] = "f" * 64
                elif change == "parent_count_type": report["prior_reference"]["execution_counts"]["INVALID"] = True
                elif change == "new_metadata180": final["costs"]["metadata"]["paid_wall_seconds"] = 0.
                elif change == "double_metadata": final["costs"]["charged_seconds"] += native3.PARENT_METADATA_PAID
                elif change == "phase_prefix": final["metadata_phase_history"][0]["name"] = "reset"
                elif change == "phase_rewind": final["metadata_phase_history"] = deepcopy(report["metadata_phase_history_before_publication"])
                elif change == "ledger_identity": final["authoritative_final_metadata_ledger"]["path"] = "/foreign/ledger.json"
                elif change == "cap_reset": final["costs"]["aggregate_cap_seconds"] = 21600
                elif change == "short_allowance": row["cost"]["allowance_seconds"] = 180
                elif change == "reserve_clip": row["cost"]["reserved_seconds"] = row["cost"]["unmeasured_interrupt_reserved_seconds"] = 0.
                elif change == "status_credit": report["completed"] = 19
                elif change == "count_credit": report["counts"]["PASS"] = 19
                elif change == "proof_work": proof["scorer_calls"] = 1
                elif change == "producer_work": report["producer"]["models"] = 1
                elif change == "old_credit": report["claims"]["old_grades_are_current_credit"] = True
                elif change == "default": report["claims"]["default_adoption"] = True
                elif change == "full19_credit": final["full_new_source_original19_credit"] = True
                elif change == "plaintext_nonce": row["nonce"] = "private-synthetic-value"
                elif change == "media_gate": row["media"]["original_gate"] = "FAIL"
                elif change == "missing_clean": row["native_gates"].pop("clean")
                elif change == "winner": report["winner"] = True
                elif change == "unattempted_terminal":
                    report["rows"][2]["execution_status"] = "NOT_RUN"
                    report["rows"][2].update(accepted_original_gate=None, full_protocol_complete=False,
                                             acceptance_status="UNAVAILABLE", media=None)
                    report["rows"][2]["cost"].update(terminal_status="completed", completed_terminal=True,
                                                     certified=True, paid_wall_seconds=0., charged_seconds=0.)
                elif change == "closed_ledger_pin": final["consumed_closed_metadata_ledger"]["sha256"] = "f" * 64
                elif change == "copy_flag": final["ledger_input_is_copy"] = True
                elif change == "missing_copy_provenance": final.pop("consumed_closed_metadata_ledger")
                elif change == "aggregate_overrun": final["costs"]["aggregate_overrun_seconds"] = 1.
                refresh(cut)
                with self.assertRaises(ValueError): self.latest()

    def test_case_overrun_is_retained_separately_from_aggregate_excess(self):
        cut = make_cut(self.root, completed=0, invalid=0)
        row = cut["report"]["rows"][0]
        row["cost"].update(paid_wall_seconds=1471., charged_seconds=1471., overrun_seconds=1.)
        for costs in (cut["report"]["cost_snapshot_before_publication"], cut["final"]["costs"]):
            costs.update(current_case_paid_wall_seconds=1471., current_case_charged_seconds=1471.,
                         charged_seconds=costs["charged_seconds"] + 1470.,
                         remaining_seconds=costs["remaining_seconds"] - 1470.,
                         overrun_seconds=1., aggregate_overrun_seconds=0., halt_required=True)
        refresh(cut)
        projected = self.latest()
        self.assertEqual(projected["status"], "INCOMPLETE")
        self.assertEqual(projected["cost"]["current_case_paid_wall_seconds"], 1471.)
        self.assertEqual(projected["cost"]["overrun_seconds"], 1.)
        self.assertEqual(projected["cost"]["aggregate_overrun_seconds"], 0.)

    def test_final_metadata_tail_overrun_suppresses_full_cut_without_rewriting_case_flags(self):
        cut = make_cut(self.root)
        phases = cut["final"]["metadata_phase_history"]
        added = 181. - math.fsum(phase["paid_wall_seconds"] for phase in phases)
        phases[-1]["paid_wall_seconds"] += added
        phases[-1]["status"] = "BUDGET_EXCEEDED"
        costs = cut["final"]["costs"]
        costs["metadata"].update(paid_wall_seconds=181., charged_seconds=181., overrun_seconds=1., blocked=True)
        costs.update(charged_seconds=costs["charged_seconds"] + added,
                     remaining_seconds=costs["remaining_seconds"] - added, overrun_seconds=1., halt_required=True)
        cut["final"]["status"] = "INCOMPLETE"
        refresh(cut)
        projected = self.latest()
        self.assertEqual(projected["status"], "INCOMPLETE")
        self.assertEqual(projected["accepted_counts"], {"PASS": 3, "FAIL": 0, "UNAVAILABLE": 0})
        self.assertTrue(all(cell["original_gate"] == "PASS" for cell in projected["cells"]))
        self.assertEqual(projected["cost"]["metadata"]["charged_seconds"], 181.)

    def test_trusted_closed_copy_keeps_same_canonical_identity_without_hydrating_either_path(self):
        cut = make_cut(self.root)
        cut["final"]["consumed_closed_metadata_ledger"]["path"] = "/private/not-present/closed-ledger.json"
        cut["final"]["ledger_input_is_copy"] = True
        refresh(cut)
        self.assertEqual(self.latest()["authoritative_final_metadata_ledger"]["path"], native3.CANONICAL_LEDGER)
        self.assertEqual(self.latest()["consumed_closed_metadata_ledger"]["path"], "/private/not-present/closed-ledger.json")

    def test_missing_unsafe_and_changed_goal_inputs_fail_closed(self):
        for role in ("result", "final_cost", "verification", "readout", "protocol", "parent_metadata_anchor", "media"):
            with self.subTest(role=role):
                cut = make_cut(self.root)
                registry = read_json(self.root / memory.PASSIVE_REGISTRY)
                entry = registry["publications"][0]
                pin = entry["media"][0] if role == "media" else entry[role]
                (self.root / pin["path"]).write_bytes(b"changed pinned input")
                with self.assertRaises(ValueError): self.latest()
        make_cut(self.root)
        registry = read_json(self.root / memory.PASSIVE_REGISTRY)
        registry["publications"][0]["media"][0]["path"] = "../outside.gif"
        atomic_json(self.root / memory.PASSIVE_REGISTRY, registry)
        with self.assertRaises(ValueError): self.latest()

    def test_operation_does_not_open_raw_or_live_provenance_or_import_science(self):
        make_cut(self.root)
        original = Path.read_bytes
        attempts = []
        before = set(sys.modules)

        def guarded(path):
            if not Path(path).resolve().is_relative_to(self.root):
                attempts.append(str(path))
                raise AssertionError("Uncommitted/raw/live input access")
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            self.latest()
        self.assertEqual(attempts, [])
        imported = set(sys.modules) - before
        self.assertFalse(any(name.split(".", 1)[0] in {"torch", "numpy", "particlegan", "benchmarks"} for name in imported))

    def test_full19_synthetic_history_is_unchanged_with_separate_native3_schema(self):
        spec = importlib.util.spec_from_file_location("private_old_passive_fixture", ROOT / "tests/test_forge_passive_publications.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        old.make_cut(self.root)
        old_registry = read_json(self.root / memory.PASSIVE_REGISTRY)
        old_projection = deepcopy(memory.load_passive_publications(self.root))
        old_records = deepcopy(memory.normalize(self.root))
        old_entry = old_registry["publications"][0]
        old_bytes = (self.root / old_entry["result"]["path"]).read_bytes()
        make_cut(self.root)
        registry = read_json(self.root / memory.PASSIVE_REGISTRY)
        registry["publications"].insert(0, old_entry)
        registry["latest"].update(old_registry["latest"])
        atomic_json(self.root / memory.PASSIVE_REGISTRY, registry)
        projection = memory.load_passive_publications(self.root)
        self.assertEqual(projection["latest"][memory.PR223_SCHEMA], old_projection["latest"][memory.PR223_SCHEMA])
        self.assertEqual([record for record in memory.normalize(self.root) if record["study_id"] == "pr223-original-full19-retest"], old_records)
        self.assertEqual((self.root / old_entry["result"]["path"]).read_bytes(), old_bytes)
        self.assertEqual(len(projection["latest"][native3.SCHEMA]["cells"]), 3)


if __name__ == "__main__":
    unittest.main()

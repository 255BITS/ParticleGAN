"""Model-free durability, deadline and maintained recovery controls."""

from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pr223_budget_ledger_controls", HERE / "budget_ledger.py")
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def rows():
    return {f"original-{index:02d}": {"status": "NOT_RUN"} for index in range(19)}


def completed(paid, *, exit_code=0):
    return {"attempt_status": "completed", "paid_wall_seconds": paid,
            "child_returncode": exit_code, "token": "synthetic-owned-token"}


class MetadataControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "shared-metadata.json"
        self.clock = Clock()
        self.budget = ledger.SharedMetadataLedger(self.path)
        self.budget._clock = self.clock

    def test_commands_share_one_cap_and_final_write_tail(self):
        with self.budget.phase("prepare"):
            self.clock.advance(30)
        another = ledger.SharedMetadataLedger(self.path)
        another._clock = self.clock
        with another.phase("copied-source-preflight"):
            self.clock.advance(40)
        with another.phase("publication-write"):
            self.clock.advance(2)
            in_phase = another.snapshot()
            self.clock.advance(3)  # Writing the study after its metadata snapshot.
        final = ledger.SharedMetadataLedger(self.path).snapshot()
        self.assertEqual(in_phase["charged_seconds"], 72)
        self.assertEqual(final["charged_seconds"], 75)
        self.assertEqual(len(final["phases"]), 3)
        self.assertIsNone(final["current_phase"])

    def test_only_launch_wait_is_paused_and_timer_resumes_remaining_cap(self):
        calls = []
        actual = signal.setitimer

        def timer(kind, seconds, interval=0):
            calls.append(seconds)
            return actual(kind, seconds, interval)

        with patch.object(ledger.signal, "setitimer", side_effect=timer):
            with self.budget.phase("run"):
                self.clock.advance(20)
                with self.budget.pause():
                    self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
                    self.clock.advance(7000)
                self.clock.advance(10)
        result = self.budget.snapshot()
        self.assertEqual(result["paid_wall_seconds"], 30)
        self.assertEqual(result["phases"][0]["paused_wall_seconds"], 7000)
        self.assertIn(160.0, calls)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_pause_without_phase_refused(self):
        with self.assertRaises(ledger.BudgetError):
            with self.budget.pause():
                self.fail("no launch wait outside a bounded phase")

    def test_nested_phase_refused(self):
        with self.budget.phase("run"):
            with self.assertRaises(ledger.BudgetError):
                with self.budget.phase("nested"):
                    self.fail("nested phase would evade accounting")

    def test_nested_pause_refused(self):
        with self.budget.phase("run"):
            with self.budget.pause():
                with self.assertRaises(ledger.BudgetError):
                    with self.budget.pause():
                        self.fail("nested pause would evade accounting")

    def test_normal_error_closes_known_phase_and_charges_it(self):
        with self.assertRaisesRegex(ValueError, "source error"):
            with self.budget.phase("verification"):
                self.clock.advance(8)
                raise ValueError("source error")
        result = self.budget.snapshot()
        self.assertEqual(result["paid_wall_seconds"], 8)
        self.assertEqual(result["phases"][0]["status"], "ERROR")
        self.assertFalse(result["blocked"])

    def test_keyboard_interrupt_reserves_remaining_metadata_and_blocks_restart(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.budget.phase("run"):
                self.clock.advance(10)
                raise KeyboardInterrupt
        result = ledger.SharedMetadataLedger(self.path).snapshot()
        self.assertEqual(result["paid_wall_seconds"], 10)
        self.assertEqual(result["reserved_seconds"], 170)
        self.assertEqual(result["charged_seconds"], 180)
        with self.assertRaises(ledger.InterruptedMetadata):
            with ledger.SharedMetadataLedger(self.path).phase("run-again"):
                self.fail("interrupted accounting cannot reset")

    def test_sigalarm_interrupts_a_real_active_phase_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(ledger.BudgetExceeded):
            with self.budget.phase("run"):
                os.kill(os.getpid(), signal.SIGALRM)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        result = self.budget.snapshot()
        self.assertEqual(result["charged_seconds"], 180)
        self.assertTrue(result["blocked"])

    def test_real_timer_expires_without_a_caller_checkpoint(self):
        # Shorten only this private synthetic ledger's fixed cap to exercise
        # actual expiry; no production file, queue, or scientific cap changes.
        with patch.object(ledger, "METADATA_CAP_SECONDS", 0.03):
            budget = ledger.SharedMetadataLedger(self.path)
            start = time.monotonic()
            with self.assertRaises(ledger.BudgetExceeded):
                with budget.phase("bounded-synthetic-parent"):
                    time.sleep(1)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.8)
            result = budget.snapshot()
            self.assertTrue(result["blocked"])
            self.assertGreaterEqual(result["charged_seconds"], 0.03)

    def test_active_metadata_overrun_is_durable_and_halts_next_phase(self):
        with self.assertRaises(ledger.BudgetExceeded):
            with self.budget.phase("publication"):
                self.clock.advance(181)
        result = self.budget.snapshot()
        self.assertEqual(result["paid_wall_seconds"], 181)
        self.assertEqual(result["reserved_seconds"], 0)
        self.assertEqual(result["overrun_seconds"], 1)
        with self.assertRaises(ledger.InterruptedMetadata):
            with self.budget.phase("more"):
                self.fail("metadata overrun must halt")

    def test_existing_deadline_is_preserved_and_refused(self):
        signal.setitimer(signal.ITIMER_REAL, 100)
        try:
            with self.assertRaises(ledger.BudgetError):
                with self.budget.phase("run"):
                    self.fail("existing parent deadline cannot be overwritten")
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 90)
            self.assertFalse(self.path.exists())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def test_dead_process_open_phase_reserves_cap(self):
        program = (
            "import importlib.util, os, sys\n"
            "spec=importlib.util.spec_from_file_location('budget',sys.argv[1])\n"
            "mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n"
            "with mod.SharedMetadataLedger(sys.argv[2]).phase('prepare'):\n"
            "    os._exit(9)\n"
        )
        process = subprocess.run([sys.executable, "-c", program, str(HERE / "budget_ledger.py"),
                                  str(self.path)], capture_output=True, timeout=5, check=False)
        self.assertEqual(process.returncode, 9, process.stderr.decode())
        result = self.budget.snapshot()
        self.assertEqual(result["charged_seconds"], 180)
        self.assertEqual(result["phases"][0]["status"], "INTERRUPTED")
        with self.assertRaises(ledger.InterruptedMetadata):
            with self.budget.phase("prepare-again"):
                self.fail("dead process cannot start a fresh cap")

    def test_crash_while_paused_still_blocks_parent_restart(self):
        program = (
            "import importlib.util, os, sys\n"
            "spec=importlib.util.spec_from_file_location('budget',sys.argv[1])\n"
            "mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n"
            "b=mod.SharedMetadataLedger(sys.argv[2])\n"
            "with b.phase('run'):\n"
            "    with b.pause():\n"
            "        os._exit(9)\n"
        )
        process = subprocess.run([sys.executable, "-c", program, str(HERE / "budget_ledger.py"),
                                  str(self.path)], capture_output=True, timeout=5, check=False)
        self.assertEqual(process.returncode, 9, process.stderr.decode())
        result = self.budget.snapshot()
        self.assertEqual(result["charged_seconds"], 180)
        self.assertTrue(result["blocked"])

    def test_deleted_initialized_ledger_cannot_reset_cap(self):
        with self.budget.phase("prepare"):
            self.clock.advance(10)
        self.path.unlink()
        result = ledger.SharedMetadataLedger(self.path).snapshot()
        self.assertEqual(result["charged_seconds"], 180)
        self.assertTrue(result["blocked"])
        with self.assertRaises(ledger.InterruptedMetadata):
            with ledger.SharedMetadataLedger(self.path).phase("prepare-again"):
                self.fail("a missing initialized ledger cannot replenish the cap")

    def test_live_other_parent_cannot_reset_or_mark_owner_interrupted(self):
        program = (
            "import importlib.util, sys\n"
            "spec=importlib.util.spec_from_file_location('budget',sys.argv[1])\n"
            "mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n"
            "with mod.SharedMetadataLedger(sys.argv[2]).phase('run'):\n"
            "    print('locked',flush=True);sys.stdin.readline()\n"
        )
        process = subprocess.Popen([sys.executable, "-c", program, str(HERE / "budget_ledger.py"),
                                    str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            before = self.path.read_bytes()
            with self.assertRaises(ledger.BudgetError):
                self.budget.snapshot()
            self.assertEqual(self.path.read_bytes(), before)
            stdout, stderr = process.communicate(input="done\n", timeout=5)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

    def test_lowered_cap_or_inconsistent_saved_spend_is_rejected(self):
        with self.budget.phase("prepare"):
            self.clock.advance(5)
        original = json.loads(self.path.read_text())
        for field, value in (("cap_seconds", 181), ("aggregate_cap_seconds", 20000),
                             ("paid_wall_seconds", 0), ("reserved_seconds", -1),
                             ("charged_seconds", True), ("current_phase", {})):
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed[field] = value
                self.path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.budget.snapshot()
        self.path.write_text(json.dumps(original))


class CaseAccountingControls(unittest.TestCase):
    def test_numeric_exit_one_is_measured_even_before_certification(self):
        result = ledger.case_cost(300, completed(12, exit_code=1), certified=False)
        self.assertEqual(result["charged_seconds"], 12)
        self.assertEqual(result["reserved_seconds"], 0)
        self.assertTrue(result["completed_terminal"])

    def test_failed_certification_cannot_turn_completed_cost_into_reservation(self):
        terminal = completed(10)
        self.assertEqual(ledger.case_cost(300, terminal, False)["charged_seconds"],
                         ledger.case_cost(300, terminal, True)["charged_seconds"])

    def test_missing_terminal_reserves_full_allowance(self):
        result = ledger.case_cost(300, None)
        self.assertEqual(result["paid_wall_seconds"], 0)
        self.assertEqual(result["reserved_seconds"], 300)
        self.assertEqual(result["unmeasured_interrupt_reserved_seconds"], 300)

    def test_error_cancel_timeout_use_residual_reserve_only(self):
        for status in ("error", "cancelled", "timeout"):
            with self.subTest(status=status):
                result = ledger.case_cost(300, {"attempt_status": status, "paid_wall_seconds": 12})
                self.assertEqual(result["charged_seconds"], 300)
                self.assertEqual(result["reserved_seconds"], 288)
                self.assertEqual(result["unmeasured_interrupt_reserved_seconds"], 288)

    def test_real_overrun_paid_is_retained(self):
        for status in ("completed", "timeout", "error"):
            with self.subTest(status=status):
                result = ledger.case_cost(300, {"attempt_status": status, "paid_wall_seconds": 301})
                self.assertEqual(result["charged_seconds"], 301)
                self.assertEqual(result["reserved_seconds"], 0)
                self.assertEqual(result["overrun_seconds"], 1)

    def test_unknown_supervisor_status_cannot_become_conservative_evidence(self):
        for status in ("interrupted", "COMPLETE", "missing", "running", "forged"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                ledger.case_cost(300, {"attempt_status": status, "paid_wall_seconds": 1})

    def test_invalid_number_types_and_false_certification_rejected(self):
        for value in (True, -1, math.inf, math.nan, "12"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ledger.case_cost(300, completed(value))
        for allowance in (False, 0, -1, math.inf):
            with self.assertRaises(ValueError):
                ledger.case_cost(allowance, None)
        with self.assertRaises(ValueError):
            ledger.case_cost(300, None, certified=True)
        with self.assertRaises(ValueError):
            ledger.case_cost(300, {}, certified=False)
        with self.assertRaises(ValueError):
            ledger.case_cost(300, completed(1), certified=1)

    def test_exact_maintained_released_attempt_matches_seven_cases(self):
        # Import only the maintained coordinator definition, with no instance,
        # shared queue, model, Torch, scorer, or execution constructed.
        root = HERE.parents[2]
        imported_before = set(sys.modules)
        sys.path.insert(0, str(root))
        try:
            from experiments.forge.policy_execution import _released_attempt
        finally:
            sys.path.remove(str(root))
        imported_after = set(sys.modules)
        self.assertFalse(any(name == "torch" or name.startswith("torch.")
                             for name in imported_after - imported_before))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            terminal_path = directory / "supervisor-terminal.json"
            for status, paid, exit_code in (
                ("completed", 12, 0), ("completed", 12, 1), ("error", 12, 2),
                ("cancelled", 12, -9), ("timeout", 12, -9), (None, 0, None),
                ("timeout", 301, -9),
            ):
                with self.subTest(status=status, paid=paid, exit_code=exit_code):
                    terminal = None if status is None else {
                        "attempt_status": status, "paid_wall_seconds": paid,
                        "child_returncode": exit_code, "token": "synthetic-owned-token",
                    }
                    if terminal is None:
                        terminal_path.unlink(missing_ok=True)
                    else:
                        terminal_path.write_text(json.dumps(terminal))
                    entry = {"lease_path": str(directory / "execution.lease"),
                             "token": "synthetic-owned-token", "allowance_seconds": 300}
                    _released_attempt(entry)
                    ours = ledger.case_cost(300, terminal)
                    self.assertEqual(ours["charged_seconds"], entry["charged_seconds"])
                    self.assertEqual(ours["paid_wall_seconds"] +
                                     ours["unmeasured_interrupt_reserved_seconds"],
                                     entry["charged_seconds"])


class FullReservationControls(unittest.TestCase):
    def setUp(self):
        self.snapshot = ledger._initial()
        self.rows = rows()

    def test_all_nineteen_rows_remain_and_full_allowance_fits_exact_boundary(self):
        self.rows["original-00"].update(ledger.case_cost(10300, completed(10300)))
        result = ledger.require_next_reservation(self.rows, self.snapshot, 500)
        self.assertEqual(result["charged_seconds"], 10300)
        self.assertEqual(result["remaining_seconds"], 500)
        with self.assertRaises(ledger.BudgetExceeded):
            ledger.require_next_reservation(self.rows, self.snapshot, 501)

    def test_reservation_and_metadata_count_once_not_just_measured_paid(self):
        self.rows["original-00"].update(ledger.case_cost(300, {"attempt_status": "timeout", "paid_wall_seconds": 10}))
        self.snapshot["phases"] = [{"index": 0, "name": "prepare", "status": "COMPLETE",
                                    "paid_wall_seconds": 20.0, "paused_wall_seconds": 0.0}]
        ledger._update_totals(self.snapshot)
        result = ledger.require_next_reservation(self.rows, self.snapshot, 100)
        self.assertEqual(result["case_paid_wall_seconds"], 10)
        self.assertEqual(result["case_reserved_seconds"], 290)
        self.assertEqual(result["charged_seconds"], 320)

    def test_full_next_allowance_cannot_be_reduced_to_remaining_budget(self):
        self.rows["original-00"].update(ledger.case_cost(10510, completed(10510)))
        with self.assertRaises(ledger.BudgetExceeded):
            ledger.require_next_reservation(self.rows, self.snapshot, 300)

    def test_missing_denominator_duplicate_ids_or_missing_retained_costs_refused(self):
        with self.assertRaises(ValueError):
            ledger.require_next_reservation(dict(list(self.rows.items())[:18]), self.snapshot, 1)
        repeated = [{"id": "same", "status": "NOT_RUN"} for _ in range(19)]
        with self.assertRaises(ValueError):
            ledger.require_next_reservation(repeated, self.snapshot, 1)
        self.rows["original-00"]["status"] = "COMPLETE"
        with self.assertRaises(ValueError):
            ledger.require_next_reservation(self.rows, self.snapshot, 1)

    def test_partial_or_inconsistent_costs_fail_closed(self):
        honest = ledger.case_cost(300, {"attempt_status": "timeout", "paid_wall_seconds": 10})
        changes = (
            ("paid_wall_seconds", -1), ("charged_seconds", 10),
            ("reserved_seconds", 300), ("unmeasured_interrupt_reserved_seconds", 300),
            ("overrun_seconds", 1), ("completed_terminal", True),
            ("certified", True), ("allowance_seconds", True),
            ("terminal_status", "completed"), ("paid_wall_seconds", math.nan),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value):
                current = rows()
                current["original-00"].update(honest)
                current["original-00"][field] = value
                with self.assertRaises(ValueError):
                    ledger.require_next_reservation(current, self.snapshot, 1)
        for field in honest:
            if field in {"reserved_seconds", "unmeasured_interrupt_reserved_seconds"}:
                continue  # One reserve alias is sufficient, both must agree if present.
            current = rows()
            current["original-00"].update(honest)
            del current["original-00"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                ledger.require_next_reservation(current, self.snapshot, 1)

    def test_case_overrun_remains_visible_in_final_accounting_but_stops_admission(self):
        self.rows["original-00"].update(ledger.case_cost(300, completed(301)))
        final = ledger.require_next_reservation(self.rows, self.snapshot, 0)
        self.assertEqual(final["charged_seconds"], 301)
        self.assertEqual(final["overrun_seconds"], 1)
        self.assertTrue(final["halt_required"])
        with self.assertRaises(ledger.BudgetExceeded):
            ledger.require_next_reservation(self.rows, self.snapshot, 1)

    def test_total_ceiling_overrun_is_preserved_never_clamped(self):
        self.rows["original-00"].update(ledger.case_cost(10900, completed(10900)))
        final = ledger.require_next_reservation(self.rows, self.snapshot, 0)
        self.assertEqual(final["charged_seconds"], 10900)
        self.assertEqual(final["remaining_seconds"], -100)
        self.assertFalse(final["within_cap"])
        self.assertTrue(final["halt_required"])
        with self.assertRaises(ledger.BudgetExceeded):
            ledger.require_next_reservation(self.rows, self.snapshot, 1)

    def test_interrupted_metadata_reserve_cannot_be_ignored(self):
        self.snapshot.update(blocked=True, reason="interrupted phase")
        ledger._update_totals(self.snapshot)
        self.assertEqual(ledger.require_next_reservation(self.rows, self.snapshot, 0)["charged_seconds"], 180)
        with self.assertRaises(ledger.InterruptedMetadata):
            ledger.require_next_reservation(self.rows, self.snapshot, 300)

    def test_tampered_metadata_or_nonfinite_allowance_refused(self):
        self.snapshot["charged_seconds"] = 1
        with self.assertRaises(ValueError):
            ledger.require_next_reservation(self.rows, self.snapshot, 300)
        for value in (True, -1, math.inf, math.nan, "300"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ledger.require_next_reservation(self.rows, ledger._initial(), value)


if __name__ == "__main__":
    unittest.main(verbosity=2)

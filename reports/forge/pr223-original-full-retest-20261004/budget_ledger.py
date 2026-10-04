"""Durable accounting for the one PR223 full-winner retest.

The caller authenticates supervisor tokens and immutable case allowances before
calling these functions.  The ledger owns no queue, model, or execution loop.
Every parent command, including copied-source preflight and publication, must
enter a phase of the same ledger.  Only the maintained coordinator.launch wait
may be enclosed in pause(); its complete elapsed cost is charged as case time.
Parent setup overlapping an admitted case can consequently be charged twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import threading
import time


SCHEMA = "pg_pr223_original_full_retest_metadata_budget_v1"
METADATA_CAP_SECONDS = 180.0
TOTAL_CAP_SECONDS = 10800.0
REQUIRED_CASES = 19
_TERMINAL_STATUSES = {"completed", "timeout", "error", "cancelled"}
_COST_KEYS = {
    "paid_wall_seconds", "reserved_seconds", "unmeasured_interrupt_reserved_seconds",
    "charged_seconds", "overrun_seconds",
}


class BudgetError(RuntimeError):
    """A fixed budget or durable accounting prerequisite refused progress."""


class BudgetExceeded(BudgetError):
    pass


class InterruptedMetadata(BudgetError):
    pass


def _number(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _equal(actual, expected, label):
    actual = _number(actual, label)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} contradicts retained accounting")


def _initial():
    return {
        "schema": SCHEMA, "cap_seconds": METADATA_CAP_SECONDS,
        "aggregate_cap_seconds": TOTAL_CAP_SECONDS, "phases": [],
        "current_phase": None, "blocked": False, "reason": None,
        "paid_wall_seconds": 0.0, "reserved_seconds": 0.0,
        "unmeasured_interrupt_reserved_seconds": 0.0,
        "charged_seconds": 0.0, "overrun_seconds": 0.0,
    }


def _totals(state):
    phases = state["phases"] + ([state["current_phase"]] if state["current_phase"] is not None else [])
    paid = math.fsum(phase["paid_wall_seconds"] for phase in phases)
    reserved = max(0.0, METADATA_CAP_SECONDS - paid) if state["blocked"] else 0.0
    return paid, reserved, paid + reserved, max(0.0, paid - METADATA_CAP_SECONDS)


def _update_totals(state):
    paid, reserved, charged, overrun = _totals(state)
    state.update(paid_wall_seconds=paid, reserved_seconds=reserved,
                 unmeasured_interrupt_reserved_seconds=reserved,
                 charged_seconds=charged, overrun_seconds=overrun)


def _validate_snapshot(state):
    if not isinstance(state, dict) or set(state) != set(_initial()):
        raise ValueError("invalid metadata ledger schema")
    if state["schema"] != SCHEMA:
        raise ValueError("wrong metadata ledger schema")
    if _number(state["cap_seconds"], "metadata cap") != METADATA_CAP_SECONDS:
        raise ValueError("metadata cap changed")
    if _number(state["aggregate_cap_seconds"], "aggregate cap") != TOTAL_CAP_SECONDS:
        raise ValueError("aggregate cap changed")
    if type(state["blocked"]) is not bool or not isinstance(state["phases"], list):
        raise ValueError("invalid metadata phase history")
    if state["blocked"]:
        if not isinstance(state["reason"], str) or not state["reason"]:
            raise ValueError("blocked metadata needs a reason")
    elif state["reason"] is not None:
        raise ValueError("unblocked metadata has a blocking reason")
    phases = state["phases"] + ([state["current_phase"]] if state["current_phase"] is not None else [])
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError("invalid metadata phase")
        keys = {"index", "name", "status", "paid_wall_seconds", "paused_wall_seconds"}
        if set(phase) != keys or type(phase["index"]) is not int or phase["index"] != index:
            raise ValueError("metadata phase sequence changed")
        if not isinstance(phase["name"], str) or not phase["name"].strip():
            raise ValueError("metadata phase name is missing")
        current = index == len(state["phases"])
        statuses = {"ACTIVE", "PAUSED"} if current else {
            "COMPLETE", "ERROR", "INTERRUPTED", "BUDGET_EXCEEDED",
        }
        if phase["status"] not in statuses:
            raise ValueError("invalid metadata phase status")
        _number(phase["paid_wall_seconds"], "phase paid seconds")
        _number(phase["paused_wall_seconds"], "phase paused seconds")
    for key, expected in zip(
        ("paid_wall_seconds", "reserved_seconds", "charged_seconds", "overrun_seconds"),
        _totals(state), strict=True,
    ):
        _equal(state[key], expected, key)
    _equal(state["unmeasured_interrupt_reserved_seconds"], state["reserved_seconds"],
           "metadata interruption reservation")
    if not state["blocked"] and state["paid_wall_seconds"] >= METADATA_CAP_SECONDS:
        raise ValueError("exhausted metadata ledger is not blocked")
    return state


class SharedMetadataLedger:
    """One durable 180-second active-parent clock shared by all commands.

    A nonblocking exclusive lock remains held throughout an active phase,
    including launch waiting.  A lost process leaves the phase sentinel intact;
    the next reader conservatively reserves the rest of the cap and blocks any
    further phase.  No constructor or command can reset that retained state.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._fd = None
        self._locked_fd = None
        self._state = None
        self._last_clock = None
        self._paused = False
        self._timed_out = False
        self._clock = time.monotonic

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BudgetError("another parent metadata phase still owns the ledger") from exc
            anchor = os.pread(fd, 512, 0)
            if anchor not in (b"", (SCHEMA + "\n").encode()):
                raise ValueError("metadata lock identity changed")
            self._locked_fd = fd
            yield fd
        finally:
            self._locked_fd = None
            os.close(fd)

    def _read(self):
        if not self.path.exists():
            state = _initial()
            if os.fstat(self._locked_fd).st_size:
                state.update(blocked=True, reason="initialized metadata ledger is missing; no reset")
                _update_totals(state)
                self._write(state)
            return state
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("metadata ledger must be a regular file")
        return _validate_snapshot(json.loads(self.path.read_text()))

    def _write(self, state):
        _validate_snapshot(state)
        # Retain an initialized marker so deleting the JSON cannot create a
        # fresh 180-second allowance at the same command/ledger path.
        if os.fstat(self._locked_fd).st_size == 0:
            os.write(self._locked_fd, (SCHEMA + "\n").encode())
            os.fsync(self._locked_fd)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _recover(self, state):
        if state["current_phase"] is not None:
            phase = state["current_phase"]
            phase["status"] = "INTERRUPTED"
            state["phases"].append(phase)
            state.update(current_phase=None, blocked=True,
                         reason="interrupted open parent metadata phase; no reset or retry")
            _update_totals(state)
            self._write(state)
        return state

    def _checkpoint(self):
        now = self._clock()
        elapsed = now - self._last_clock
        if not math.isfinite(elapsed) or elapsed < 0:
            raise BudgetError("parent monotonic clock changed")
        phase = self._state["current_phase"]
        key = "paused_wall_seconds" if self._paused else "paid_wall_seconds"
        phase[key] += elapsed
        self._last_clock = now
        _update_totals(self._state)

    def _alarm(self, _signum, _frame):
        self._timed_out = True
        raise BudgetExceeded("shared active-parent metadata cap exhausted")

    def _arm(self):
        remaining = METADATA_CAP_SECONDS - self._state["paid_wall_seconds"]
        if remaining <= 0 or self._timed_out:
            self._timed_out = True
            raise BudgetExceeded("shared active-parent metadata cap exhausted")
        signal.setitimer(signal.ITIMER_REAL, remaining)

    @contextmanager
    def phase(self, name):
        if self._fd is not None:
            raise BudgetError("parent metadata phases cannot nest")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("parent metadata phase needs a name")
        if threading.current_thread() is not threading.main_thread():
            raise BudgetError("metadata deadline requires the main parent thread")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise BudgetError("refusing to replace an existing parent real-time deadline")
        start = self._clock()
        with self._lock() as fd:
            state = self._recover(self._read())
            if state["blocked"]:
                raise InterruptedMetadata(state["reason"])
            state["current_phase"] = {
                "index": len(state["phases"]), "name": name, "status": "ACTIVE",
                "paid_wall_seconds": 0.0, "paused_wall_seconds": 0.0,
            }
            self._fd, self._state, self._last_clock = fd, state, start
            self._paused = self._timed_out = False
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._alarm)
            error = None
            try:
                self._write(state)  # Open sentinel precedes the first caller operation.
                self._checkpoint()
                self._arm()
                yield self
            except BaseException as exc:
                error = exc
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                try:
                    self._checkpoint()
                    interrupted = error is not None and not isinstance(error, Exception)
                    exhausted = self._timed_out or state["paid_wall_seconds"] >= METADATA_CAP_SECONDS
                    phase = state["current_phase"]
                    phase["status"] = (
                        "BUDGET_EXCEEDED" if exhausted else "INTERRUPTED" if interrupted
                        else "ERROR" if error is not None else "COMPLETE"
                    )
                    state["phases"].append(phase)
                    state["current_phase"] = None
                    if interrupted or exhausted:
                        state["blocked"] = True
                        state["reason"] = (
                            "shared active-parent metadata cap exhausted" if exhausted
                            else "interrupted parent metadata phase; no reset or retry"
                        )
                    _update_totals(state)
                    self._write(state)
                    if exhausted and error is None:
                        raise BudgetExceeded(state["reason"])
                finally:
                    signal.signal(signal.SIGALRM, previous_handler)
                    self._fd = self._state = self._last_clock = None
                    self._paused = False

    @contextmanager
    def pause(self):
        """Exclude only waiting inside the already-charged coordinator.launch."""
        if self._fd is None or self._paused:
            raise BudgetError("launch-wait pause requires one active metadata phase")
        self._checkpoint()
        self._arm()  # Refuse a pause entered after metadata is already exhausted.
        self._state["current_phase"]["status"] = "PAUSED"
        self._write(self._state)
        self._checkpoint()  # Entry bookkeeping is active-parent time.
        self._paused = True
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        try:
            yield
        finally:
            self._checkpoint()
            self._paused = False
            self._state["current_phase"]["status"] = "ACTIVE"
            self._write(self._state)
            self._checkpoint()  # Return bookkeeping also consumes metadata time.
            self._arm()

    def snapshot(self):
        if self._fd is not None:
            self._checkpoint()
            if self._state["paid_wall_seconds"] >= METADATA_CAP_SECONDS:
                self._timed_out = True
                raise BudgetExceeded("shared active-parent metadata cap exhausted")
            self._write(self._state)
            return deepcopy(self._state)
        with self._lock():
            return deepcopy(self._recover(self._read()))


def case_cost(allowance, terminal, certified=False):
    """Project a token-matched durable terminal using maintained recovery rules.

    Scientific certification never changes a known completed supervisor's paid
    cost.  A numeric FAIL (child exit 1) is still a completed execution.  Missing,
    errored, cancelled, or timed-out executions reserve the original allowance,
    minus the measured part, rather than adding another complete allowance.
    """
    allowance = _number(allowance, "case allowance", positive=True)
    if type(certified) is not bool:
        raise ValueError("certified must be Boolean")
    if terminal is not None and not isinstance(terminal, Mapping):
        raise ValueError("terminal must be a matching durable mapping or absent")
    paid = _number(terminal.get("paid_wall_seconds", 0.0) if terminal else 0.0,
                   "case paid seconds")
    status = terminal.get("attempt_status") if terminal is not None else "missing"
    if not isinstance(status, str) or not status:
        raise ValueError("durable terminal is missing attempt_status")
    if terminal is not None and status not in _TERMINAL_STATUSES:
        raise ValueError("durable terminal has an unknown attempt_status")
    completed = status == "completed"
    if certified and not completed:
        raise ValueError("an unfinished supervisor cannot have completed certification")
    charged = paid if completed else max(allowance, paid)
    reserved = charged - paid
    return {
        "allowance_seconds": allowance, "terminal_status": status,
        "completed_terminal": completed, "certified": certified,
        "paid_wall_seconds": paid, "reserved_seconds": reserved,
        "unmeasured_interrupt_reserved_seconds": reserved,
        "charged_seconds": charged, "overrun_seconds": max(0.0, paid - allowance),
    }


def require_next_reservation(rows, snapshot, next_allowance):
    """Require all 19 rows and the full next allowance inside the fixed ceiling.

    NOT_RUN/RUNNING rows may omit costs before their attempt is retained.  The
    envelope must first recover any existing durable attempt; this function does
    not infer missing execution costs or grant an admission lease.  A zero next
    allowance is a final accounting projection: even an actual overrun is kept
    visible with halt_required, rather than preventing the durable final write.
    """
    _validate_snapshot(snapshot)
    next_allowance = _number(next_allowance, "next complete allowance")
    if isinstance(rows, Mapping):
        values = list(rows.values())
        if any(not isinstance(key, str) or not key for key in rows):
            raise ValueError("all original case identities must be present")
    else:
        values = list(rows)
        ids = [row.get("id", row.get("case_id")) if isinstance(row, Mapping) else None
               for row in values]
        if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("all original case identities must be unique")
    if len(values) != REQUIRED_CASES:
        raise ValueError("budget accounting requires all 19 original rows")
    paid, reserved, overruns = [], [], []
    for row in values:
        if not isinstance(row, Mapping):
            raise ValueError("invalid original case cost row")
        present = _COST_KEYS.intersection(row)
        if not present:
            if row.get("status") not in {"NOT_RUN", "RUNNING", "UNKNOWN", "BLOCKED"}:
                raise ValueError("retained execution is missing its paid cost")
            continue
        required = {"paid_wall_seconds", "charged_seconds", "overrun_seconds",
                    "allowance_seconds", "completed_terminal", "terminal_status", "certified"}
        if not required.issubset(row) or not ({"reserved_seconds", "unmeasured_interrupt_reserved_seconds"} & row.keys()):
            raise ValueError("retained execution has partial cost accounting")
        measured = _number(row["paid_wall_seconds"], "retained case paid seconds")
        reserve = _number(row.get("reserved_seconds", row.get("unmeasured_interrupt_reserved_seconds")),
                          "retained case reserve")
        if "unmeasured_interrupt_reserved_seconds" in row:
            _equal(row["unmeasured_interrupt_reserved_seconds"], reserve, "case reserve aliases")
        _equal(row["charged_seconds"], measured + reserve, "case charged seconds")
        overrun = _number(row["overrun_seconds"], "case overrun seconds")
        allowance = _number(row["allowance_seconds"], "retained full allowance", positive=True)
        _equal(overrun, max(0.0, measured - allowance), "case overrun")
        completed, certified = row["completed_terminal"], row["certified"]
        if type(completed) is not bool or type(certified) is not bool:
            raise ValueError("retained terminal completion/certification is not Boolean")
        if not isinstance(row["terminal_status"], str) or not row["terminal_status"]:
            raise ValueError("retained terminal status is missing")
        if row["terminal_status"] not in _TERMINAL_STATUSES | {"missing"}:
            raise ValueError("retained terminal status is unknown")
        if completed != (row["terminal_status"] == "completed") or (certified and not completed):
            raise ValueError("retained completion contradicts the supervisor terminal")
        _equal(reserve, 0.0 if completed else max(0.0, allowance - measured),
               "case conservative reservation")
        paid.append(measured)
        reserved.append(reserve)
        overruns.append(overrun)
    case_paid, case_reserved = math.fsum(paid), math.fsum(reserved)
    charged = snapshot["charged_seconds"] + case_paid + case_reserved
    overrun = math.fsum(overruns) + snapshot["overrun_seconds"]
    if next_allowance > 0 and overrun > 0:
        raise BudgetExceeded("retained overrun halts further admission")
    if next_allowance > 0 and charged > TOTAL_CAP_SECONDS:
        raise BudgetExceeded("retained total exceeds the original 10800-second ceiling")
    if next_allowance > 0 and snapshot["blocked"]:
        raise InterruptedMetadata(snapshot["reason"])
    if next_allowance > 0 and charged + next_allowance > TOTAL_CAP_SECONDS:
        raise BudgetExceeded("the complete next allowance does not fit; no truncated launch")
    return {
        "metadata_charged_seconds": snapshot["charged_seconds"],
        "case_paid_wall_seconds": case_paid, "case_reserved_seconds": case_reserved,
        "case_charged_seconds": case_paid + case_reserved,
        "charged_seconds": charged, "cap_seconds": TOTAL_CAP_SECONDS,
        "remaining_seconds": TOTAL_CAP_SECONDS - charged,
        "next_allowance_seconds": next_allowance,
        "overrun_seconds": overrun,
        "within_cap": charged <= TOTAL_CAP_SECONDS,
        "halt_required": snapshot["blocked"] or overrun > 0 or charged > TOTAL_CAP_SECONDS,
    }

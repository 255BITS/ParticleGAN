"""Finite original common26 diagnostic accounting; no queue, source capture or authorization API.

ROOT authenticates the future allocation and supplies byte-trusted CLOSED
predecessor inputs inside its paid phase. This module consumes only supplied
bytes/metadata. The 41288/300 proposal is not an approval. ROOT must pin authenticated Ember
claim/ACK bytes for that exact envelope before creating a ledger or doing actual
work. Loading the maintained ledger source changes isolated constants,
never the original ledger/module. It does not construct a ledger or admit work.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import types


SCHEMA = "pg_common26_original_diagnostic_campaign_metadata_budget_v1"
METADATA_CAP_SECONDS = 300.0
TOTAL_CAP_SECONDS = 41288.0
TASK_ALLOWANCES = (
    ("ae_gan_hold", 300), ("ring16_acquisition", 300), ("mode_hold", 1800),
    ("vector_two_broad", 1800), ("vector_unequal_mass", 1800),
    ("vector_unequal_width", 1800), ("vector_anisotropic", 1800),
    ("vector_overlap", 1800), ("vector_spiral", 1800),
    ("img_stripes2", 1800), ("img_bars4", 1800),
    ("img_blobs4", 1800), ("img_intensity2", 1800),
    ("grid100", 3600), ("rotated100", 3600), ("staggered100", 3600),
    ("ring_hold", 3600),
)
TASK_IDS = tuple(name for name, allowance in TASK_ALLOWANCES)
ALLOWANCES = dict(TASK_ALLOWANCES)
REQUIRED_CASES = len(TASK_IDS)
UNRESOLVED_TASKS = ("unused_token_hold", "five_word_joint_acquisition", "trajectory",
    "residual_student", "unipolar", "cover_leftover", "mid_scale_identity")
SHARED_CONSUMERS = {"ring_extension": "ring_hold"}
SCIENCE_MAXIMUM_SECONDS = float(sum(ALLOWANCES.values()))
assert REQUIRED_CASES == 17 and SCIENCE_MAXIMUM_SECONDS == 34800.0
PRIOR_COMPLETED_CHARGE = 5994.11364284181
FIRST_CASE_PAID_SECONDS = 12.931742924032733
OLD_PARENT_SCHEMA = "pg_common26_first_two_pole_parent_budget_v1"
OLD_PARENT_CAP_SECONDS = 180.0
MAINTAINED_SHA256 = "0c5513e650d00c11a69486da1ff112ba867251779174bfe04de2900180dfa272"
FIRST_PUBLIC_SHA256 = "66551a0018a22cc96ffe88bd27754634ddb5768c5fd896cf6f5dd262e72aa38b"
PLANNED_MAXIMUM_SECONDS = math.fsum((PRIOR_COMPLETED_CHARGE, FIRST_CASE_PAID_SECONDS,
    OLD_PARENT_CAP_SECONDS, METADATA_CAP_SECONDS, SCIENCE_MAXIMUM_SECONDS))
_SEAL = object()
_COST_KEYS = {"paid_wall_seconds", "reserved_seconds", "unmeasured_interrupt_reserved_seconds",
              "charged_seconds", "overrun_seconds"}


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("a ROOT-trusted exact SHA256 is required")
    return value


def _json(raw):
    if type(raw) is not bytes:
        raise ValueError("immutable input bytes are required")
    def no_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(raw, parse_constant=no_constant, object_pairs_hook=unique)


def _isolated(source_bytes, *, schema, metadata_cap, aggregate_cap, required):
    if type(source_bytes) is not bytes or hashlib.sha256(source_bytes).hexdigest() != MAINTAINED_SHA256:
        raise ValueError("maintained ledger source bytes changed")
    module = types.ModuleType("common26_original_diagnostic_maintained_budget")
    module.__file__ = "<bound-maintained-budget-ledger>"
    exec(compile(source_bytes, module.__file__, "exec"), module.__dict__)
    if (module.SCHEMA != "pg_pr223_original_full_retest_metadata_budget_v1"
            or module.METADATA_CAP_SECONDS != 180.0 or module.TOTAL_CAP_SECONDS != 10800.0
            or module.REQUIRED_CASES != 19):
        raise ValueError("maintained ledger identity changed")
    module.SCHEMA, module.METADATA_CAP_SECONDS = schema, metadata_cap
    module.TOTAL_CAP_SECONDS, module.REQUIRED_CASES = aggregate_cap, required
    module._campaign_source_sha256 = MAINTAINED_SHA256
    return module


def load_campaign_budget(source_bytes):
    """Source-only load. ROOT must authenticate approval before any phase call."""
    return _isolated(source_bytes, schema=SCHEMA, metadata_cap=METADATA_CAP_SECONDS,
                     aggregate_cap=TOTAL_CAP_SECONDS, required=REQUIRED_CASES)


def _identity(budget):
    if (getattr(budget, "_campaign_source_sha256", None) != MAINTAINED_SHA256
            or budget.SCHEMA != SCHEMA or budget.METADATA_CAP_SECONDS != METADATA_CAP_SECONDS
            or budget.TOTAL_CAP_SECONDS != TOTAL_CAP_SECONDS or budget.REQUIRED_CASES != REQUIRED_CASES):
        raise ValueError("campaign accounting source or caps changed")


@dataclass(frozen=True)
class Predecessors:
    first_parent_sha256: str
    first_public_sha256: str
    first_parent_charged_seconds: float
    first_parent_phase_count: int
    _closed_state: dict = field(repr=False)
    _closed_state_digest: str = field(repr=False)
    _seal: object = field(repr=False)


def bind_predecessors(source_bytes, first_parent_bytes, first_public_bytes, *,
                      expected_first_parent_sha256):
    """Validate ROOT byte-bound closed evidence; never open a LIVE path.

    The immutable first public cut binds the already-completed two-pole result.
    Its measured cost is the ROOT-attested fixed amount above, not a new reserve.
    The older four ledgers are included in PRIOR_COMPLETED_CHARGE, not added here.
    """
    expected = _sha(expected_first_parent_sha256)
    if type(first_parent_bytes) is not bytes or hashlib.sha256(first_parent_bytes).hexdigest() != expected:
        raise ValueError("ROOT-captured final first-parent bytes changed")
    if type(first_public_bytes) is not bytes or hashlib.sha256(first_public_bytes).hexdigest() != FIRST_PUBLIC_SHA256:
        raise ValueError("retained first-case public cut changed")
    old = _isolated(source_bytes, schema=OLD_PARENT_SCHEMA, metadata_cap=OLD_PARENT_CAP_SECONDS,
                    aggregate_cap=10800.0, required=1)
    state = old._validate_snapshot(_json(first_parent_bytes))
    if (state["current_phase"] is not None or state["blocked"]
            or len(state["phases"]) < 16
            or state["charged_seconds"] < 81.90421384293586
            or any(state[key] != 0 for key in ("reserved_seconds", "unmeasured_interrupt_reserved_seconds", "overrun_seconds"))
            or any(p["status"] not in {"COMPLETE", "ERROR"} for p in state["phases"])):
        raise ValueError("first-parent boundary is not the retained unblocked CLOSED continuation")
    record = _json(first_public_bytes)
    result = record.get("result", {})
    certificate = result.get("certificate", {})
    if (record.get("schema") != "pg_canonical_two_pole_first_case_public_v1"
            or record.get("task_id") != "two_pole" or result.get("status") != "FAIL"
            or certificate.get("full_protocol_complete") is not True):
        raise ValueError("the retained first case must stay its completed FAIL")
    state_digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
    return Predecessors(expected, FIRST_PUBLIC_SHA256, state["charged_seconds"],
                        len(state["phases"]), deepcopy(state), state_digest, _SEAL)


def _predecessors(value):
    if (type(value) is not Predecessors or value._seal is not _SEAL
            or value.first_public_sha256 != FIRST_PUBLIC_SHA256
            or value.first_parent_charged_seconds != value._closed_state["charged_seconds"]
            or value.first_parent_phase_count != len(value._closed_state["phases"])
            or hashlib.sha256(json.dumps(value._closed_state, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest() != value._closed_state_digest):
        raise ValueError("verified immutable predecessor cut is required")
    _sha(value.first_parent_sha256)


def case_cost(budget, terminal, *, allowance_seconds, expected_token=None, certified=False):
    """Completed PASS/FAIL/nonzero exits pay measured cost; interruptions reserve."""
    _identity(budget)
    if terminal is not None:
        if (type(terminal) is not dict or type(expected_token) is not str or not expected_token
                or terminal.get("token") != expected_token):
            raise ValueError("durable attempt token is stale or missing")
        if terminal.get("attempt_status") == "completed":
            budget._number(terminal.get("paid_wall_seconds"), "completed case paid", positive=True)
    if type(allowance_seconds) is not int or allowance_seconds not in set(ALLOWANCES.values()):
        raise ValueError("the full source-owned task allowance is required")
    return budget.case_cost(allowance_seconds, terminal, certified=certified)


def _rows(rows):
    if type(rows) is not list or [r.get("id") if type(r) is dict else None for r in rows] != list(TASK_IDS):
        raise ValueError("all seventeen ordered original physical case rows are required")
    for row in rows:
        task_id = row["id"]
        if "task_id" in row and row["task_id"] != task_id:
            raise ValueError("foreign physical task identity")
        if _COST_KEYS.intersection(row):
            if row.get("allowance_seconds") != ALLOWANCES[task_id]:
                raise ValueError("the unchanged full task allowance is required")
        elif row.get("status") == "BLOCKED":
            dependency = row.get("source_bound_dependency", {})
            if (task_id != "ring_hold" or row.get("blocker_kind") != "dependency" or row.get("models") != 0
                    or row.get("case_reservation_seconds") != 0
                    or not isinstance(dependency, dict) or dependency.get("status") != "BLOCKED"
                    or dependency.get("dependency") != {"task": "mode_hold", "kind": "gate"}):
                raise ValueError("a zero-attempt genuine source-bound dependency blocker is required")
        elif row.get("status") != "NOT_RUN":
            raise ValueError("admitted cases need their retained cost before projection")
        if not _COST_KEYS.intersection(row) and any(row.get(key) for key in
                ("attempt_id", "attempt_key", "attempt_token", "started_monotonic")):
            raise ValueError("an actual attempt cannot be erased into a zero-cost row")
    return deepcopy(rows)


def inclusive_accounting(budget, snapshot, *, predecessors, rows):
    """Each physical group is charged once; unsupported/dependency rows reserve zero."""
    _identity(budget)
    _predecessors(predecessors)
    values = _rows(rows)
    current = budget.require_next_reservation(values, snapshot, 0)
    prior = math.fsum((PRIOR_COMPLETED_CHARGE, FIRST_CASE_PAID_SECONDS,
                      predecessors.first_parent_charged_seconds))
    charged = math.fsum((prior, current["charged_seconds"]))
    halt = (current["halt_required"] or charged > TOTAL_CAP_SECONDS
        or any(row.get("status") not in {"NOT_RUN", "BLOCKED", "PASS", "FAIL"}
               or (row.get("status") in {"PASS", "FAIL"} and row.get("certified") is not True)
               for row in values))
    return {"schema": "pg_common26_original_diagnostic_inclusive_accounting_v1",
        "prior_completed_charged_seconds": PRIOR_COMPLETED_CHARGE,
        "first_case_paid_wall_seconds": FIRST_CASE_PAID_SECONDS,
        "first_case_reserved_seconds": 0.0,
        "first_parent_charged_seconds": predecessors.first_parent_charged_seconds,
        "first_parent_sha256": predecessors.first_parent_sha256,
        "first_public_sha256": predecessors.first_public_sha256,
        "current_case_paid_wall_seconds": current["case_paid_wall_seconds"],
        "current_case_reserved_seconds": current["case_reserved_seconds"],
        "current_case_charged_seconds": current["case_charged_seconds"],
        "metadata_charged_seconds": snapshot["charged_seconds"],
        "charged_seconds": charged, "cap_seconds": TOTAL_CAP_SECONDS,
        "remaining_seconds": TOTAL_CAP_SECONDS - charged,
        "overrun_seconds": current["overrun_seconds"],
        "aggregate_overrun_seconds": max(0.0, charged - TOTAL_CAP_SECONDS),
        "within_cap": charged <= TOTAL_CAP_SECONDS, "halt_required": bool(halt),
        "planned_maximum_charged_seconds": PLANNED_MAXIMUM_SECONDS,
        "old_charges_added_once": True, "blocked_case_reservations": 0,
        "required_original_slots": 26, "current_physical_attempt_slots": REQUIRED_CASES,
        "current_supported_scientific_slots": 18,
        "ring_allowance_rule": "one uninterrupted producer, max3600 once; extension is a consumer",
        "unresolved_original_task_ids": list(UNRESOLVED_TASKS),
        "qualification_credit": False, "default_adoption": False, "speed_ranking": False,
        "science_admission_granted_by_accounting": False}


def require_next_case_fit(budget, snapshot, *, predecessors, rows, next_task_id):
    """Require the complete remaining finite envelope, with no partial/retry launch."""
    values = _rows(rows)
    if next_task_id not in ALLOWANCES:
        raise ValueError("completed, unsupported and shared-consumer tasks are not physical cases")
    index = TASK_IDS.index(next_task_id)
    for row in values[:index]:
        if row.get("status") == "BLOCKED":
            continue
        if (row.get("status") not in {"PASS", "FAIL"} or row.get("certified") is not True
                or row.get("completed_terminal") is not True
                or row.get("terminal_status") != "completed"):
            raise ValueError("only accepted complete PASS/FAIL or actual dependency blocks permit continuation")
    for row in values[index:]:
        if row.get("status") != "NOT_RUN" or _COST_KEYS.intersection(row):
            raise ValueError("no retries, overlaps or unrecorded downstream attempts")
    result = inclusive_accounting(budget, snapshot, predecessors=predecessors, rows=values)
    if result["halt_required"]:
        raise budget.BudgetExceeded("retained invalidity/interruption/overrun/cap requires a halt")
    remaining_allowances = math.fsum(ALLOWANCES[row["id"]] for row in values[index:])
    worst = math.fsum((result["charged_seconds"],
        max(0.0, METADATA_CAP_SECONDS - snapshot["charged_seconds"]), remaining_allowances))
    if worst > TOTAL_CAP_SECONDS:
        raise budget.BudgetExceeded("the full remaining source-bound cases and metadata do not fit")
    return {**result, "next_task_id": next_task_id,
        "next_allowance_seconds": ALLOWANCES[next_task_id],
        "remaining_envelope_maximum_seconds": worst,
        "full_remaining_envelope_fits": True}

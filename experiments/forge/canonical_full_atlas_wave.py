"""Pure wave-fit seam for the SAME source-bound campaign_budget418 module.

Caller supplies the already verified campaign module. Its maintained budget,
Predecessors class/seal, row validation and inclusive accounting remain exact.
No source reads, ledger construction, authorization, reservation or admission.
"""
from copy import deepcopy
import hashlib
import json
import math


def require_wave_fit(budget, snapshot, *, campaign, predecessors, rows, wave_task_ids, readiness):
    """Pure full-envelope check for ROOT's explicit one/two READY physical cases.

    ROOT validates actual devices, model cohort, source and dependencies before
    either admission. This function never admits or reserves; the CPU AE runs
    solo. Resolved cases may occupy any original slot, while all other slots
    retain NOT_RUN. The ring consumer never becomes a physical case.
    """
    campaign._identity(budget)
    values = campaign._rows(rows)
    if (type(wave_task_ids) not in (list, tuple) or not 1 <= len(wave_task_ids) <= 2
            or any(type(name) is not str or name not in campaign.ALLOWANCES for name in wave_task_ids)
            or len(set(wave_task_ids)) != len(wave_task_ids)
            or (len(wave_task_ids) == 2 and "ae_gan_hold" in wave_task_ids)):
        raise ValueError("one/two distinct physical cases are required; CPU AE is solo")
    if (type(readiness) is not dict or set(readiness) != set(wave_task_ids)
            or any(type(readiness[name]) is not dict or readiness[name].get("status") != "READY"
                   for name in wave_task_ids)):
        raise ValueError("each explicit wave member requires its current READY prerequisite check")
    by_id = {row["id"]: row for row in values}
    for row in values:
        if row["status"] == "BLOCKED":
            if campaign._COST_KEYS.intersection(row):
                raise ValueError("BLOCKED cannot erase a retained physical attempt cost")
            continue  # _rows has validated the genuine zero-attempt dependency block.
        if row["status"] == "NOT_RUN":
            if campaign._COST_KEYS.intersection(row) or row.get("certified") is True or row.get("certificate"):
                raise ValueError("NOT_RUN cannot conceal a cost, active attempt or old grade")
            continue
        if (row["status"] not in {"PASS", "FAIL"} or row.get("certified") is not True
                or row.get("completed_terminal") is not True or row.get("terminal_status") != "completed"):
            raise ValueError("existing attempts must be accepted complete PASS/FAIL; no active wave overlap")
    for name in wave_task_ids:
        if by_id[name]["status"] != "NOT_RUN":
            raise ValueError("a wave cannot retry, reuse or select an already resolved case")
    if "ring_hold" in wave_task_ids:
        prerequisite = by_id["mode_hold"]
        certificate = prerequisite.get("certificate", {})
        if (prerequisite["status"] != "PASS" or prerequisite.get("certified") is not True
                or type(certificate) is not dict or certificate.get("full_protocol_complete") is not True
                or certificate.get("grade", {}).get("status") != "PASS"):
            raise ValueError("ring requires this campaign's already accepted mode_hold PASS, not a same-wave intent")
    result = campaign.inclusive_accounting(budget, snapshot, predecessors=predecessors, rows=values)
    if result["halt_required"]:
        raise budget.BudgetExceeded("retained invalidity/interruption/overrun/cap requires a halt")
    remaining_allowances = math.fsum(campaign.ALLOWANCES[row["id"]] for row in values if row["status"] == "NOT_RUN")
    worst = math.fsum((result["charged_seconds"],
        max(0.0, campaign.METADATA_CAP_SECONDS - snapshot["charged_seconds"]), remaining_allowances))
    if worst > campaign.TOTAL_CAP_SECONDS:
        raise budget.BudgetExceeded("all remaining source-bound cases and metadata must fit before either admission")
    row_digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    return {**result, "wave_task_ids": list(wave_task_ids),
        "wave_allowance_seconds": {name: campaign.ALLOWANCES[name] for name in wave_task_ids},
        "wave_maximum_case_charged_seconds": math.fsum(campaign.ALLOWANCES[name] for name in wave_task_ids),
        "readiness": deepcopy(readiness), "physical_rows_sha256": row_digest,
        "remaining_case_allowances_seconds": remaining_allowances,
        "remaining_envelope_maximum_seconds": worst,
        "full_remaining_envelope_fits": True, "wave_fit_grants_admission": False}

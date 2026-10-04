"""Discover declared techniques and submit a bounded ordinary Forge inventory.

This is a batch interface to the existing planner and queue, not a second
execution lane. Tier prerequisites, frozen recipes and scientific reuse retain
their ordinary semantics. Unsupported declarations remain visible in the plan
without manufacturing an attempt or spending on known blocked tasks.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import identifier, positive_number, read_json, stable_hash
from .execution_policy import completes_tier, group_blockers
from .planning import plan_summary, resolve_idea
from .queue import Queue, drain


DEFAULT_CAMPAIGN = Path("configs/forge/campaigns/technique-inventory.json")


def discover_techniques(root: Path) -> list[str]:
    """New idea cards enter the next inventory without a maintained name list."""
    # Saved tuning trials are evidence variants; only an explicit search runs
    # them. Expanding an ordinary inventory must not rerun every historical grid.
    names = sorted(path.stem for path in (Path(root) / "configs/forge/ideas").glob("*.json"))
    if not names:
        raise ValueError("technique inventory requires at least one declared idea")
    return names


def _campaign(root, campaign):
    definition = deepcopy(campaign) if isinstance(campaign, dict) else read_json(Path(root) / campaign)
    identifier(definition["id"], "campaign")
    positive_number(definition["budget_seconds"], "campaign budget_seconds")
    positive_number(definition["candidate_budget_seconds"], "candidate_budget_seconds")
    return definition


def _first_task_blockers(request):
    assignments = sorted((a for a in request["view"]["assignments"]
                          if a["qualification_tier"] <= request["through_tier"]),
                         key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"]))
    if not assignments:
        return []
    if completes_tier(request):
        # Reject only a wholly blocked initial tier. One unsupported task must
        # not erase independent measurements from this ordinary batch.
        tier = assignments[0]["qualification_tier"]
        by_task = {member: job for job in request["jobs"]
                   for member in job.get("task_ids", [job["task_id"]])}
        first_tier = [item for item in assignments if item["qualification_tier"] == tier]
        reasons = []
        for item in first_tier:
            blocked = group_blockers(request, by_task[item["task"]])
            if not blocked:
                return []
            reasons.extend(blocked)
        return list(dict.fromkeys(reasons))
    first = assignments[0]
    if first["importance"] != "required":
        return []
    return [f"{first['task']}: {reason}" for reason in
            request["tasks"][first["task"]].get("preflight_blockers", [])]


def _signature(request):
    value = {key: request[key] for key in (
        "candidate", "candidate_revision", "policy_fingerprint", "tasks", "jobs", "through_tier",
        "protocol", "rng", "runtime", "execution_backend", "compute_profiles")}
    if "execution_policy" in request:
        value["execution_policy"] = request["execution_policy"]
    return stable_hash(value)


def _prepare(root, queue_root, *, view_id, through_tier, execution_backend, cuda_model, campaign, queue):
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    definition = _campaign(root, campaign)
    state = queue.inspect()
    existing = state.get("campaigns", {}).get(definition["id"])
    if existing and existing["definition"] != definition:
        raise ValueError("campaign definition is immutable; use a new campaign id")
    requests = [resolve_idea(root, name, view_id=view_id, through_tier=through_tier,
                            execution_backend=execution_backend, cuda_model=cuda_model)
                for name in discover_techniques(root)]
    sources = {request["source"]["digest"] for request in requests}
    rows = []
    for request in requests:
        row = plan_summary(request, state)
        row["source_digest"] = request["source"]["digest"]
        blockers = list(request["preflight_blockers"]) + _first_task_blockers(request)
        by_task = {member: job for job in request["jobs"] for member in job["task_ids"]}
        permitted = {t["task"] for t in row["tasks"] if t["permitted_by_tier_cap"]}
        row["declared_worst_case_seconds"] = sum(job["budget_seconds"] for job in request["jobs"]
                                                 if set(job["task_ids"]) <= permitted)
        row["submission_blockers"] = blockers
        row["submission_status"] = "BLOCKED" if blockers else "READY"
        row["candidate_budget_covers_declared_ceiling"] = (
            definition["candidate_budget_seconds"] >= row["declared_worst_case_seconds"])
        row["required_tier_totals"] = {
            str(tier): sum(t["qualification_tier"] == tier and t["importance"] == "required"
                           for t in row["tasks"]) for tier in (1, 2, 3)}
        for task in row["tasks"]:
            saved = state.get("jobs", {}).get(by_task[task["task"]]["compatibility_key"], {})
            receipt = next((r for r in (saved.get("result") or {}).get("task_results", [])
                            if r["task_id"] == task["task"]), None)
            task["evidence_status"] = receipt["gate_status"] if receipt else "UNKNOWN"
            task["preflight_status"] = "BLOCKED" if request["preflight_blockers"] or task["blockers"] else "READY"
        rows.append(row)
    declared = sum(row["declared_worst_case_seconds"] for row in rows)
    summary = {"schema_version": 1, "stage": "planned", "view": view_id, "through_tier": through_tier,
               "execution_backend": execution_backend, "source_digests": sorted(sources),
               "source_digest": next(iter(sources)) if len(sources) == 1 else None,
               "campaign": definition, "queue_root": str(queue_root),
               "logs": str(queue_root / "events.jsonl"), "technique_count": len(rows), "candidates": rows,
               "declared_worst_case_seconds": declared,
               "worst_case_seconds": sum(row["worst_case_seconds"] for row in rows
                                         if not row["submission_blockers"]),
               "campaign_budget_covers_declared_ceiling": definition["budget_seconds"] >= declared,
               "guide": "EXPERIMENTATION.md"}
    return requests, summary


def plan_inventory(root: Path, queue_root: Path, *, view_id="discriminator_stability", through_tier=3,
                   execution_backend="cuda", cuda_model=None, campaign=DEFAULT_CAMPAIGN, queue=None) -> dict:
    """Read-only full-denominator task inventory, preflight and declared budgets."""
    queue = queue or Queue(queue_root)
    _, summary = _prepare(root, queue_root, view_id=view_id, through_tier=through_tier,
                          execution_backend=execution_backend, cuda_model=cuda_model,
                          campaign=campaign, queue=queue)
    return summary


def enqueue_inventory(root: Path, queue_root: Path, *, view_id="discriminator_stability", through_tier=3,
                      execution_backend="cuda", cuda_model=None, campaign=DEFAULT_CAMPAIGN, queue=None) -> dict:
    """Freeze every executable row before admitting any request; launch nothing.

    Repeated calls attach to the queue's exact request/job identities. Validation
    errors, source drift and lifecycle refusals propagate rather than becoming
    scientific BLOCKED receipts.
    """
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    if queue.root != queue_root:
        raise ValueError("inventory queue_root differs from the supplied Queue")
    if queue.report_root is None:
        queue.report_root = root / "reports/forge"
    elif queue.report_root.parent.parent != root:
        raise ValueError("inventory Queue report_root differs from its research repository")
    requests, summary = _prepare(root, queue_root, view_id=view_id, through_tier=through_tier,
                                 execution_backend=execution_backend, cuda_model=cuda_model,
                                 campaign=campaign, queue=queue)
    frozen = []
    for request, row in zip(requests, summary["candidates"]):
        if row["submission_blockers"]:
            continue
        resolved = resolve_idea(root, request["candidate"]["id"], view_id=view_id,
                                through_tier=through_tier, execution_backend=execution_backend,
                                cuda_model=cuda_model, queue_root=queue_root, freeze_source=True)
        if (resolved["source"]["digest"] != request["source"]["digest"]
                or _signature(resolved) != _signature(request)):
            raise ValueError("inventory source or scientific declarations changed before submission; plan and submit again")
        frozen.append((resolved, row))
    for request, row in frozen:
        entry = queue.submit(request, summary["campaign"])
        row.update(request_id=entry["request"]["request_id"], submission_status=entry["status"],
                   request=str(queue_root / "queue/requests" / f"{entry['request']['request_id']}.json"))
    summary["stage"] = "enqueued"
    summary["submitted_count"] = len(frozen)
    summary["blocked_count"] = len(requests) - len(frozen)
    return summary


def run_inventory(root: Path, queue_root: Path, *, devices=None, view_id="discriminator_stability", through_tier=3,
                  execution_backend="cuda", cuda_model=None, campaign=DEFAULT_CAMPAIGN, queue=None) -> dict:
    """Submit, then drain only this bounded campaign with ordinary prerequisites."""
    devices = list(devices or (["cpu"] if execution_backend == "cpu" else ["0", "1"]))
    if (execution_backend == "cpu") != (devices == ["cpu"]):
        raise ValueError("inventory execution backend must agree with devices")
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    queue = queue or Queue(queue_root, report_root=root / "reports/forge")
    summary = enqueue_inventory(root, queue_root, view_id=view_id, through_tier=through_tier,
                                execution_backend=execution_backend, cuda_model=cuda_model,
                                campaign=campaign, queue=queue)
    if summary["submitted_count"]:
        drain(queue, devices, campaign=summary["campaign"]["id"])
    state = queue.inspect()
    for row in summary["candidates"]:
        if "request_id" in row:
            entry = state["submissions"][row["request_id"]]
            row.update(submission_status=entry["status"], reason=entry.get("reason"))
    summary["stage"] = "drained"
    summary["campaign_accounting"] = state.get("campaigns", {}).get(summary["campaign"]["id"])
    return summary

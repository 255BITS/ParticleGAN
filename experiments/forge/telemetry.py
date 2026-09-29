"""Measured execution telemetry and read-only automation accounting.

Wall time is not FLOPs. Process/allocator peaks are not whole-host/device peaks.
Missing instrumentation remains unavailable, rather than being reported as zero.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import math
import resource
import statistics
import sys
import time

from .contracts import read_json, stable_hash


PHASES = ("training_updates", "evaluation", "sampling")


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


class PhaseTimer:
    """Exclusive nested wall time; sampling inside evaluation is counted once."""
    def __init__(self, *, clock=None, synchronize=None):
        self.clock = clock or time.perf_counter
        self.synchronize = synchronize or (lambda: None)
        self.seconds, self.calls, self.stack = Counter(), Counter(), []

    @contextmanager
    def measure(self, phase):
        if phase not in PHASES:
            raise ValueError("unknown measured phase")
        self.synchronize()
        frame = {"start": self.clock(), "nested": 0.}
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = max(0., self.clock() - frame["start"])
            self.stack.pop()
            self.seconds[phase] += max(0., elapsed - frame["nested"])
            self.calls[phase] += 1
            if self.stack:
                self.stack[-1]["nested"] += elapsed

    def snapshot(self):
        return {"method": "exclusive_synchronized_wall_time", "coverage": "partial_adapter",
                "phases": {name: {"seconds": self.seconds[name], "calls": self.calls[name]} for name in PHASES},
                "measured_seconds": sum(self.seconds.values()),
                "scope": "Training updates exclude input preparation; evaluation includes metrics and its artifact I/O but excludes separately measured sampling."}


def peak_rss(*, children=False):
    value = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class MemoryProbe:
    def __init__(self, *, device="cpu", torch_module=None):
        self.device, self.torch, self.cuda_error = str(device), torch_module, None
        if self.device.startswith("cuda"):
            try:
                if self.torch is None:
                    raise ValueError("CUDA allocator instrumentation unavailable")
                self.torch.cuda.reset_peak_memory_stats(self.device)
            except Exception as error:
                self.cuda_error = str(error)

    def snapshot(self):
        result = {"process_peak_rss_bytes": peak_rss(), "cpu_scope": "this process lifetime, including imports",
                  "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
                  "cuda_scope": "PyTorch allocator since probe reset; excludes other processes and non-PyTorch allocations"}
        if self.device.startswith("cuda") and self.cuda_error is None:
            try:
                result.update(cuda_peak_allocated_bytes=int(self.torch.cuda.max_memory_allocated(self.device)),
                              cuda_peak_reserved_bytes=int(self.torch.cuda.max_memory_reserved(self.device)))
            except Exception as error:
                self.cuda_error = str(error)
        if self.cuda_error:
            result["cuda_unavailable_reason"] = self.cuda_error
        return result


def normalize_adapter_costs(raw):
    """Legacy hosts report inclusive elapsed time under a misleading field name."""
    def needs_change(value):
        return ("training_seconds" in value.get("cost", {})
                or any(needs_change(member) for member in value.get("task_results", {}).values()))
    if not needs_change(raw):
        return raw
    result = deepcopy(raw)
    cost = result.setdefault("cost", {})
    if "training_seconds" in cost:
        cost["adapter_reported_inclusive_seconds"] = cost.pop("training_seconds")
        cost["component_timing"] = "unavailable; legacy elapsed time includes evaluation and sampling"
    for name, member in result.get("task_results", {}).items():
        result["task_results"][name] = normalize_adapter_costs(member)
    return result


def observed_concurrency(intervals):
    """Half-open process intervals; touching endpoints never imply overlap."""
    points, valid = [], []
    for row in intervals:
        start, end = _number(row.get("started_at")), _number(row.get("finished_at"))
        if start is None or end is None or end < start:
            continue
        valid.append(row)
        if end > start:
            points.extend([(start, 1, row["attempt_id"], row.get("device")),
                           (end, -1, row["attempt_id"], row.get("device"))])
    active, peak, gpu_peak = {}, 0, 0
    for _, delta, name, device in sorted(points):
        if delta < 0:
            active.pop(name, None)
        else:
            active[name] = device
        peak = max(peak, len(active))
        gpu_peak = max(gpu_peak, len({d for d in active.values() if d not in {None, "cpu"}}))
    span = max((r["finished_at"] for r in valid), default=0) - min((r["started_at"] for r in valid), default=0)
    return {"measured_attempts": len(valid), "peak_workers": peak if valid else None,
            "peak_distinct_gpu_devices": gpu_peak if valid else None,
            "mean_active_workers": sum(r["finished_at"] - r["started_at"] for r in valid) / span if span else None,
            "observation_window_seconds": span if valid else None,
            "scope": "Actual supervised-attempt intervals; this is process occupancy, not GPU kernel utilization."}


def _distribution(values, denominator):
    known = [value for value in values if value is not None]
    return {"outcomes": denominator, "measured_outcomes": len(known), "unknown_cost_outcomes": denominator - len(known),
            "mean_seconds": statistics.mean(known) if known else None,
            "median_seconds": statistics.median(known) if known else None,
            "maximum_seconds": max(known) if known else None}


def _recorded_qualification(request, assignments, rows):
    """Observed scheduler attainment under frozen policy, never a live regrade."""
    indexed = defaultdict(list)
    for row in rows:
        indexed[row["task_id"]].append(row)
    statuses = {}
    for assignment in assignments:
        recorded = indexed.get(assignment["task"], [])
        signatures = {stable_hash({key: row.get(key) for key in ("gate_status", "metrics", "reasons")}) for row in recorded}
        statuses[assignment["task"]] = ("NOT_RUN" if not recorded else "INVALID" if len(signatures) != 1
                                        else recorded[0].get("gate_status", "INVALID"))
    qualified, status = 0, "INCOMPLETE"
    tiers = sorted({a["qualification_tier"] for a in assignments})
    if not tiers or tiers != list(range(1, max(tiers) + 1)):
        return {"status": "UNAVAILABLE", "qualified_tier": None, "task_statuses": statuses}
    for tier in tiers:
        required = [statuses[a["task"]] for a in assignments if a["qualification_tier"] == tier and a["importance"] == "required"]
        if not required:
            return {"status": "UNAVAILABLE", "qualified_tier": None, "task_statuses": statuses}
        if all(value == "PASS" for value in required):
            qualified = tier
            status = "PASS"
        else:
            status = next((value for value in ("FAIL", "INVALID", "BLOCKED", "INCOMPLETE", "NOT_RUN") if value in required), "INVALID")
            break
    return {"status": "INCOMPLETE" if status == "NOT_RUN" else status,
            "qualified_tier": qualified, "task_statuses": statuses}


def summarize_automation(root: Path, queue_root: Path | None = None) -> dict:
    """Read canonical receipts and optional queue state; never materialize files.

    Shared evidence costs appear once in global spend. Per-request readout costs
    describe evidence consumed, may overlap, and must not be summed as spending.
    """
    from .knowledge import _attempts, _queue_states
    root = Path(root).resolve()
    attempts, issues = _attempts(root)
    if queue_root is None:
        states = _queue_states(root, attempts)
    else:
        path = Path(queue_root).resolve() / "queue/state.json"
        states = {str(path.parent.parent): read_json(path)} if path.is_file() else {}
    entries, jobs, charges, owners = {}, {}, {}, {}
    for location, state in sorted(states.items()):
        entries.update({(location, name): entry for name, entry in state.get("submissions", {}).items()})
        for key, job in state.get("jobs", {}).items():
            jobs[(location, key)] = job
        for charge in state.get("charges", []):
            name, seconds = charge["attempt_id"], _number(charge.get("seconds"))
            owners[name] = charge.get("owner", {}).get("request")
            if name in charges and charges[name] != seconds:
                issues.append({"attempt_id": name, "reason": "Conflicting canonical queue cost receipts."})
                charges[name] = None
            else:
                charges[name] = seconds
    costs, outcomes, gates, intervals, memory, phase_seconds = {}, Counter(), Counter(), [], defaultdict(list), Counter()
    phases_measured = 0
    selected = defaultdict(list)
    for attempt in attempts:
        name = attempt["attempt_id"]
        directory = root / "reports/forge/attempts" / name
        result = read_json(directory / "result.json")
        terminal = result.get("raw", {})
        valid = attempt["valid_receipt"]
        outcomes[terminal.get("attempt_status", "unrecorded") if valid else "invalid_receipt"] += 1
        seconds = _number(terminal.get("elapsed_seconds")) if valid else None
        if name in charges and seconds is not None and charges[name] != seconds:
            issues.append({"attempt_id": name, "reason": "Queue charge and durable elapsed cost disagree."})
            seconds = None
        elif name in charges:
            seconds = charges[name]
        costs[name] = seconds
        owners.setdefault(name, result.get("cost_owner", {}).get("request"))
        if not valid:
            for row in attempt["task_results"]:
                gates["INVALID"] += 1
                selected[row["compatibility_key"]].append({**row, "gate_status": "INVALID"})
            continue
        interval = terminal.get("telemetry", {}).get("interval")
        if isinstance(interval, dict):
            intervals.append({**interval, "attempt_id": name})
        telemetry = terminal.get("result", {}).get("telemetry", {})
        for key, value in telemetry.get("memory", {}).items():
            if key.endswith("_bytes") and _number(value) is not None:
                memory[key].append(value)
        for key in ("supervisor_peak_rss_bytes", "maximum_reaped_child_peak_rss_bytes"):
            value = terminal.get("telemetry", {}).get(key)
            if _number(value) is not None:
                memory[key].append(value)
        timing = terminal.get("result", {}).get("cost", {}).get("phase_timing")
        if isinstance(timing, dict) and all(_number(timing.get("phases", {}).get(p, {}).get("seconds")) is not None for p in PHASES):
            phases_measured += 1
            for phase in PHASES:
                phase_seconds[phase] += timing["phases"][phase]["seconds"]
        for row in attempt["task_results"]:
            # These are certified frozen evaluator verdicts. Regrading with the
            # current checkout would silently change pinned scientific history.
            gates[row.get("gate_status", "INVALID")] += 1
            diagnostic = (attempt["request"].get("calibration_lane") or attempt["request"].get("promotion")
                          or attempt["request"].get("view", {}).get("evidence_scope") == "calibration_diagnostic")
            if not attempt.get("superseded_by") and not diagnostic:
                selected[row["compatibility_key"]].append(row)
    all_costs = {**charges, **costs}
    spend = {"wall_seconds": sum(all_costs.values()) if all(v is not None for v in all_costs.values()) else None,
             "known_wall_seconds": sum(v for v in all_costs.values() if v is not None),
             "attempts": len(all_costs), "unknown_cost_attempts": sum(v is None for v in all_costs.values()),
             "scope": "Unique paid attempts, including retries; grouped predicates and subscribers are not charged again."}
    # Preserve archived requests when a queue has moved, but mark queue-dependent
    # avoided/reuse denominators unavailable rather than guessing their absence.
    if not entries:
        for attempt in attempts:
            request = attempt["request"]
            entries.setdefault((None, request.get("request_id", stable_hash(request))), {"request": request})
    units, use_counts, demand, unavailable_caps = {}, Counter(), 0, 0
    for (location, request_id), entry in sorted(entries.items(), key=lambda item: str(item[0])):
        request = entry["request"]
        if request.get("calibration_lane") or request.get("promotion"):
            continue
        if type(request.get("through_tier")) is not int or request["through_tier"] not in (1, 2, 3):
            unavailable_caps += 1
            continue
        assignments = [a for a in request["view"]["assignments"] if a["qualification_tier"] <= request["through_tier"]]
        names = {a["task"] for a in assignments}
        definitions = [j for j in request["jobs"] if set(j.get("task_ids", [j["task_id"]])) <= names]
        keys = {j["compatibility_key"] for j in definitions}
        demand += len(keys)
        for key in keys:
            if selected.get(key):
                use_counts[key] += 1
        view = {**request["view"], "assignments": assignments}
        identity = stable_hash({"candidate_revision": request["candidate_revision"], "view": view,
                               "protocol": request.get("protocol"), "keys": sorted(keys)})
        if identity in units:
            units[identity]["request_ids"].append(request_id)
            continue
        rows = [row for key in sorted(keys) for row in selected.get(key, [])]
        try:
            verdict = _recorded_qualification(request, assignments, rows)
        except (KeyError, TypeError, ValueError) as error:
            verdict = {"status": "INVALID", "qualified_tier": 0, "task_statuses": {}}
            issues.append({"request_id": request_id, "reason": "Cannot reduce recorded qualification: " + str(error)})
        used_attempts = {row["_attempt_id"] for row in rows}
        # Charge certified superseded infrastructure attempts too.
        used_attempts.update(a["attempt_id"] for a in attempts if any(r.get("compatibility_key") in keys for r in a["task_results"]))
        values = [costs.get(name) for name in used_attempts]
        cost = sum(values) if values and all(v is not None for v in values) else None
        rejected_tier = verdict.get("qualified_tier", 0) + 1 if verdict["status"] == "FAIL" else None
        avoided_keys = set()
        avoided_tasks = []
        if rejected_tier is not None and location is not None:
            ordered = sorted(assignments, key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"]))
            failed_index = next((i for i, a in enumerate(ordered) if a["importance"] == "required"
                                 and verdict.get("task_statuses", {}).get(a["task"]) == "FAIL"), len(ordered))
            for assignment in ordered[failed_index + 1:]:
                job = next((j for j in definitions if assignment["task"] in j.get("task_ids", [j["task_id"]])), None)
                saved = jobs.get((location, job["compatibility_key"])) if job else None
                if saved and saved["status"] == "pending" and not saved.get("attempts") and not selected.get(job["compatibility_key"]):
                    avoided_tasks.append(assignment["task"])
                    avoided_keys.add(job["compatibility_key"])
        units[identity] = {"candidate_revision": request["candidate_revision"], "view": view.get("id"),
            "through_tier": request["through_tier"], "request_ids": [request_id], "status": verdict["status"],
            "qualified_tier": verdict.get("qualified_tier", 0), "rejected_tier": rejected_tier,
            "evidence_wall_seconds": cost, "attempt_ids": sorted(used_attempts),
            "avoided_tasks": avoided_tasks if states else None, "avoided_execution_jobs": len(avoided_keys) if states else None}
    rows = [units[key] for key in sorted(units)]
    for row in rows:
        owned = [costs.get(name, charges.get(name)) for name, owner in owners.items() if owner in row["request_ids"]]
        unknown_owner = any(owners.get(name) is None for name in row["attempt_ids"])
        row["incremental_paid_seconds"] = None if unknown_owner or any(value is None for value in owned) else sum(owned)
    rejected = [row["incremental_paid_seconds"] for row in rows if row["status"] == "FAIL"]
    qualified = [row["incremental_paid_seconds"] for row in rows if row["status"] == "PASS"]
    incomplete = sum(count for status, count in gates.items() if status in {"INVALID", "INCOMPLETE"})
    errors = sum(outcomes.get(status, 0) for status in ("error", "timeout", "cancelled"))
    concurrency = observed_concurrency(intervals)
    concurrency["missing_interval_attempts"] = len(attempts) - concurrency["measured_attempts"]
    return {"schema_version": 1, "scope": "recorded current Forge receipts; historical imports are excluded",
        "queue_available": bool(states), "queue_locations": sorted(states), "receipt_issues": issues,
        "qualification_scope": "Recorded attainment under each frozen request view and cap; not a current-source regrade or accepted public-default claim.",
        "requests_with_unavailable_cap": unavailable_caps, "public_default_claim": False,
        "spend": spend, "attempt_outcomes": dict(sorted(outcomes.items())),
        "execution_error_rate": {"numerator": errors, "denominator": len(attempts), "fraction": errors / len(attempts) if attempts else None},
        "incomplete_task_rate": {"numerator": incomplete, "denominator": sum(gates.values()), "fraction": incomplete / sum(gates.values()) if gates else None},
        "task_gate_counts": dict(sorted(gates.items())), "qualification_units": rows,
        "cost_to_reject": _distribution(rejected, len(rejected)), "cost_to_qualify": _distribution(qualified, len(qualified)),
        "tier_rejections": dict(sorted(Counter(str(row["rejected_tier"]) for row in rows if row["rejected_tier"] is not None).items())),
        "tasks_avoided": sum(len(row["avoided_tasks"]) for row in rows) if states else None,
        "execution_jobs_avoided": sum(row["avoided_execution_jobs"] for row in rows) if states else None,
        "reuse": {"submitted_execution_demands": demand if states else None, "evidence_uses": sum(use_counts.values()) if states else None,
                  "unique_evidence_jobs": len(use_counts) if states else None,
                  "reused_evidence_uses": sum(use_counts.values()) - len(use_counts) if states else None},
        "concurrency": concurrency,
        "memory": {key: {"maximum_bytes": max(values), "measured_attempts": len(values)} for key, values in sorted(memory.items())},
        "memory_scope": "Process-lifetime RSS and PyTorch allocator peaks have distinct scopes; none is whole-host or whole-device memory.",
        "component_timing": {"measured_attempts": phases_measured, "unmeasured_attempts": len(attempts) - phases_measured,
                             "seconds": dict(phase_seconds), "scope": "Exclusive phases in instrumented adapters; missing phase timing is not zero."},
        "flops": {"kind": "unavailable", "value": None},
        "limits": ["Cost-to-outcome uses owned incremental paid costs. Evidence costs may overlap across requests and are not additive spending.",
                   "Avoided tasks are requested work left unattempted after a required scientific failure; cap-excluded work is not counted.",
                   "Queue-only paid attempts contribute spend, but require durable certified receipts for outcome/timing metrics."]}

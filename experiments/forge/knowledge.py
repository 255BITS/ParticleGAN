"""Deterministic experiment memory and goal views over immutable evidence.

These reducers never train, enqueue, or alter historical verdicts. Compatible
current receipts are independently regraded; historical imports remain visible
in separate cohorts until an explicit compatibility audit binds them.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import re

from .contracts import atomic_json, atomic_text, file_hash, file_lock, identifier, read_json, stable_hash

REDUCER_VERSION = "forge-knowledge-v1"


def _json_output(path: Path, value) -> None:
    """Avoid rewriting unaffected reducer materializations."""
    if path.exists() and read_json(path) == value:
        return
    atomic_json(path, value)


def _text_output(path: Path, value: str) -> None:
    if path.exists() and path.read_text() == value:
        return
    atomic_text(path, value)


def _records(root: Path) -> tuple[list[dict], list[dict]]:
    indexed, conflicts = {}, []
    for path in sorted((root / "reports/forge/records").glob("*.json")):
        record = read_json(path)
        if record.get("schema_version") != 1 or not record.get("record_id"):
            raise ValueError(f"invalid experiment record: {path}")
        if not isinstance(record.get("task_results", []), list):
            raise ValueError(f"task_results must be a list: {path}")
        identity = record["record_id"]
        if identity in indexed and stable_hash(indexed[identity]) != stable_hash(record):
            conflicts.append({"record_id": identity, "reason": "Different records have the same record_id.",
                              "path": str(path.relative_to(root))})
        else:
            indexed[identity] = record
    return [indexed[key] for key in sorted(indexed)], conflicts


def _gaps(root: Path) -> dict:
    path = root / "reports/forge/import-gaps.json"
    return read_json(path) if path.exists() else {
        "gaps": [{"kind": "history_not_imported", "reason": "Run forge history before treating memory as complete."}],
        "scientific_normalization_complete": False,
    }


def _cost(rows: list[dict]) -> dict:
    values = []
    unavailable = 0
    counted_attempts = set()
    for row in rows:
        attempt_id = row.get("_attempt_id")
        if attempt_id and attempt_id in counted_attempts:
            continue
        if attempt_id:
            counted_attempts.add(attempt_id)
        cost = row.get("cost", {})
        seconds = cost.get("wall_seconds", cost.get("seconds")) if isinstance(cost, dict) else None
        if type(seconds) in (int, float) and seconds >= 0:
            values.append(seconds)
        else:
            unavailable += 1
    return {"wall_seconds": sum(values) if values else None,
            "measured_tasks": len(values), "unmeasured_tasks": unavailable,
            "flops": {"kind": "unavailable", "value": None}}


def _attempts(root: Path) -> tuple[list[dict], list[dict]]:
    attempts, issues = [], []
    for directory in sorted((root / "reports/forge/attempts").glob("*")):
        if not directory.is_dir():
            continue
        request_path, result_path = directory / "request.json", directory / "result.json"
        if not request_path.exists() or not result_path.exists():
            issues.append({"attempt_id": directory.name, "reason": "Attempt lacks a durable request/result pair."})
            continue
        resolved, result = read_json(request_path), read_json(result_path)
        request = resolved.get("request", resolved)
        evidence_path = directory / "evidence.json"
        certificate = read_json(evidence_path) if evidence_path.exists() else {}
        reason = None
        if certificate.get("result_hash") != stable_hash(result):
            reason = "Missing or mismatched durable result hash."
        elif certificate.get("source") != request.get("source"):
            reason = "Source certificate does not match the immutable request."
        elif certificate.get("runtime") != request.get("runtime"):
            reason = "Runtime certificate does not match the immutable request."
        elif result.get("candidate_revision") != request.get("candidate_revision"):
            reason = "Result and request candidate revisions differ."
        elif result.get("attempt_id") != directory.name:
            reason = "Result attempt id does not match its durable directory."
        elif result.get("retry_of") != resolved.get("retry_of"):
            reason = "Retry declaration differs between the resolved request and result."
        jobs = request.get("jobs", [])
        keys = {task_id: job["compatibility_key"] for job in jobs
                for task_id in job.get("task_ids", [job["task_id"]])}
        rows = []
        for row in result.get("task_results", []):
            if row.get("compatibility_key") != keys.get(row.get("task_id")) or row.get("task_id") not in keys:
                reason = "Task result compatibility key does not match its immutable request."
            rows.append({**row, "_attempt_id": directory.name})
        if reason:
            issues.append({"attempt_id": directory.name, "reason": reason})
            rows = [{**row, "gate_status": "INVALID", "evidence": {}, "error": reason,
                     "reason": reason} for row in rows]
        attempts.append({"attempt_id": directory.name, "request": request, "task_results": rows,
                         "request_id": request.get("request_id", resolved.get("request_id")),
                         "valid_receipt": reason is None, "receipt_error": reason,
                         "result_hash": stable_hash(result),
                         "retry_of": result.get("retry_of"),
                         "attempt_status": result.get("raw", {}).get("attempt_status"),
                         "source": str(result_path.relative_to(root))})
    issues.extend(_resolve_retries(attempts))
    return attempts, issues


def _resolve_retries(attempts):
    """Only certified repairs supersede infrastructure outcomes; keep every cost."""
    indexed = {attempt["attempt_id"]: attempt for attempt in attempts}
    successors = defaultdict(list)
    errors = {}
    for attempt in attempts:
        link = attempt.get("retry_of")
        if not link:
            continue
        previous = indexed.get(link.get("attempt_id")) if isinstance(link, dict) else None
        reason = None
        if (not previous or not attempt["valid_receipt"] or not previous["valid_receipt"]
                or link.get("result_hash") != previous["result_hash"]
                or not isinstance(link.get("reason"), str) or not link["reason"].strip()
                or not link.get("authorized_at")):
            reason = "Retry lacks a valid prior receipt and explicit repair authorization."
        elif (previous.get("attempt_status") not in {"error", "timeout", "cancelled"}
                or not previous["task_results"]
                or any(row.get("gate_status") != "INCOMPLETE" for row in previous["task_results"])):
            reason = "Retry cannot supersede scientific, applicability, or invalid evidence."
        else:
            old, new = previous["request"], attempt["request"]
            keys = lambda a: {(row["task_id"], row["compatibility_key"]) for row in a["task_results"]}
            if (old["candidate_revision"] != new["candidate_revision"]
                    or old.get("source", {}).get("digest") != new.get("source", {}).get("digest")
                    or _runtime_cohort(old) != _runtime_cohort(new)
                    or old.get("protocol") != new.get("protocol") or keys(previous) != keys(attempt)):
                reason = "Retry changes the scientific identity of the prior attempt."
        if reason:
            errors[attempt["attempt_id"]] = reason
        elif previous:
            successors[previous["attempt_id"]].append(attempt["attempt_id"])
    for children in successors.values():
        if len(children) > 1:
            errors.update({child: "Conflicting retry branches require investigation." for child in children})
    # A valid queue cannot produce cycles; imported receipts must prove this too.
    for identity in indexed:
        chain, cursor = set(), identity
        while cursor in indexed and isinstance(indexed[cursor].get("retry_of"), dict):
            if cursor in chain:
                errors.update({name: "Cyclic retry lineage is invalid." for name in chain})
                break
            chain.add(cursor)
            cursor = indexed[cursor]["retry_of"].get("attempt_id")
    for identity, reason in errors.items():
        attempt = indexed[identity]
        attempt.update(valid_receipt=False, receipt_error=reason)
        attempt["task_results"] = [{**row, "gate_status": "INVALID", "evidence": {}, "reason": reason}
                                   for row in attempt["task_results"]]
    for parent, children in successors.items():
        child = children[0]
        if len(children) == 1 and parent not in errors and child not in errors:
            indexed[parent]["superseded_by"] = child
    return [{"attempt_id": identity, "reason": reason} for identity, reason in sorted(errors.items())]


def _retry_history(attempts):
    return [{"attempt_id": a["attempt_id"], "superseded_by": a["superseded_by"],
             "task_results": a["task_results"]} for a in attempts if a.get("superseded_by")]


def _historical_row(record: dict) -> dict:
    tasks = record.get("task_results", [])
    counts = dict(sorted(Counter(row.get("gate_status", "NOT_RUN") for row in tasks).items()))
    controls = [row for row in tasks if row.get("role") == "negative_control"]
    return {"candidate_id": record.get("candidate_id"), "candidate_revision": record.get("candidate_revision"),
            "record_id": record["record_id"], "evidence_scope": "historical",
            "cohort": stable_hash({"revision": record.get("candidate_revision"),
                                   "source": record.get("source"), "claim": record.get("claim_contract")}),
            "qualified_tier": None, "status": "HISTORICAL", "rank": None,
            "counts": counts, "recorded_tasks": len(tasks), "negative_controls": len(controls),
            "cost": _cost(tasks), "mechanism_class": record.get("mechanism_class", "unknown"),
            "prior": record.get("prior"), "claim_contract": record.get("claim_contract"),
            "task_results": [{key: row[key] for key in ("task_id", "gate_status", "raw_status", "metrics", "cost", "reason", "role")
                              if key in row} for row in tasks], "source": record.get("source"),
            "conclusion": record.get("conclusion"), "next_action": record.get("next_action"),
            "note": "Historical recorded outcomes; overlapping summaries/attempts are not independent trials or a combined cost total."}


def _current_request(root: Path, idea_id: str, view_id: str, execution_backend="cuda", cuda_model=None) -> dict:
    from .planning import resolve_idea
    return resolve_idea(root, idea_id, view_id=view_id, through_tier=3, freeze_source=False,
                        execution_backend=execution_backend, cuda_model=cuda_model)


def _runtime_cohort(request):
    profiles = request.get("compute_profiles") or {
        job["task_id"]: job.get("science", {}).get("compute") for job in request.get("jobs", [])
        if job.get("science", {}).get("compute")}
    return {"execution_backend": request.get("execution_backend", "unrecorded"),
            "runtime": request.get("runtime"), "compute_profiles": profiles}


def _queue_states(root, attempts):
    """Atomic queue snapshots are read without creating state or lock files."""
    from .__main__ import queue_location
    locations = {queue_location(root), root / "runs/forge"}
    locations.update(Path(a["request"]["queue_root"]) for a in attempts if a["request"].get("queue_root"))
    return {str(path.resolve()): read_json(path / "queue/state.json")
            for path in sorted(locations) if (path / "queue/state.json").exists()}


def _lifecycle(request, selected, records, queue_states):
    ids = {a["attempt_id"] for a in selected}
    candidate_id, revision = request["candidate"]["id"], request["candidate_revision"]
    relevant = []
    for location, state in queue_states.items():
        for request_id, entry in state.get("submissions", {}).items():
            original = entry["request"]
            if (original.get("candidate", {}).get("id") == candidate_id
                    and original.get("candidate_revision") == revision
                    and _runtime_cohort(original) == _runtime_cohort(request)):
                relevant.append({"queue_root": location, "request_id": request_id,
                                 "status": entry["status"], "lifecycle": entry.get("lifecycle")})
    active = [e for e in relevant if e["status"] in {"queued", "running", "paused"}]
    # Compare the exact attempted cohort covered by a readout. A subsequent retry
    # adds an id and invalidates coverage even though candidate revision is fixed.
    has_readout = bool(ids) and any(
        r.get("lifecycle") == "concluded" and r.get("evidence_scope") != "historical"
        and r.get("candidate_id") == candidate_id and r.get("candidate_revision") == revision
        and ids <= set(r.get("attempt_ids", []))
        and all(next((p.get("result_hash") for p in r.get("provenance", {}).get("attempts", [])
                      if p.get("attempt_id") == a["attempt_id"]), None) == a["result_hash"] for a in selected)
        for r in records)
    lifecycle = ("running" if any(e["status"] == "running" for e in active) else "ready") if active else (
        "concluded" if has_readout else "awaiting_readout" if selected else "proposed")
    return {"lifecycle": lifecycle, "pending_readout": bool(selected) and not has_readout and not active,
            "queue_submissions": relevant, "readout_covers_attempt_ids": sorted(ids) if has_readout else []}


def _pinned_row(group, records, queue_states):
    request = group[0]["request"]
    tasks = [row for attempt in group for row in attempt["task_results"]]
    active = [row for attempt in group if not attempt.get("superseded_by") for row in attempt["task_results"]]
    counts = dict(sorted(Counter(row.get("gate_status", "NOT_RUN") for row in active).items()))
    status = next((s for s in ("INVALID", "FAIL", "BLOCKED", "INCOMPLETE", "NOT_RUN", "PASS") if counts.get(s)), "NOT_RUN")
    diagnostic = bool(request.get("calibration_lane"))
    return {"candidate_id": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
            "evidence_scope": "calibration_diagnostic" if diagnostic else "pinned",
            "calibration_lane": request.get("calibration_lane"),
            "cohort": stable_hash({"revision": request["candidate_revision"],
                "runtime": _runtime_cohort(request), "protocol": request.get("protocol"),
                "tasks": {k: v for k, v in request.get("tasks", {}).items()}}),
            "runtime_cohort": _runtime_cohort(request), "qualified_tier": None,
            "status": "DIAGNOSTIC" if diagnostic else status, "counts": counts, "cost": _cost(tasks),
            "attempt_ids": sorted(a["attempt_id"] for a in group),
            "retry_history": _retry_history(group),
            "task_results": [{k: r[k] for k in ("task_id", "gate_status", "raw_status", "cost", "reason", "metrics") if k in r}
                             for r in tasks],
            "source": request.get("source"), "recorded_view": request.get("view"),
            "qualification_reuse": False, "grading": "Recorded verdicts under pinned source; never regraded with a changed live evaluator.",
            **_lifecycle(request, group, records, queue_states),
            "next_action": "Publish the stopped cohort readout, or declare a new compatible revision before further qualification."}


def board(root: Path, view_id: str) -> dict:
    """Read-only view; current hardware cohorts and pinned observations stay separate."""
    from . import views
    root = Path(root)
    view = views.load_view(root, view_id)
    records, conflicts = _records(root)
    attempts, receipt_issues = _attempts(root)
    states = _queue_states(root, attempts)
    conflicts.extend(receipt_issues)
    current, matched_attempts, seen_cohorts = [], set(), set()
    for path in sorted((root / "configs/forge/ideas").glob("*.json")):
        idea = read_json(path)
        idea_id = idea.get("id", path.stem)
        configurations = {("cuda", None), ("cpu", None)}
        for attempt in attempts:
            original = attempt["request"]
            if original.get("candidate", {}).get("id") != idea_id:
                continue
            backend = original.get("execution_backend")
            if backend in {"cpu", "cuda"}:
                model = original.get("compute_profiles", {}).get("cuda", {}).get("model") if backend == "cuda" else None
                configurations.add((backend, model))
        for backend, model in sorted(configurations, key=lambda item: (item[0], item[1] or "")):
            try:
                request = _current_request(root, idea_id, view_id, backend, model)
            except (ValueError, FileNotFoundError, ImportError) as exc:
                current.append({"candidate_id": idea_id, "candidate_revision": None, "qualified_tier": 0,
                                "runtime_cohort": {"execution_backend": backend, "cuda_model": model},
                                "status": "BLOCKED", "lifecycle": "proposed", "evidence_scope": "current",
                                "blockers": [{"reason": str(exc)}], "counts": {}, "cost": _cost([]),
                                "next_action": "Resolve the candidate declaration/capability blocker."})
                continue
            expected = {task_id: job["compatibility_key"] for job in request["jobs"]
                        for task_id in job.get("task_ids", [job["task_id"]])}
            cohort = stable_hash({"candidate_revision": request["candidate_revision"], "keys": expected})
            if cohort in seen_cohorts:
                continue
            seen_cohorts.add(cohort)
            selected, evidence_attempts, invalid_tasks = [], [], set()
            for attempt in attempts:
                original = attempt["request"]
                if original.get("calibration_lane"):
                    continue
                if (original.get("candidate_revision") != request["candidate_revision"]
                        or original.get("source", {}).get("digest") != request.get("source", {}).get("digest")
                        or _runtime_cohort(original) != _runtime_cohort(request)):
                    continue
                used = False
                for row in attempt["task_results"]:
                    if expected.get(row["task_id"]) != row.get("compatibility_key"):
                        continue
                    old_task = original.get("tasks", {}).get(row["task_id"])
                    if not old_task or views.task_fingerprint(old_task) != views.task_fingerprint(request["tasks"][row["task_id"]]):
                        continue
                    selected.append(row)
                    if not attempt["valid_receipt"]:
                        invalid_tasks.add(row["task_id"])
                    used = True
                if used:
                    evidence_attempts.append(attempt)
                    matched_attempts.add(attempt["attempt_id"])
            superseded = {a["attempt_id"] for a in evidence_attempts if a.get("superseded_by")}
            safe = [{**row, "applicability": {"status": "unknown", "reason": "Durable receipt validation failed."}}
                    if row["task_id"] in invalid_tasks else row for row in selected
                    if row["_attempt_id"] not in superseded]
            covered = {row["task_id"] for row in safe}
            for task_id, task in request["tasks"].items():
                if task_id not in covered and task.get("preflight_blockers"):
                    safe.append({"task_id": task_id, "gate_status": "BLOCKED",
                                 "reason": "; ".join(task["preflight_blockers"])})
            qualified = views.qualify(view, request["tasks"], safe, candidate=request["candidate"])
            if request.get("preflight_blockers"):
                qualified.update(status="BLOCKED", eligible=False, qualified_tier=0)
                qualified["blockers"] = [{"task_id": None, "status": "BLOCKED", "reasons": request["preflight_blockers"]},
                                         *qualified["blockers"]]
            lifecycle = _lifecycle(request, evidence_attempts, records, states)
            current.append({"candidate_id": idea_id, "candidate_revision": request["candidate_revision"],
                            "evidence_scope": "current", "cohort": cohort,
                            "runtime_cohort": _runtime_cohort(request),
                            "source_digest": request.get("source", {}).get("digest"),
                            "status": qualified["status"], "qualified_tier": qualified["qualified_tier"],
                            "qualification": qualified,
                            "counts": dict(sorted(Counter(qualified["task_statuses"].values()).items())),
                            "cost": _cost(selected), "attempt_ids": sorted(a["attempt_id"] for a in evidence_attempts),
                            "retry_history": _retry_history(evidence_attempts),
                            **lifecycle,
                            "mechanism_class": idea.get("mechanism_class"), "prior": request["candidate"].get("prior"),
                            "claim_contract": request["candidate"].get("claim_contract"),
                            "preflight_blockers": request.get("preflight_blockers", []),
                            "next_action": "Publish an explanation, comparison, and next action." if lifecycle["pending_readout"]
                                           else "Run the next eligible task within an explicit budget."})
    current.sort(key=lambda row: (-row["qualified_tier"], row["candidate_id"], row.get("cohort", "")))
    historical = [_historical_row(record) for record in records
                  if record.get("evidence_scope") == "historical" and record.get("record_type") != "family_context"]
    historical.sort(key=lambda row: (str(row["candidate_id"]), row["record_id"]))
    archived_groups = defaultdict(list)
    for attempt in attempts:
        if attempt["attempt_id"] not in matched_attempts:
            request = attempt["request"]
            identity = {k: request.get(k) for k in ("candidate_revision", "source", "protocol", "tasks")}
            identity["runtime"] = _runtime_cohort(request)
            identity["evidence_use"] = "calibration_diagnostic" if request.get("calibration_lane") else "qualification"
            archived_groups[stable_hash(identity)].append(attempt)
    archived_rows = [_pinned_row(group, records, states) for _, group in sorted(archived_groups.items())]
    pinned = [row for row in archived_rows if row["evidence_scope"] == "pinned"]
    diagnostic = [row for row in archived_rows if row["evidence_scope"] == "calibration_diagnostic"]
    archived = [{"attempt_id": a["attempt_id"], "candidate_id": a["request"].get("candidate", {}).get("id"),
                 "candidate_revision": a["request"].get("candidate_revision"), "source": a["source"],
                 "cost": _cost(a["task_results"]), "reason": "Registered calibration diagnostics cannot qualify a candidate."
                 if a["request"].get("calibration_lane") else "Pinned source/task/protocol/runtime differs from the current cohort."}
                for a in attempts if a["attempt_id"] not in matched_attempts]
    return {"schema_version": 1, "view": view_id, "view_revision": view["revision"],
            "policy_fingerprint": views.view_fingerprint(view), "calibration": view.get("calibration"),
            "rows": current + pinned + diagnostic + historical, "current_rows": current, "historical_rows": historical,
            "calibration_rows": diagnostic,
            "pinned_rows": pinned, "archived_attempts": archived, "conflicts": conflicts, "import_gaps": _gaps(root),
            "ranking_note": "Tier attainment precedes metrics. CPU/CUDA and GPU models remain separate; pinned and historical verdicts are unranked and never automatically reused."}


def recall(root: Path, query: str = "", goal: str | None = None) -> list[dict]:
    """Search normalized records; missing goal tags remain visible as unknown."""
    records, _ = _records(Path(root))
    words = re.findall(r"[a-z0-9_]+", query.lower())
    matches = []
    for record in records:
        searchable = " ".join(str(record.get(k, "")) for k in (
            "candidate_id", "hypothesis", "mechanism_class", "conclusion", "next_action", "source", "provenance")).lower()
        score = sum(word in searchable for word in words)
        if words and score == 0:
            continue
        declared_goal = record.get("goal")
        if goal and declared_goal and declared_goal != goal:
            continue
        matches.append({"record_id": record["record_id"], "candidate_id": record.get("candidate_id"),
                        "candidate_revision": record.get("candidate_revision"), "evidence_scope": record.get("evidence_scope"),
                        "hypothesis": record.get("hypothesis"), "conclusion": record.get("conclusion"),
                        "next_action": record.get("next_action"), "source": record.get("source"),
                        "cost": _cost(record.get("task_results", [])), "match_score": score,
                        "goal_applicability": "unknown" if not declared_goal else declared_goal})
    return sorted(matches, key=lambda row: (-row["match_score"], row["record_id"]))


def _link(source) -> str:
    if isinstance(source, dict):
        return source.get("url") or "../../" + source.get("path", "")
    return "../../" + str(source or "")


def _cell(value) -> str:
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def _board_markdown(result: dict) -> str:
    lines = [f"# {result['view']} — revision {result['view_revision']}", "", result["ranking_note"], "",
             "| Candidate / exact revision | Scope | Tier | Outcomes | Wall seconds | Next action |",
             "| --- | --- | ---: | --- | ---: | --- |"]
    for row in result["rows"]:
        identity = str(row["candidate_id"]) + " / " + str(row.get("candidate_revision") or "unknown")[:12]
        runtime = row.get("runtime_cohort", {})
        backend = runtime.get("execution_backend")
        if backend:
            identity += " / " + backend
            models = sorted({p.get("model") for p in runtime.get("compute_profiles", {}).values() if p and p.get("model")})
            if models:
                identity += " (" + ", ".join(models) + ")"
        counts = ", ".join(f"{key} {value}" for key, value in row.get("counts", {}).items()) or "unmeasured"
        seconds = row["cost"].get("wall_seconds")
        lines.append("| " + " | ".join(_cell(x) for x in (identity, row["evidence_scope"], row["qualified_tier"],
                     counts, round(seconds, 3) if seconds is not None else None, row.get("next_action"))) + " |")
    lines += ["", "Historical summaries overlap individual attempt rows. Do not sum them as independent trials or total cost.", ""]
    return "\n".join(lines)


def _metric_excerpt(rows: list[dict]) -> str:
    fragments = []
    keys = ("modes", "hq", "precision", "acc_center_rms_sigma", "center_rms_sigma", "mass_tv",
            "acc_radial_ks", "radial_ks", "recon_mse", "error", "passing_checks", "observations")
    for row in rows[:3]:
        metrics = row.get("metrics", {})
        source = metrics.get("final", metrics.get("metrics", metrics)) if isinstance(metrics, dict) else {}
        if not isinstance(source, dict):
            continue
        selected = {key: source[key] for key in keys if type(source.get(key)) in (int, float)}
        if selected:
            fragments.append(str(row.get("task_id")) + ": " + ", ".join(f"{key}={value:.5g}" for key, value in selected.items()))
    return "; ".join(fragments)


def compile_memory(root: Path) -> dict:
    """Map outputs reduce deterministically, with one writer and explicit gaps."""
    root = Path(root)
    output = root / "reports/forge"
    with file_lock(root / "runs/forge/compile.lock"):
        records, conflicts = _records(root)
        boards = []
        for path in sorted((root / "configs/forge/views").glob("*.json")):
            result = board(root, path.stem)
            boards.append(result)
            _json_output(output / "leaderboards" / (path.stem + ".json"), result)
            _text_output(output / "leaderboards" / (path.stem + ".md"), _board_markdown(result))
        gaps = _gaps(root)
        inputs = {}
        for directory in (root / "reports/forge/records", root / "reports/forge/attempts", root / "configs/forge",
                          root / "reports/forge/calibration-lanes", root / "reports/forge/promotions"):
            for path in sorted(directory.rglob("*.json")):
                inputs[str(path.relative_to(root))] = file_hash(path)
        for name in ("import-gaps.json", "history-sources.json"):
            path = output / name
            if path.exists():
                inputs[str(path.relative_to(root))] = file_hash(path)
        coverage = gaps.get("inventory_coverage", {"valid": False, "reason": "inventory unavailable"})
        try:
            from .history import validate_inventory
            coverage = validate_inventory(root)
        except Exception as exc:
            # Repo-less fixtures and exported artifacts can still be read, never called complete.
            coverage = {"valid": False, "reason": str(exc)}
        pending = sorted({row["candidate_id"] for result in boards for row in result["current_rows"] + result.get("pinned_rows", []) + result.get("calibration_rows", [])
                          if row.get("pending_readout")})
        scientific_sources = sorted({row["source_digest"] for result in boards for row in result["current_rows"]
                                     if row.get("source_digest")})
        reducer_hash = file_hash(Path(__file__))
        manifest = {"schema_version": 1, "reducer_version": REDUCER_VERSION, "reducer_sha256": reducer_hash,
                    "input_hashes": inputs, "scientific_source_digests": scientific_sources,
                    "input_digest": stable_hash({"files": inputs, "sources": scientific_sources, "reducer": reducer_hash}),
                    "record_count": len(records), "view_count": len(boards), "inventory_coverage": coverage,
                    "conflicts": conflicts + [issue for result in boards for issue in result["conflicts"]],
                    "pending_readout": pending, "import_gap_count": len(gaps.get("gaps", [])),
                    "scientific_normalization_complete": gaps.get("scientific_normalization_complete", False)}
        manifest["operational_lifecycle_digest"] = stable_hash([
            {"cohort": row.get("cohort"), "lifecycle": row.get("lifecycle"), "requests": row.get("queue_submissions", [])}
            for result in boards for row in result["current_rows"] + result.get("pinned_rows", []) + result.get("calibration_rows", [])])
        lines = ["# ParticleGAN Forge experiment memory", "",
                 "Read the relevant prior evidence before declaring an idea. Historical outcomes retain their original scope; current qualification is recomputed from compatible receipts.", "",
                 f"Records: {len(records)}. Inventory coverage: {'complete' if coverage.get('valid') else 'incomplete'}. "
                 f"Unresolved import items: {len(gaps.get('gaps', []))}.", "",
                 "## Goal views", ""]
        lines.extend(f"- [{result['view']}](leaderboards/{result['view']}.md)" for result in boards)
        lines += ["", "## Pending readouts", "", ", ".join(pending) or "None recorded.", "",
                  "## Experiment and family records", ""]
        for record in records:
            rows = record.get("task_results", [])
            counts = dict(sorted(Counter(row.get("gate_status", "NOT_RUN") for row in rows).items()))
            seconds = _cost(rows)["wall_seconds"]
            lines += [f"### {record.get('candidate_id', 'unknown')} · {record['record_id']}", "",
                      f"**Scope:** {record.get('evidence_scope', 'current')}; {record.get('record_type', 'scientific')}; "
                      f"revision `{record.get('candidate_revision', 'unknown')}`.", "",
                      str(record.get("hypothesis", "Hypothesis unrecorded.")), "",
                      f"**Observed:** {_cell(counts) if counts else 'No normalized scientific verdict'}; "
                      f"wall seconds {_cell(round(seconds, 3) if seconds is not None else None)}; mechanism `{record.get('mechanism_class', 'unknown')}`.", "",
                      _metric_excerpt(rows), "",
                      str(record.get("conclusion", "Conclusion pending.")), "",
                      f"**Next:** {record.get('next_action', 'Readout pending.')}", "",
                      f"[Evidence]({_link(record.get('source'))}) · [Record](records/{record['record_id']}.json)", ""]
        lines += ["## Unresolved imports and limitations", ""]
        for gap in gaps.get("gaps", []):
            lines.append(f"- **{gap.get('gap_id', gap.get('kind', 'unknown'))}:** {gap.get('reason', 'Unresolved')}"
                         + (f" ({len(gap['paths'])} classified sources.)" if "paths" in gap else ""))
        lines += ["", "## Compilation provenance", "",
                  f"Reducer `{REDUCER_VERSION}`; input digest `{manifest['input_digest']}`. "
                  "[Full input hashes and coverage](compilation.json). No training or image inspection occurs during compilation.", ""]
        _text_output(output / "EXPERIMENT_MEMORY.md", "\n".join(lines))
        _json_output(output / "compilation.json", manifest)
        return {"records": len(records), "views": len(boards), "pending_readout": pending,
                "inventory_coverage": coverage, "input_digest": manifest["input_digest"],
                "memory": "reports/forge/EXPERIMENT_MEMORY.md", "conflicts": manifest["conflicts"]}


def readout(root: Path, candidate_id: str, conclusion: str, comparison: str, next_action: str) -> dict:
    """Publish a readout for one exact attempted revision, then update memory."""
    root = Path(root)
    identifier(candidate_id, "candidate id")
    if not all(isinstance(value, str) and value.strip() for value in (conclusion, comparison, next_action)):
        raise ValueError("readout requires conclusion, comparison, and next_action")
    requested_revision = None
    if "@" in candidate_id and not (root / "configs/forge/ideas" / (candidate_id + ".json")).exists():
        candidate_id, requested_revision = candidate_id.rsplit("@", 1)
    attempts, _ = _attempts(root)
    selected = [a for a in attempts if a["request"].get("candidate", {}).get("id") == candidate_id
                and (requested_revision is None or a["request"]["candidate_revision"].startswith(requested_revision))]
    revisions = {a["request"]["candidate_revision"] for a in selected}
    if not revisions:
        raise ValueError("No durable execution attempt exists for this candidate/revision.")
    if len(revisions) != 1:
        raise ValueError("Multiple revisions have attempts; use candidate-id@revision to bind the readout.")
    revision = revisions.pop()
    candidate = selected[0]["request"]["candidate"]
    diagnostic_only = all(a["request"].get("calibration_lane") for a in selected)
    record = {"schema_version": 1, "record_id": "readout-" + stable_hash({"candidate": candidate_id, "revision": revision})[:24],
              "record_type": "scientific", "candidate_id": candidate_id, "candidate_revision": revision,
              "evidence_scope": "calibration_diagnostic" if diagnostic_only else "current",
              "qualification_reuse": not diagnostic_only, "lifecycle": "concluded", "goal": candidate.get("goal"),
              "hypothesis": candidate.get("hypothesis", "See immutable candidate declaration."),
              "mechanism_class": candidate.get("mechanism_class", "unknown"), "prior": candidate.get("prior"),
              "claim_contract": candidate.get("claim_contract"),
              "source": {"path": selected[0]["source"]},
              "task_results": [row for attempt in selected for row in attempt["task_results"]],
              "attempt_ids": sorted(a["attempt_id"] for a in selected),
              "provenance": {"attempts": [{"attempt_id": a["attempt_id"], "result_hash": a["result_hash"],
                                             "valid_receipt": a["valid_receipt"]} for a in selected]},
              "conclusion": conclusion.strip(), "comparison": comparison.strip(), "next_action": next_action.strip()}
    states = _queue_states(root, attempts)
    with file_lock(root / "runs/forge/readout.lock"), ExitStack() as locks:
        for location in sorted(states):
            locks.enter_context(file_lock(Path(location) / "queue.lock"))
        stopped = []
        for location in sorted(states):
            state = read_json(Path(location) / "queue/state.json")
            matching = [e for e in state.get("submissions", {}).values()
                        if e["request"].get("candidate", {}).get("id") == candidate_id
                        and e["request"].get("candidate_revision") == revision]
            if any(e["status"] in {"queued", "running", "paused"} for e in matching):
                raise ValueError("Candidate has active queued/running/paused work; stop or complete it before readout.")
            stopped.append((location, state, matching))
        atomic_json(root / "reports/forge/records" / (record["record_id"] + ".json"), record)
        for location, state, matching in stopped:
            if not matching:
                continue
            for entry in matching:
                entry.update(status="concluded", lifecycle="concluded", readout_record_id=record["record_id"],
                             readout_attempt_ids=record["attempt_ids"])
            atomic_json(Path(location) / "queue/state.json", state)
    compile_memory(root)
    return record

"""Immutable administrative dispositions, independent of scientific verdicts.

Every attempted revision needs a concluded readout before it can be abandoned
or superseded. A disposition preserves that readout and the exact result hashes;
it neither replaces evidence nor confers qualification.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

from .contracts import atomic_json, file_lock, identifier, read_json, stable_hash, transition

TERMINAL = frozenset({"abandoned", "superseded"})
ACTIVE = frozenset({"queued", "running", "paused"})


def _identity(reference: str, root: Path):
    identifier(reference, "candidate reference")
    if "@" in reference and not (root / "configs/forge/ideas" / (reference + ".json")).exists():
        name, revision = reference.rsplit("@", 1)
        identifier(name, "candidate id")
        identifier(revision, "revision prefix")
        return name, revision
    return reference, None


def _matches(request, candidate_id, revision):
    return (request.get("candidate", {}).get("id") == candidate_id
            and request.get("candidate_revision") == revision)


def _disposition(records, candidate_id, revision):
    found = []
    for record in records:
        if (record.get("record_type") != "administrative"
                or record.get("candidate_id") != candidate_id
                or record.get("candidate_revision") != revision):
            continue
        body = {key: value for key, value in record.items() if key != "record_id"}
        if (record.get("schema_version") != 1 or record.get("lifecycle") not in TERMINAL
                or record.get("record_id") != "lifecycle-" + stable_hash(body)[:24]):
            raise ValueError("Invalid immutable lifecycle receipt")
        found.append(record)
    if len(found) > 1:
        raise ValueError("Conflicting lifecycle dispositions for the same candidate revision")
    return found[0] if found else None


def concluded_readout(request, attempts, records):
    """Find a concluded scientific readout covering every exact selected result."""
    if not attempts:
        return None
    ids = {attempt["attempt_id"] for attempt in attempts}
    for record in records:
        if (record.get("lifecycle") != "concluded"
                or record.get("evidence_scope") in {"historical", "administrative"}
                or record.get("candidate_id") != request["candidate"]["id"]
                or record.get("candidate_revision") != request["candidate_revision"]
                or not ids <= set(record.get("attempt_ids", []))):
            continue
        recorded = {row.get("attempt_id"): row.get("result_hash")
                    for row in record.get("provenance", {}).get("attempts", [])}
        if (all(recorded.get(a["attempt_id"]) == a["result_hash"] for a in attempts)
                and all(isinstance(record.get(key), str) and record[key].strip()
                        for key in ("conclusion", "comparison", "next_action"))):
            return record
    return None


def ensure_open(root: Path, request: dict) -> None:
    """Queue integration: call under its queue lock before admitting more work."""
    from .knowledge import _records
    records, _ = _records(Path(root))
    existing = _disposition(records, request["candidate"]["id"], request["candidate_revision"])
    if existing:
        raise ValueError(f"Candidate revision is {existing['lifecycle']}; declare a new revision before execution")


def overlay(request, selected, records, queue_states, base):
    """Add administrative metadata to a lifecycle result; retain all science."""
    record = _disposition(records, request["candidate"]["id"], request["candidate_revision"])
    if record is None:
        return base
    active = [entry for state in queue_states.values() for entry in state.get("submissions", {}).values()
              if _matches(entry["request"], record["candidate_id"], record["candidate_revision"])
              and entry["status"] in ACTIVE]
    bound = {a["attempt_id"]: a["result_hash"] for a in record["provenance"]["attempts"]}
    changed = any(bound.get(a["attempt_id"]) != a["result_hash"] for a in selected)
    metadata = {"disposition_record_id": record["record_id"], "disposition_reason": record["reason"],
                "successor": record.get("successor"), "readout_record_id": record.get("readout_record_id")}
    if active or changed:
        return {**base, **metadata, "lifecycle_conflict":
                "Work appeared after an immutable disposition; preserve it and publish any missing readout."}
    return {**base, **metadata, "lifecycle": record["lifecycle"]}


def _select(root, reference, attempts, records, states, *, successor=False):
    name, prefix = _identity(reference, root)
    known, attempted = {}, set()
    for attempt in attempts:
        request = attempt["request"]
        if request.get("candidate", {}).get("id") == name:
            known[request["candidate_revision"]] = request
            attempted.add(request["candidate_revision"])
    for state in states.values():
        for entry in state.get("submissions", {}).values():
            request = entry["request"]
            if request.get("candidate", {}).get("id") == name:
                known.setdefault(request["candidate_revision"], request)
    for record in records:
        if record.get("record_type") == "administrative" and record.get("candidate_id") == name:
            known.setdefault(record["candidate_revision"], record["candidate_snapshot"])
    if prefix:
        matches = {key: value for key, value in known.items() if key.startswith(prefix)}
        if len(matches) == 1:
            return next(iter(matches.values()))
        if len(matches) > 1:
            raise ValueError("Ambiguous revision prefix; use the full candidate revision")
    elif attempted and not successor:
        if len(attempted) != 1:
            raise ValueError("Multiple attempted revisions; use candidate-id@revision")
        return known[next(iter(attempted))]
    idea = root / "configs/forge/ideas" / (name + ".json")
    if idea.exists():
        from .planning import resolve_idea
        current = resolve_idea(root, name, through_tier=1, freeze_source=False, execution_backend="cpu")
        if prefix is None or current["candidate_revision"].startswith(prefix):
            return current
    if prefix is None and len(known) == 1:
        return next(iter(known.values()))
    raise ValueError("Unknown or ambiguous candidate revision; use an existing declaration or exact saved revision")


def dispose(root: Path, candidate_id: str, destination: str, reason: str, *,
            successor: str | None = None, queue_root: Path | None = None) -> dict:
    """Record one irreversible administrative decision, without changing evidence.

    Repeating the identical decision is idempotent. Stop queued, running or
    paused work explicitly first; this function never cancels a worker for you.
    """
    from .knowledge import _attempts, _cost, _lifecycle, _queue_states, _records
    root = Path(root).resolve()
    if destination not in TERMINAL:
        raise ValueError("Disposition must be abandoned or superseded")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Disposition requires a nonempty reason")
    reason = reason.strip()
    if destination == "superseded" and not successor:
        raise ValueError("Superseded requires an existing successor candidate/revision")
    if destination == "abandoned" and successor is not None:
        raise ValueError("Use supersede when naming a successor")
    with file_lock(root / "runs/forge/readout.lock"), ExitStack() as locks:
        attempts, _ = _attempts(root)
        states = _queue_states(root, attempts)
        from .__main__ import queue_location
        for location in (queue_location(root), root / "runs/forge"):
            states.setdefault(str(location.resolve()), {})
        if queue_root is not None:
            location = Path(queue_root).resolve()
            states.setdefault(str(location), {})
        for location in sorted(states):
            locks.enter_context(file_lock(Path(location) / "queue.lock"))
        states = {location: read_json(Path(location) / "queue/state.json")
                  for location in states if (Path(location) / "queue/state.json").exists()}
        attempts, _ = _attempts(root)
        records, conflicts = _records(root)
        if conflicts:
            raise ValueError("Resolve conflicting experiment records before changing lifecycle")
        request = _select(root, candidate_id, attempts, records, states)
        name, revision = request["candidate"]["id"], request["candidate_revision"]
        complete_ids = {a["attempt_id"] for a in attempts}
        for path in (root / "reports/forge/attempts").glob("*/request.json"):
            if path.parent.name in complete_ids:
                continue
            unresolved = read_json(path)
            if _matches(unresolved.get("request", unresolved), name, revision):
                raise ValueError("Repair the incomplete durable attempt before readout and disposition")
        selected = [a for a in attempts if _matches(a["request"], name, revision)]
        relevant = [{"queue_root": location, "request_id": request_id, "status": entry["status"]}
                    for location, state in states.items() for request_id, entry in state.get("submissions", {}).items()
                    if _matches(entry["request"], name, revision)]
        if any(entry["status"] in ACTIVE for entry in relevant):
            raise ValueError("Candidate has active queued/running/paused work; stop or complete it before disposition")
        replacement = _select(root, successor, attempts, records, states, successor=True) if successor else None
        successor_ref = None
        if replacement:
            successor_ref = replacement["candidate"]["id"] + "@" + replacement["candidate_revision"]
            if _matches(replacement, name, revision):
                raise ValueError("A successor must be a different candidate revision")
            if _disposition(records, replacement["candidate"]["id"], replacement["candidate_revision"]):
                raise ValueError("Successor revision is already abandoned or superseded")
        attempt_provenance = [{"attempt_id": a["attempt_id"], "result_hash": a["result_hash"],
                               "valid_receipt": a["valid_receipt"]} for a in selected]
        existing = _disposition(records, name, revision)
        if existing:
            if (existing["lifecycle"] == destination and existing["reason"] == reason
                    and existing.get("successor") == successor_ref
                    and existing["provenance"]["attempts"] == attempt_provenance):
                return existing
            raise ValueError("Candidate revision already has an immutable disposition; it cannot be rewritten")
        readout = concluded_readout(request, selected, records)
        if selected and readout is None:
            raise ValueError("Attempted candidate needs a concluded readout covering all current attempt/result hashes")
        base = _lifecycle(request, selected, records, states)
        body = transition({"lifecycle": "concluded" if readout else base["lifecycle"]}, destination,
                          reason=reason, successor=successor_ref)
        body.update(schema_version=1, record_type="administrative", evidence_scope="administrative",
                    qualification_reuse=False, candidate_id=name, candidate_revision=revision,
                    candidate_snapshot=deepcopy(request), source=deepcopy(request.get("source")),
                    goal=request["candidate"].get("goal"), hypothesis=request["candidate"].get("hypothesis"),
                    mechanism_class=request["candidate"].get("mechanism_class"),
                    prior=request["candidate"].get("prior"), claim_contract=request["candidate"].get("claim_contract"),
                    reason=reason, successor=successor_ref,
                    successor_identity=None if replacement is None else {
                        "candidate_id": replacement["candidate"]["id"],
                        "candidate_revision": replacement["candidate_revision"], "source": replacement.get("source")},
                    task_results=[], attempt_ids=sorted(a["attempt_id"] for a in selected),
                    cost_at_disposition=_cost([row for a in selected for row in a["task_results"]]),
                    readout_record_id=None if readout is None else readout["record_id"],
                    provenance={"attempts": attempt_provenance, "queue_submissions": relevant,
                                "concluded_readout": None if readout is None else {
                                    "record_id": readout["record_id"], "sha256": stable_hash(readout),
                                    "snapshot": deepcopy(readout)}},
                    conclusion=f"Administrative {destination}: {reason}",
                    next_action=f"See successor {successor_ref}." if successor_ref else
                                "Retain this revision's evidence and readout; declare a new revision before further work.")
        record = {**body, "record_id": "lifecycle-" + stable_hash(body)[:24]}
        path = root / "reports/forge/records" / (record["record_id"] + ".json")
        if path.exists() and read_json(path) != record:
            raise ValueError("Lifecycle receipt identity collision")
        atomic_json(path, record)
    return record


def abandon(root: Path, candidate_id: str, reason: str, *, queue_root: Path | None = None) -> dict:
    return dispose(root, candidate_id, "abandoned", reason, queue_root=queue_root)


def supersede(root: Path, candidate_id: str, successor: str, reason: str, *,
              queue_root: Path | None = None) -> dict:
    return dispose(root, candidate_id, "superseded", reason, successor=successor, queue_root=queue_root)

"""Publish a registered performance campaign from frozen queue evidence.

No planner, trainer, evaluator or artifact hydration is called. Bulk artifacts
stay at their original locations; only source-bound final summaries are copied.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from .contracts import atomic_json, file_hash, identifier, read_json, stable_hash
from .queue import Queue
from .scaling import SCHEMA, manifest_path
from .scaling_publication import (BENCHMARK, MANIFEST, build_manifest, performance_status,
                                  refresh_performance, validate_receipt)

EXPORT_SCHEMA = "forge_scaling_export_v1"
UNAVAILABLE_SCHEMA = "forge_scaling_unavailable_v1"
_ADAPTER_FIELDS = {"applied", "evidence", "cost", "telemetry", "execution_path", "api_version",
                   "api_changes", "claim_contract", "qualification_input", "qualification_reuse"}


def _fail(message):
    raise ValueError("scaling export: " + message)


def _read(path):
    data = Path(path).read_bytes()
    document = json.loads(data)
    import hashlib
    return document, {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def _normalized_spec(task):
    spec = deepcopy(task["execution"]["performance"])
    spec["max_seconds"] = float(spec["max_seconds"])
    return spec


def _declared_task(task):
    return {key: value for key, value in task.items() if key not in {"field_ownership", "preflight_blockers"}}


def _verify_request(root, request, selected, registration):
    candidate = selected["candidate_id"]
    if (request["candidate"]["id"] != candidate
            or request["study"]["id"] != selected["study"]
            or request["view"]["id"] != registration["view"]
            or request["protocol"]["seed"] != 0):
        _fail("frozen request differs from registered candidate, study, view or seed")
    source = request["source"]
    if stable_hash(source["files"]) != source["digest"]:
        _fail("frozen source map digest mismatch")
    candidate_path = f"configs/forge/ideas/{candidate}.json"
    declaration_path = Path(root) / candidate_path
    authored = None
    if declaration_path.is_file() and file_hash(declaration_path) == selected["candidate_declaration_sha256"]:
        authored = read_json(declaration_path)
    elif source.get("origin_commit"):
        # Forge's code source map deliberately excludes recipe declarations.
        # Recover their exact registered bytes from Git when the live card moved.
        import hashlib
        import subprocess
        original = subprocess.run(["git", "show", source["origin_commit"] + ":" + candidate_path],
                                  cwd=root, capture_output=True, check=False)
        if original.returncode == 0 and hashlib.sha256(original.stdout).hexdigest() == selected["candidate_declaration_sha256"]:
            authored = json.loads(original.stdout)
    if authored is None or any(request["candidate"].get(key) != value for key, value in authored.items()):
        _fail("frozen candidate differs from the exact registered declaration")
    for task_id in registration["task_ids"]:
        task = request["tasks"][task_id]
        if stable_hash(_declared_task(task)) != registration["task_contract_sha256"][task_id]:
            _fail("frozen task differs from registered contract: " + task_id)


def _display_family(publication, selected):
    matches = {row["trainer_family"] for row in publication["rows"]
               if row["candidate_id"] == selected["candidate_id"]}
    if len(matches) > 1:
        _fail("candidate has ambiguous selected inventory ownership")
    # A newly selected or historical candidate still keeps its registered owner;
    # the current-board renderer will leave a different selected recipe unknown.
    return next(iter(matches)) if matches else selected["family"]


def _artifact_receipts(summary, directory):
    artifacts = {}
    for kind, name in summary.get("artifacts", {}).items():
        if not name:
            continue
        if not isinstance(name, str):
            _fail("core artifact reference is not a relative filename")
        path = directory / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(directory.resolve()):
            _fail("core artifact path escapes its original run")
        declared = summary.get("artifact_sha256", {}).get(kind)
        available = path.is_file()
        if available and declared and file_hash(path) != declared:
            _fail("original bulk artifact bytes changed: " + str(path))
        artifacts[kind] = {"path": str(path.resolve()), "sha256": declared,
                           "available_locally": available,
                           "verification": "exact bytes" if available and declared else "original declared digest retained"}
    return artifacts


def _core_summary(raw, request, task_id, actual_request, canonical_result):
    if raw.get("benchmark") != BENCHMARK:
        return None
    summary = {key: deepcopy(value) for key, value in raw.items() if key not in _ADAPTER_FIELDS}
    validate_receipt(summary)
    binding = summary.get("forge_binding", {})
    expected = {"candidate_revision": actual_request["candidate_revision"],
                "candidate_sha256": stable_hash(actual_request["candidate"]),
                "request_id": actual_request["request_id"],
                "source_digest": actual_request["source"]["digest"],
                "task_sha256": stable_hash(actual_request["tasks"][task_id])}
    if (binding != expected or summary.get("candidate_id") != request["candidate"]["id"]
            or summary.get("task_id") != task_id
            or summary["spec"] != _normalized_spec(request["tasks"][task_id])):
        _fail("core summary differs from its frozen Forge execution bindings")
    if (actual_request["candidate_revision"] != request["candidate_revision"]
            or actual_request["source"]["digest"] != request["source"]["digest"]):
        _fail("reused attempt has an incompatible candidate/source revision")
    for name, digest in summary["identity"]["source"]["files_sha256"].items():
        if actual_request["source"]["files"].get(name) != digest:
            _fail("core source file differs from the frozen queue source: " + name)
    if raw.get("applied", {}).get("recipe") != summary["recipe"]:
        _fail("core recipe differs from the applied public-API receipt")
    row = next((item for item in canonical_result["task_results"] if item["task_id"] == task_id), None)
    if row is None:
        _fail("canonical queue result has no registered task outcome")
    summary["independent_evaluation"] = {"gate_status": row.get("gate_status"),
                                         "raw_status": row.get("raw_status"),
                                         "reason": row.get("reason", "; ".join(row.get("reasons", [])))}
    return summary


def _unavailable_status(submission, job, result=None):
    raw = (result or {}).get("raw", {})
    detail = raw.get("result", {})
    if detail.get("applicability", {}).get("status") == "unsupported":
        return "UNSUPPORTED", detail["applicability"].get("reason", "unsupported capability")
    error = detail.get("error", {})
    if error.get("type") in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or "out of memory" in error.get("message", "").lower():
        return "OOM", error.get("message", "out of memory before a core summary was recorded")
    if submission.get("status") == "blocked":
        return "BLOCKED", submission.get("reason", "submission blocked")
    if submission.get("status") == "cancelled":
        return "INCOMPLETE", submission.get("reason", "submission cancelled")
    if job is not None and job.get("status") in {"pending", "running"}:
        return "UNKNOWN", "queued" if job["status"] == "pending" else "running; no final timing receipt"
    if raw.get("attempt_status") in {"timeout", "cancelled"}:
        return "INCOMPLETE", raw.get("reason", raw["attempt_status"])
    if raw.get("attempt_status") == "error":
        return "ERROR", raw.get("reason", detail.get("error", {}).get("message", "runner error"))
    return "UNKNOWN", "no completed performance receipt"


def _write_unchanged(path, document):
    if not path.is_file() or read_json(path) != document:
        atomic_json(path, document)


def publish(root, queue_root, registration, *, allow_partial=False, refresh=True):
    """Export one current registration; refuse pending/running work by default."""
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    registration_file = manifest_path(root, registration)
    card, card_proof = _read(registration_file)
    if (card.get("schema") != SCHEMA or card.get("protocol_seed") != 0
            or card.get("qualification_input") is not False or card.get("qualification_reuse") is not False):
        _fail("registration is not a seed-0 performance-only campaign")
    registration_id = identifier(card["id"], "registration id")
    inventory = read_json(root / "reports/forge/technique-inventory.json")
    state = Queue(queue_root).inspect()
    documents, entries, unavailable, task_specs, roster = {}, [], [], {}, []
    for selected in card["family_selections"]:
        family = _display_family(inventory, selected)
        candidate = selected["candidate_id"]
        roster.append({"trainer_family": family, "candidate_id": candidate,
                       "registration_family": selected["family"]})
        matches = [(key, item) for key, item in state["submissions"].items()
                   if item["request"].get("campaign_id") == card["campaign"]["id"]
                   and item["request"].get("study", {}).get("id") == selected["study"]
                   and item["request"]["candidate"]["id"] == candidate]
        if len(matches) > 1:
            _fail("registration has multiple source-bound requests; use a new registration id")
        if not matches:
            if not allow_partial:
                _fail("registered candidate has not been enqueued: " + candidate)
            # Only a partial publication can rely on unexecuted local task
            # declarations, whose exact bytes must still match registration.
            request_id, submission, request = None, {"status": "not_enqueued"}, None
        else:
            request_id, submission = matches[0]
            request = submission["request"]
            if request_id != request["request_id"]:
                _fail("queue submission identity mismatch")
            _verify_request(root, request, selected, card)
        for task_id in card["task_ids"]:
            if request is None:
                task = read_json(root / "configs/forge/tasks" / (task_id + ".json"))
                if stable_hash(task) != card["task_contract_sha256"][task_id]:
                    _fail("unexecuted task changed since registration")
                task_specs[task_id] = _normalized_spec(task)
                definition, job = None, None
            else:
                task_specs[task_id] = _normalized_spec(request["tasks"][task_id])
                definitions = [item for item in request["jobs"] if item["task_id"] == task_id]
                if len(definitions) != 1:
                    _fail("registered task must own exactly one frozen job")
                definition = definitions[0]
                job = state["jobs"].get(definition["compatibility_key"])
                if job and job["definition"] != definition:
                    _fail("queue job differs from its frozen definition")
            if not allow_partial and (job is None or job.get("status") in {"pending", "running"}):
                if submission.get("status") not in {"blocked", "cancelled", "concluded"}:
                    _fail("campaign has unfinished cells; wait or pass --allow-partial")
            provenance = {"schema": EXPORT_SCHEMA, "registration": registration_id,
                          "exporter_sha256": file_hash(Path(__file__)),
                          "registration_file": card_proof,
                          "registration_family": selected["family"], "display_family": family,
                          "request_id": request_id, "candidate_id": candidate, "task_id": task_id,
                          "candidate_revision": request.get("candidate_revision") if request else None,
                          "source_digest": request.get("source", {}).get("digest") if request else None,
                          "compatibility_key": definition["compatibility_key"] if definition else None,
                          "frozen_job_sha256": stable_hash(definition) if definition else None,
                          "frozen_request_sha256": stable_hash(request) if request else None,
                          "observed_queue_cell_sha256": stable_hash(job) if job else None,
                          "frozen_task_sha256": card["task_contract_sha256"][task_id],
                          "original_files": {}, "local_artifact_root": None}
            result = job.get("result") if job else None
            summary = None
            if job and job.get("status") == "terminal" and result:
                attempt_id = result["attempt_id"]
                attempts = [attempt for attempt in job["attempts"] if attempt["attempt_id"] == attempt_id]
                if len(attempts) != 1:
                    _fail("terminal result has no unique original attempt")
                directory = Path(attempts[0]["path"]).resolve()
                if not directory.is_relative_to(queue_root):
                    _fail("original attempt path escapes the selected queue")
                saved, proof = _read(directory / "result.json")
                if saved != result:
                    _fail("terminal queue evidence differs from original result bytes")
                resolved, request_proof = _read(directory / "request.json")
                actual_request = resolved["request"]
                if resolved["job"] != definition or resolved["worker"]["attempt"] != attempt_id:
                    _fail("original attempt differs from frozen job or attempt id")
                provenance.update(attempt_id=attempt_id,
                                  actual_request_id=actual_request["request_id"],
                                  canonical_result_sha256=stable_hash(result),
                                  local_artifact_root=str(directory),
                                  frozen_job=deepcopy(definition),
                                  original_files={"canonical_result": proof, "frozen_request": request_proof})
                for label, name in (("terminal", "terminal.json"), ("raw_result", "raw-result.json"),
                                    ("adapter_receipt", "adapter-receipt.json"), ("graded_result", "graded-result.json")):
                    if (directory / name).is_file():
                        _, provenance["original_files"][label] = _read(directory / name)
                raw = result.get("raw", {}).get("result", {})
                grading = result.get("raw", {}).get("grading", {})
                if request.get("requires_independent_grading") and result.get("raw", {}).get("attempt_status") == "completed":
                    if (grading.get("raw_hash") != stable_hash(raw)
                            or grading.get("source_digest") != actual_request["source"]["digest"]):
                        _fail("independent grading is not bound to the frozen raw result")
                summary = _core_summary(raw, request, task_id, actual_request, result)
                if summary:
                    provenance["bulk_artifacts"] = _artifact_receipts(summary, directory)
                    provenance["summary_origin"] = ("Core field projection from the canonical queue raw result; "
                                                     "the collector supersedes the runner's result.json filename.")
                    summary["export_provenance"] = provenance
            if summary is None:
                status, reason = _unavailable_status(submission, job, result)
                summary = {"schema": UNAVAILABLE_SCHEMA, "candidate_id": candidate, "task_id": task_id,
                           "status": status, "reason": reason, "measurements_available": False,
                           "qualification_input": False, "default_adoption": False,
                           "export_provenance": provenance}
            digest = stable_hash(summary)
            relative = (Path("reports/forge/scaling-receipts") / registration_id /
                        f"{identifier(family)}-{identifier(task_id)}-{digest[:16]}.json").as_posix()
            documents[relative] = summary
            entry = {"trainer_family": family, "candidate_id": candidate, "task_id": task_id, "receipt": relative}
            if summary.get("schema") == UNAVAILABLE_SCHEMA:
                unavailable.append({**entry, "status": summary["status"], "reason": summary["reason"]})
            else:
                entries.append(entry)
    tasks = [{"id": task_id, "spec": spec} for task_id, spec in sorted(task_specs.items())]
    # Validate every projected core receipt before changing current publication.
    with tempfile.TemporaryDirectory(prefix="forge-scaling-export-") as temporary:
        staging = Path(temporary)
        for relative, summary in documents.items():
            atomic_json(staging / relative, summary)
        manifest = build_manifest(staging, entries, tasks=tasks)
        for outcome in unavailable:
            outcome["receipt_sha256"] = file_hash(staging / outcome["receipt"])
        manifest["unavailable"] = unavailable
        manifest["export"] = {"schema": EXPORT_SCHEMA, "registration": registration_id,
                              "exporter_sha256": file_hash(Path(__file__)), "family_selections": roster,
                              "registration_file": card_proof, "partial_snapshot": allow_partial,
                              "counts": dict(Counter(performance_status(item) if item.get("benchmark") == BENCHMARK
                                                     else item["status"] for item in documents.values())),
                              "training_launched": False, "qualification_input": False}
        manifest.pop("input_digest")
        manifest["input_digest"] = stable_hash(manifest)
        atomic_json(staging / MANIFEST, manifest)
        from .scaling_publication import load_performance
        load_performance(staging)
    for relative, summary in documents.items():
        _write_unchanged(root / relative, summary)
    _write_unchanged(root / MANIFEST, manifest)
    output = refresh_performance(root) if refresh else {"training_launched": False}
    return {**output, "registration": registration_id, "manifest": MANIFEST.as_posix(),
            "performance_records": len(entries), "unavailable_records": len(unavailable),
            "counts": manifest["export"]["counts"], "partial_snapshot": allow_partial,
            "bulk_artifacts_copied": False}

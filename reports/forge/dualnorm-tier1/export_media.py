"""Incrementally export certified saved observations while the queue drains.

Progress and stdout stay in the local queue. Completed GIFs are verified and
reused on resume. This adds no model calls, samples, optimizer updates or grades.
The final media.json has the same schema as run.py's from-scratch exporter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import export_attempt

CAMPAIGN = "bcap-dualnorm-tier1-v1"
SOURCE_COMMIT = "15eb7cb0911905e401bdfcd7e264945a7ea64d97"
TASKS = {"gaussian1d_acquisition", "two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_acquisition", "clockfree_audit_measurement_v1"}


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def inventory(queue_root):
    state = read_json(queue_root / "queue/state.json")
    attempts, jobs = set(), {}
    for entry in state["submissions"].values():
        request = entry["request"]
        if request.get("campaign_id") != CAMPAIGN:
            continue
        for declared in request["jobs"]:
            if not TASKS.intersection(declared.get("task_ids", [declared["task_id"]])):
                continue
            key = declared["compatibility_key"]
            job = state["jobs"][key]
            jobs[key] = job["status"]
            attempts.update(attempt["attempt_id"] for attempt in job.get("attempts", []))
    return sorted(attempts), len(jobs) == 287 and set(jobs.values()) == {"terminal"}


def certified(root, queue_root, identity):
    directory = root / "reports/forge/attempts" / identity
    if not all((directory / (name + ".json")).is_file() for name in ("request", "result", "evidence")):
        return None
    envelope, result, certificate = [read_json(directory / (name + ".json")) for name in ("request", "result", "evidence")]
    request = envelope.get("request", envelope)
    if (certificate.get("result_hash") != stable_hash(result)
            or certificate.get("source") != request.get("source")
            or certificate.get("runtime") != request.get("runtime")
            or result.get("candidate_revision") != request.get("candidate_revision")
            or result.get("attempt_id") != identity
            or result.get("retry_of") != envelope.get("retry_of")
            or request.get("campaign_id") != CAMPAIGN
            or request.get("protocol", {}).get("seed") != 0
            or request.get("source", {}).get("origin_commit") != SOURCE_COMMIT
            or stable_hash(request.get("source", {}).get("files", {})) != request.get("source", {}).get("digest")
            or Path(certificate["local_artifact_root"]).resolve() != (queue_root / CAMPAIGN / identity).resolve()):
        raise ValueError("original receipt binding mismatch: " + identity)
    return directory, request, result, certificate


def verify_gif(output, receipt, renderer_hash):
    path = output / (receipt["task_id"] + ".gif")
    if (receipt.get("gif_sha256") != file_hash(path)
            or receipt.get("renderer_sha256") != renderer_hash
            or receipt.get("qualification_input") is not False
            or receipt.get("optimizer_updates_added") != 0
            or receipt.get("sampling_draws_added") != 0
            or receipt.get("kind") != "actual_training_saved_observations_gif"
            or read_json(path.with_suffix(".json")) != receipt):
        raise ValueError("GIF receipt mismatch: " + str(path))
    for name, digest in receipt.get("source_inputs", {}).items():
        if file_hash(Path(name)) != digest:
            raise ValueError("GIF source input differs: " + name)


def verify_observations(row, receipt, local):
    evidence = row["evidence"]
    if "artifact_manifest" in evidence and "comparisons" in evidence:
        # Clock observations are reconstructed from the original comparisons;
        # the exporter verifies its full artifact manifest before rendering.
        proof = Path(evidence["artifact_root"]) / "comparisons.pt"
        expected_inputs = {str(proof): file_hash(proof)}
    else:
        observations = evidence.get("observations", [])
        if (receipt.get("observations_sha256") != stable_hash(observations)
                or receipt.get("observation_count") != len(observations)):
            raise ValueError("media observations differ from the certified original")
        descriptor = evidence.get("saved_observer_outputs")
        expected_inputs = {str((local / descriptor["path"]).resolve()): descriptor["sha256"]} if descriptor else {}
    if receipt.get("source_inputs") != expected_inputs:
        raise ValueError("media source inputs differ from the original descriptor")


def verify_completed(root, queue_root, identity, record, renderer_hash):
    bundle = certified(root, queue_root, identity)
    if bundle is None:
        raise ValueError("processed media lost its original certificate: " + identity)
    _, request, result, certificate = bundle
    if (record["result_hash"] != certificate["result_hash"]
            or record["candidate"] != request["candidate"]["id"]):
        raise ValueError("media progress differs from the original result certificate")
    eligible = {row["task_id"]: row for row in result["task_results"]
                if row["gate_status"] in {"PASS", "FAIL", "BLOCKED"} and row.get("evidence")}
    items = record["items"]
    if len(items) != len(eligible) or {item["task_id"] for item in items} != set(eligible):
        raise ValueError("media progress changed the eligible task roster")
    output = root / "reports/forge/dualnorm-tier1/media" / identity
    for item in items:
        receipt = {key: value for key, value in item.items() if key not in {"candidate", "attempt", "gif"}}
        row = eligible[receipt["task_id"]]
        if (item["candidate"] != record["candidate"] or item["attempt"] != identity
                or item["gif"] != (output / (receipt["task_id"] + ".gif")).relative_to(root).as_posix()
                or receipt["recorded_grade"] != row["gate_status"]):
            raise ValueError("media item differs from its certified task/candidate")
        verify_gif(output, receipt, renderer_hash)
        verify_observations(row, receipt, Path(certificate["local_artifact_root"]))


def process(root, queue_root, identity, renderer_hash):
    bundle = certified(root, queue_root, identity)
    if bundle is None:
        return None
    directory, request, result, certificate = bundle
    output = root / "reports/forge/dualnorm-tier1/media" / identity
    eligible = [row for row in result["task_results"] if row["gate_status"] in {"PASS", "FAIL", "BLOCKED"} and row.get("evidence")]
    receipts = []
    for row in eligible:
        receipt_path = output / (row["task_id"] + ".json")
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if receipt.get("recorded_grade") != row["gate_status"]:
                raise ValueError("existing media grade differs from the original result")
            verify_gif(output, receipt, renderer_hash)
            receipts.append(receipt)
    if len(receipts) != len(eligible):
        if receipts:
            raise ValueError("partial multi-task media export requires explicit review")
        receipts = export_attempt(directory, output)
    for receipt in receipts:
        verify_gif(output, receipt, renderer_hash)
        row = next(row for row in eligible if row["task_id"] == receipt["task_id"])
        verify_observations(row, receipt, Path(certificate["local_artifact_root"]))
    items = [{"candidate": request["candidate"]["id"], "attempt": identity, **receipt,
              "gif": (output / (receipt["task_id"] + ".gif")).relative_to(root).as_posix()}
             for receipt in receipts]
    return {"result_hash": certificate["result_hash"], "candidate": request["candidate"]["id"],
            "status": "exported" if items else "no_eligible_observations", "items": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path, default=ROOT / "runs/forge/bcap-dualnorm-tier1-v1-queue")
    parser.add_argument("--once", action="store_true", help="export available completed attempts without waiting")
    parser.add_argument("--verify-only", action="store_true", help="wait for existing exports and verify them without rendering")
    args = parser.parse_args()
    root, queue_root = args.root.resolve(), args.queue_root.resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(queue_root / "media-mpl-cache"))
    import torch
    torch.set_num_threads(1)
    renderer_hash = file_hash(root / "experiments/forge/tier1_media.py")
    progress_path = queue_root / "media-progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {
        "schema_version": 1, "campaign": CAMPAIGN, "renderer_sha256": renderer_hash,
        "processed_attempts": {}, "errors": {}}
    if progress.get("renderer_sha256") != renderer_hash:
        raise ValueError("renderer changed; preserve original exports and review before resuming")
    while True:
        if args.verify_only and progress_path.is_file():
            progress = read_json(progress_path)
        identities, terminal = inventory(queue_root)
        for identity in identities:
            if args.verify_only:
                continue
            if identity in progress["processed_attempts"] or identity in progress["errors"]:
                continue
            try:
                value = process(root, queue_root, identity, renderer_hash)
                if value is None:
                    continue
                progress["processed_attempts"][identity] = value
                emit("media_exported", attempt=identity, gifs=len(value["items"]), processed=len(progress["processed_attempts"]))
            except Exception as error:
                progress["errors"][identity] = {"type": type(error).__name__, "message": str(error)}
                emit("media_error", attempt=identity, **progress["errors"][identity])
            atomic_json(progress_path, progress)
        items = [item for identity in sorted(progress["processed_attempts"])
                 for item in progress["processed_attempts"][identity]["items"]]
        accounted = set(progress["processed_attempts"]) | set(progress["errors"])
        done = terminal and set(identities) <= accounted
        if not args.verify_only or done or args.once:
            atomic_json(root / "reports/forge/dualnorm-tier1/media.json", {
                "schema_version": 1, "qualification_input": False, "items": items})
        emit("media_progress", queue_terminal=terminal, finished=done, attempts=len(identities),
             processed=len(progress["processed_attempts"]), gifs=len(items), errors=len(progress["errors"]))
        if args.once or done:
            for identity, record in progress["processed_attempts"].items():
                verify_completed(root, queue_root, identity, record, renderer_hash)
            progress["verification"] = {"processed_attempts": len(progress["processed_attempts"]),
                                        "gifs": len(items), "queue_terminal": terminal,
                                        "renderer_sha256": renderer_hash, "all_original_certificates_and_inputs_valid": True}
            atomic_json(progress_path, progress)
            emit("media_verified", **progress["verification"])
            if progress["errors"]:
                raise SystemExit(2)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()

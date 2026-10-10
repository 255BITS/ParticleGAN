"""Archive and verify original v2 files; never import a model or regrade science."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile


ROUND = "gaussian-smoke-inventory-v2"
SOURCE = "f802fcf9b04d32d2027321af0894a78a2d1343dd"


def read(path):
    return json.loads(path.read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def smoke_confirmation(task):
    result = dict(task["evaluator_result"])
    step = result["first_confirmed_step"]
    result["first_primary"] = next(row for row in task["evidence"]["observations"] if row["step"] == step)
    result["first_confirmation"] = next(row for row in task["evidence"]["confirmations"] if row["step"] == step)
    return result


def verify(path):
    with tarfile.open(path, "r:gz") as archive:
        inventory = json.load(archive.extractfile("archive-members.json"))
        members = {entry.name: entry for entry in archive.getmembers()}
        if set(members) != set(inventory["files"]) | {"archive-members.json"}:
            raise ValueError("archive member set differs from its exact inventory")
        for name, expected in inventory["files"].items():
            member = members[name]
            if not member.isfile() or member.size != expected["bytes"]:
                raise ValueError("archive member type/size differs: " + name)
            hasher = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            if hasher.hexdigest() != expected["sha256"]:
                raise ValueError("archive member bytes differ: " + name)
    return inventory


def create(root, archive_path, report_directory):
    queue = root / "runs/forge" / ROUND
    state = read(queue / "queue/state.json")
    interruption = read(queue / "interruption-receipt.json")
    if any(row["status"] == "running" for row in state["jobs"].values()):
        raise ValueError("finish existing workers before archiving the queue")
    campaign = state["campaigns"][ROUND]
    paid = sum(row["seconds"] for row in state["charges"])
    if campaign["reserved_seconds"] != 0 or not math.isclose(paid, interruption["spent_seconds"], abs_tol=1e-9):
        raise ValueError("interrupted campaign accounting is inconsistent")
    attempts = sorted({entry["attempt_id"] for job in state["jobs"].values() for entry in job["attempts"]})
    originals = [root / "reports/forge/attempts" / attempt for attempt in attempts]
    searches = sorted((root / "reports/forge/configuration-search").glob("gaussian-smoke-inventory-*-v2.json"))
    if len(attempts) != 18 or len(searches) != 12:
        raise ValueError("expected all 18 original attempts and 12 registered search reports")
    files = {path for directory in [queue, *originals] for path in directory.rglob("*") if path.is_file()}
    files.update(searches)
    files.update(root / name for name in (
        "configs/forge/rounds/gaussian-smoke-inventory-v2.json",
        "configs/forge/campaigns/gaussian-smoke-inventory-v2.json"))
    source_digests, rows = set(), []
    for attempt in attempts:
        directory = root / "reports/forge/attempts" / attempt
        envelope, result = read(directory / "request.json"), read(directory / "result.json")
        request = envelope["request"]
        if request["source"]["origin_commit"] != SOURCE:
            raise ValueError("attempt belongs to another executed source")
        source_digests.add(request["source"]["digest"])
        if result["attempt_id"] != attempt or len(result["task_results"]) != 1:
            raise ValueError("unexpected v2 attempt identity/group")
        task = result["task_results"][0]
        row = dict(attempt_id=attempt, candidate_id=request["candidate"]["id"],
            candidate_revision=result["candidate_revision"], task_id=task["task_id"],
            compatibility_key=task["compatibility_key"], raw_status=result["raw"]["attempt_status"],
            gate_status=task["gate_status"], charged_seconds=result["raw"]["elapsed_seconds"],
            completed_steps=task.get("cost", {}).get("completed_steps"),
            metrics=task.get("metrics", {}), source_digest=request["source"]["digest"],
            original_receipts={name: digest(directory / (name + ".json")) for name in ("request", "evidence", "result")})
        if task["task_id"] == "gaussian1d_smoke":
            row["smoke_confirmation"] = smoke_confirmation(task)
        if row["raw_status"] == "error":
            worker = Path(envelope["worker"]["directory"])
            events = []
            for line in (worker / "run.log").read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "observation":
                    events.append(event)
            row.update(error=task["error"], declared_updates=1600, original_schedule_horizon=400,
                last_logged_observation=max(event["step"] for event in events),
                logged_observations=len(events), checkpoint_present=any(worker.rglob("*.pt")),
                interpretation="software execution error; no numerical failure verdict")
            if (row["task_id"] != "ring16_acquisition" or row["gate_status"] != "INCOMPLETE"
                    or row["last_logged_observation"] != 400 or row["checkpoint_present"]):
                raise ValueError("unexpected v2 execution error; review before publication")
        rows.append(row)
    if len(source_digests) != 1 or not math.isclose(sum(row["charged_seconds"] for row in rows), paid, abs_tol=1e-9):
        raise ValueError("original attempt source/cost accounting differs from the queue")
    inventory = {"schema_version": 1, "round": ROUND, "files": {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(files)}}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise ValueError("archive is immutable; choose a new path rather than overwrite")
    with archive_path.open("wb") as destination, gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in inventory["files"]:
                archive.add(root / name, arcname=name, recursive=False)
            payload = encoded(inventory)
            member = tarfile.TarInfo("archive-members.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    if verify(archive_path) != inventory:
        raise ValueError("saved inventory differs from original file identities")
    if any(digest(root / name) != value["sha256"] for name, value in inventory["files"].items()):
        raise ValueError("source files changed while archiving")
    receipt = dict(schema_version=1, round=ROUND, qualification_input=False,
        source_commit=SOURCE, source_digest=next(iter(source_digests)), status="interrupted_software_cohort",
        spent_seconds=paid, scientific_failure_retries=0, successor="gaussian-smoke-inventory-v3",
        jobs=dict(Counter(row["status"] for row in state["jobs"].values())),
        requests=dict(Counter(row["status"] for row in state["submissions"].values())),
        gates=dict(Counter(row["gate_status"] for row in rows)), original_attempts=len(attempts),
        registered_search_reports=len(searches), reserved_seconds=campaign["reserved_seconds"],
        archive={"path": "artifacts/" + archive_path.name, "bytes": archive_path.stat().st_size,
                 "sha256": digest(archive_path), "member_count": len(inventory["files"]) + 1,
                 "original_file_bytes": sum(row["bytes"] for row in inventory["files"].values()),
                 "inventory_sha256": hashlib.sha256(encoded(inventory)).hexdigest(), "byte_exact_verified": True},
        search_report_hashes={str(path.relative_to(root)): digest(path) for path in searches},
        interruption_receipt_sha256=digest(queue / "interruption-receipt.json"))
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / "provenance.json").write_bytes(encoded(receipt))
    (report_directory / "attempts.json").write_bytes(encoded({"schema_version": 1,
        "qualification_input": False, "round": ROUND, "source_commit": SOURCE, "attempts": rows}))
    print(json.dumps(receipt["archive"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--report-directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify_only:
        data = verify(arguments.archive)
        print(json.dumps({"verified_files": len(data["files"]), "archive_sha256": digest(arguments.archive)}))
    else:
        if arguments.source_root is None:
            parser.error("--source-root is required for archiving original files")
        create(arguments.source_root.resolve(), arguments.archive.resolve(), arguments.report_directory.resolve())

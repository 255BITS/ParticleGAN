"""Byte-exact final inventory archive; no models, draws, training or regrading.

Run only after drain and refresh of the twelve registered search reports.
Bulk output must live under an ignored Git worktree artifacts/ directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import fcntl
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROUND = "gaussian-smoke-inventory-v4"
SOURCE = "79fdf16d2ed880a9db1873245f150375e3be31b0"
SOURCE_DIGEST = "d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def idle_lock(path):
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("active coordinator or worker: " + str(path)) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def final_accounting(state):
    if any(row["status"] == "running" for row in state["jobs"].values()):
        raise ValueError("active workers remain")
    if any(row["status"] in {"queued", "running", "paused"} for row in state["submissions"].values()):
        raise ValueError("active campaign requests remain")
    if set(state["campaigns"]) != {ROUND}:
        raise ValueError("archive requires the single declared inventory campaign")
    campaign = state["campaigns"][ROUND]
    if campaign["reserved_seconds"] != 0:
        raise ValueError("campaign still reserves worker cost")
    charges = state["charges"]
    if any(row["owner"]["campaign"] != ROUND for row in charges):
        raise ValueError("queue charges belong to another campaign")
    paid = sum(row["seconds"] for row in charges)
    if not math.isfinite(paid) or not math.isclose(paid, campaign["spent_seconds"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("queue charge accounting differs from campaign")
    attempts = sorted({entry["attempt_id"] for job in state["jobs"].values() for entry in job["attempts"]})
    if len(charges) != len(attempts) or {row["attempt_id"] for row in charges} != set(attempts):
        raise ValueError("every original attempt must have exactly one finalized charge")
    return campaign, paid, attempts


def ignored_destination(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        worktree = Path(subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"], text=True,
            stderr=subprocess.DEVNULL).strip()).resolve()
        relative = path.relative_to(worktree)
        ignored = subprocess.run(["git", "-C", str(worktree), "check-ignore", "--quiet", str(relative)])
    except (ValueError, subprocess.CalledProcessError) as error:
        raise ValueError("archive destination needs an ignored worktree artifacts/ path") from error
    if relative.parts[0] != "artifacts" or ignored.returncode != 0:
        raise ValueError("archive destination needs an ignored worktree artifacts/ path")
    return relative.as_posix()


def collected_files(root, queue, originals, declarations, searches):
    files = set()
    for directory in [queue, *originals]:
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("archive input may not be a symlink: " + str(path))
            if path.is_file():
                files.add(path)
            elif not path.is_dir():
                raise ValueError("archive input must be a regular file: " + str(path))
    for path in [*declarations, *searches]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or nonregular declared archive input: " + str(path))
        files.add(path)
    return {path.relative_to(root).as_posix(): path for path in sorted(files)}


def verify(path):
    with tarfile.open(path, "r:gz") as archive:
        entries = archive.getmembers()
        members = {entry.name: entry for entry in entries}
        if len(entries) != len(members):
            raise ValueError("duplicate archive members")
        inventory = json.load(archive.extractfile("archive-members.json"))
        if set(members) != set(inventory["files"]) | {"archive-members.json"}:
            raise ValueError("archive member set differs from its exact inventory")
        for name, expected in inventory["files"].items():
            normalized = PurePosixPath(name)
            if normalized.is_absolute() or ".." in normalized.parts or normalized.as_posix() != name:
                raise ValueError("archive member path is not normalized")
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


def _create(root, queue, archive_path, receipt_path, stack):
    state = read(queue / "queue/state.json")
    campaign, paid, attempts = final_accounting(state)
    launch = read(queue / "launch-receipt.json")
    if (launch["source_origin_commit"], launch["source_digest"]) != (SOURCE, SOURCE_DIGEST):
        raise ValueError("launch receipt belongs to another executed source")
    for row in state["submissions"].values():
        source = row["request"]["source"]
        if (source["origin_commit"], source["digest"]) != (SOURCE, SOURCE_DIGEST):
            raise ValueError("submission belongs to another executed source")
    snapshot = queue / "snapshots" / SOURCE_DIGEST
    frozen = read(snapshot / "forge-source.json")
    if (frozen["origin_commit"], frozen["digest"]) != (SOURCE, SOURCE_DIGEST):
        raise ValueError("frozen source identity differs from launch")
    if hashlib.sha256(json.dumps(frozen["files"], sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode()).hexdigest() != SOURCE_DIGEST:
        raise ValueError("frozen source manifest digest differs from its files")
    for relative, expected in frozen["files"].items():
        path = snapshot / relative
        if not path.resolve().is_relative_to(snapshot) or digest(path) != expected:
            raise ValueError("frozen source file changed: " + relative)
    originals = []
    for attempt in attempts:
        directory = root / "reports/forge/attempts" / attempt
        originals.append(directory)
        envelope = read(directory / "request.json")
        for name in ("evidence.json", "result.json"):
            read(directory / name)
        source = envelope["request"]["source"]
        if (source["origin_commit"], source["digest"]) != (SOURCE, SOURCE_DIGEST):
            raise ValueError("original attempt belongs to another executed source")
        if read(directory / "result.json")["attempt_id"] != attempt:
            raise ValueError("durable result identity differs from queue attempt")
        worker = Path(envelope["worker"]["directory"]).resolve()
        if not worker.is_relative_to(queue) or not worker.is_dir():
            raise ValueError("worker artifacts are outside the complete queue archive")
        if (worker / "execution.lock").exists():
            stack.enter_context(idle_lock(worker / "execution.lock"))
    round_path = root / "configs/forge/rounds" / (ROUND + ".json")
    definition = read(round_path)
    expected_studies = set(definition["studies"]) - set(definition["study_refusals"])
    searches = sorted(root / "reports/forge/configuration-search" / (study + ".json")
                      for study in expected_studies)
    if not searches or any(not path.is_file() for path in searches):
        raise ValueError("archive requires every registered final search report")
    for path in searches:
        search = read(path)
        accounting = search.get("campaign_accounting") or {}
        if (search["source_digest"] != SOURCE_DIGEST or Path(search["queue_root"]).resolve() != queue
                or search["campaign"]["id"] != ROUND or accounting.get("reserved_seconds") != 0
                or not math.isclose(accounting.get("spent_seconds", -1), paid, rel_tol=0, abs_tol=1e-8)):
            raise ValueError("refresh final source-bound search report before archive: " + str(path))
        for trial in search["trials"]:
            request_id = trial["request_id"]
            expected = {entry["attempt_id"] for job in state["jobs"].values()
                        if request_id in job["subscribers"] for entry in job["attempts"]}
            if (request_id not in state["submissions"] or trial["status"] == "UNKNOWN"
                    or set(trial.get("attempt_ids", [])) != expected):
                raise ValueError("search report does not preserve its final original attempt set")
    declarations = [round_path, root / definition["campaign"],
                    *(root / "configs/forge/searches" / (study + ".json") for study in definition["studies"])]
    for name in ("drain.log", "controller.log"):
        if not (queue / name).is_file():
            raise ValueError("preserve both preparation drain.log and execution controller.log")
    files = collected_files(root, queue, originals, declarations, searches)
    inventory = {"schema_version": 1, "campaign": ROUND, "source_origin_commit": SOURCE,
        "source_digest": SOURCE_DIGEST, "files": {name: {"bytes": path.stat().st_size,
        "sha256": digest(path)} for name, path in files.items()}}
    with archive_path.open("xb") as destination, gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0, compresslevel=6) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, path in files.items():
                member = tarfile.TarInfo(name)
                member.size = inventory["files"][name]["bytes"]
                member.mode = 0o644  # Canonical metadata; original file bytes remain exact.
                with path.open("rb") as handle:
                    archive.addfile(member, handle)
            payload = encoded(inventory)
            member = tarfile.TarInfo("archive-members.json")
            member.size, member.mode = len(payload), 0o644
            archive.addfile(member, io.BytesIO(payload))
    if verify(archive_path) != inventory:
        raise ValueError("saved inventory differs from the original files")
    if (collected_files(root, queue, originals, declarations, searches) != files
            or any(digest(path) != inventory["files"][name]["sha256"] for name, path in files.items())):
        raise ValueError("original file set or bytes changed while archiving")
    final_accounting(read(queue / "queue/state.json"))
    receipt = {"schema_version": 1, "campaign": ROUND, "source_origin_commit": SOURCE,
        "source_digest": SOURCE_DIGEST, "qualification_input": False, "finalized": True,
        "attempts": attempts, "attempt_count": len(attempts), "paid_seconds": paid,
        "previous_paid_seconds": launch["previous_paid_seconds"],
        "combined_paid_seconds": paid + launch["previous_paid_seconds"],
        "reserved_seconds": campaign["reserved_seconds"], "scientific_retries": 0,
        "job_status_counts": dict(Counter(row["status"] for row in state["jobs"].values())),
        "submission_status_counts": dict(Counter(row["status"] for row in state["submissions"].values())),
        "archive": ignored_destination(archive_path), "sha256": digest(archive_path),
        "bytes": archive_path.stat().st_size, "original_files": len(files), "members": len(files) + 1,
        "original_file_bytes": sum(row["bytes"] for row in inventory["files"].values()),
        "members_digest": hashlib.sha256(encoded(inventory)).hexdigest(), "byte_exact_verified": True,
        "search_report_hashes": {path.relative_to(root).as_posix(): digest(path) for path in searches},
        "preserved_log_hashes": {name: digest(queue / name) for name in ("drain.log", "controller.log")}}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(encoded(receipt))
    return receipt


def create(root, archive_path, receipt_path):
    root, archive_path, receipt_path = map(lambda path: Path(path).resolve(), (root, archive_path, receipt_path))
    ignored_destination(archive_path)
    if archive_path.exists() or receipt_path.exists():
        raise ValueError("archive and receipt are immutable; choose new paths")
    queue = root / "runs/forge" / ROUND
    with ExitStack() as stack:
        stack.enter_context(idle_lock(queue / "coordinator.lock"))
        stack.enter_context(idle_lock(queue / "queue.lock"))
        return _create(root, queue, archive_path, receipt_path, stack)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--round", default=ROUND)
    parser.add_argument("--source-commit", default=SOURCE)
    parser.add_argument("--source-digest", default=SOURCE_DIGEST)
    args = parser.parse_args()
    ROUND, SOURCE, SOURCE_DIGEST = args.round, args.source_commit, args.source_digest
    if args.verify_only:
        inventory = verify(args.archive)
        print(json.dumps({"verified_files": len(inventory["files"]), "archive_sha256": digest(args.archive)}))
    else:
        if args.source_root is None or args.receipt is None:
            parser.error("--source-root and --receipt are required for creation")
        receipt = create(args.source_root, args.archive, args.receipt)
        print(json.dumps({key: receipt[key] for key in ("campaign", "bytes", "sha256", "attempt_count", "paid_seconds")}, sort_keys=True))

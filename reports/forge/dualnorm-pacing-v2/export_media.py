"""Export media and archive the completed pacing study without new model calls.

Progress and stdout stay in the local queue. Completed GIFs are verified and
reused on resume. This adds no model calls, samples, optimizer updates or grades.
The final media-index.json binds every GIF to its original numerical evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import export_attempt

CAMPAIGN = "bcap-dualnorm-pacing-v2"
SOURCE_COMMIT = "a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be"
TASKS = {"gaussian1d_acquisition", "two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_acquisition", "clockfree_audit_measurement_v1"}
SOURCE_DIGEST = "f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226"
REPORT_RELATIVE = Path("reports/forge/dualnorm-pacing-v2")
MANIFEST_NAME = "archive-member-manifest.json"


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
    return sorted(attempts), len(jobs) == 175 and set(jobs.values()) == {"terminal"}


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
            or request.get("source", {}).get("digest") != SOURCE_DIGEST
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
    output = root / "reports/forge/dualnorm-pacing-v2/media" / identity
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
    output = root / "reports/forge/dualnorm-pacing-v2/media" / identity
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


class ArchiveReader:
    """Hash exactly the bytes consumed by tarfile."""
    def __init__(self, stream):
        self.stream, self.digest, self.bytes = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def streaming_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def archive(root, queue_root):
    report = root / REPORT_RELATIVE
    destination = root / "artifacts/forge" / (CAMPAIGN + ".tar.gz")
    if destination.exists():
        raise ValueError("archive already exists; preserve its immutable identity")
    identities, terminal = inventory(queue_root)
    if not terminal or len(identities) != 175:
        raise ValueError("only the completed 175-attempt campaign can be archived")
    frozen = read_json(queue_root / "frozen-study.json")
    source = frozen["source"]
    if (source["origin_commit"] != SOURCE_COMMIT or source["digest"] != SOURCE_DIGEST
            or stable_hash(source["files"]) != SOURCE_DIGEST):
        raise ValueError("unexpected scientific source")
    snapshot = queue_root / "snapshots" / SOURCE_DIGEST
    for name, digest in source["files"].items():
        if file_hash(snapshot / name) != digest:
            raise ValueError("executed snapshot bytes changed: " + name)
    progress = read_json(queue_root / "media-progress.json")
    if progress.get("errors") or set(progress["processed_attempts"]) != set(identities):
        raise ValueError("all attempt media must be exported before archiving")
    renderer_hash = file_hash(root / "experiments/forge/tier1_media.py")
    for identity in identities:
        verify_completed(root, queue_root, identity, progress["processed_attempts"][identity], renderer_hash)
    # The complete source/input bytes are preserved independently of the merged
    # publication checkout. Certificates and their absolute original roots stay
    # byte-exact; restoration recreates the recorded locations or links them.
    executed_inputs = {}
    for name, digest in frozen["inputs_sha256"].items():
        blob = subprocess.check_output(["git", "show", SOURCE_COMMIT + ":" + name], cwd=root)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("committed execution input differs from freeze: " + name)
        executed_inputs[name] = blob
    files = {}

    def collect(path, prefix):
        for member in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if member.is_symlink():
                raise ValueError("archive evidence cannot contain symlinks: " + str(member))
            if not member.is_file() or "__pycache__" in member.parts:
                continue
            name = str(Path(prefix) / member.relative_to(path)) if path.is_dir() else prefix
            if name in files and files[name] != member:
                raise ValueError("archive member collision: " + name)
            files[name] = member

    collect(queue_root / "queue", "queue")
    collect(queue_root / CAMPAIGN, "campaign/" + CAMPAIGN)
    collect(snapshot, "snapshots/" + SOURCE_DIGEST)
    for path in sorted(queue_root.iterdir()):
        if path.is_file() and not path.name.endswith(".lock") and not path.name.startswith("archive-"):
            collect(path, "execution/" + path.name)
    for identity in identities:
        collect(root / "reports/forge/attempts" / identity, "durable/" + identity)
    for path in sorted(report.iterdir()):
        if path.name not in {"artifact-inventory.json", "archive-verification.json", "artifact-provenance.json"}:
            collect(path, "publication/" + path.name)
    results = read_json(report / "results.json")
    for stage in results["stages"]:
        if stage.get("admitted"):
            name = stage["spec"]["id"] + ".json"
            collect(root / "reports/forge/configuration-search" / name, "publication/configuration-search/" + name)
    software = report / "software-verification.json"
    if not software.is_file():
        raise ValueError("final software-verification.json is required before archiving")
    for artifact in read_json(software).get("artifacts", {}).values():
        path = Path(artifact["local_path"])
        path = path if path.is_absolute() else root / path
        if file_hash(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError("software artifact differs from certified bytes: " + str(path))
        if path.resolve().parent != queue_root.resolve():
            collect(path, "verification/external/" + path.name)
    members = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "x:gz", compresslevel=1) as bundle:
        for index, (name, path) in enumerate(sorted(files.items()), 1):
            before = path.stat()
            info = bundle.gettarinfo(path, arcname=name)
            with path.open("rb") as stream:
                reader = ArchiveReader(stream)
                bundle.addfile(info, reader)
            after = path.stat()
            if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size or reader.bytes != info.size:
                raise ValueError("archive input changed while reading: " + str(path))
            members[name] = {"sha256": reader.digest.hexdigest(), "bytes": reader.bytes}
            if index % 500 == 0:
                emit("archive_progress", members=index)
        for name, blob in sorted(executed_inputs.items()):
            name = "executed-inputs/" + name
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            bundle.addfile(info, BytesIO(blob))
            members[name] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        manifest = {"schema_version": 1, "campaign": CAMPAIGN, "source_commit": SOURCE_COMMIT,
                    "source_digest": SOURCE_DIGEST, "members": members}
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        bundle.addfile(info, BytesIO(payload))
    atomic_json(queue_root / MANIFEST_NAME, manifest)
    atomic_json(report / "artifact-inventory.json", {
        "schema_version": 1, "campaign": CAMPAIGN, "availability": "LOCAL_ONLY",
        "remote_replication": "NOT_PERFORMED", "qualification_input": False,
        "archive": {"path": destination.relative_to(root).as_posix(), "local_path": str(destination),
                    "sha256": streaming_hash(destination), "bytes": destination.stat().st_size},
        "manifest": {"archive_member": MANIFEST_NAME, "sha256": hashlib.sha256(payload).hexdigest(),
                     "bytes": len(payload), "local_path": str(queue_root / MANIFEST_NAME)},
        "member_count": len(members) + 1, "payload_bytes": sum(item["bytes"] for item in members.values()),
        "members_by_prefix": dict(sorted(Counter(name.split("/")[0] for name in members).items())),
        "attempts": len(identities), "gifs": sum(len(row["items"]) for row in progress["processed_attempts"].values()),
        "source_digests": [SOURCE_DIGEST], "executed_source_commit": SOURCE_COMMIT})
    verify_archive(root, queue_root)


def verify_archive(root, queue_root):
    report = root / REPORT_RELATIVE
    inventory_record = read_json(report / "artifact-inventory.json")
    archive_record = inventory_record["archive"]
    destination = root / archive_record["path"]
    if streaming_hash(destination) != archive_record["sha256"] or destination.stat().st_size != archive_record["bytes"]:
        raise ValueError("archive bytes differ from immutable inventory")
    with tarfile.open(destination, "r:gz") as bundle:
        entries = bundle.getmembers()
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)) or any(not entry.isfile() for entry in entries):
            raise ValueError("archive has duplicate or non-file members")
        payload = bundle.extractfile(MANIFEST_NAME).read()
        if hashlib.sha256(payload).hexdigest() != inventory_record["manifest"]["sha256"]:
            raise ValueError("archive manifest differs")
        manifest = json.loads(payload)
        if (manifest.get("source_commit") != SOURCE_COMMIT or manifest.get("source_digest") != SOURCE_DIGEST
                or len(names) != inventory_record["member_count"]):
            raise ValueError("archive scientific identity or member count differs")
        if set(names) != set(manifest["members"]) | {MANIFEST_NAME}:
            raise ValueError("archive member set differs from the full manifest")
        frozen = json.load(bundle.extractfile("execution/frozen-study.json"))
        if (frozen["source"]["origin_commit"] != SOURCE_COMMIT
                or frozen["source"]["digest"] != SOURCE_DIGEST
                or stable_hash(frozen["source"]["files"]) != SOURCE_DIGEST):
            raise ValueError("archived freeze differs from execution identity")
        for name, digest in frozen["source"]["files"].items():
            if manifest["members"]["snapshots/" + SOURCE_DIGEST + "/" + name]["sha256"] != digest:
                raise ValueError("archived source bytes differ from freeze: " + name)
        for name, digest in frozen["inputs_sha256"].items():
            if manifest["members"]["executed-inputs/" + name]["sha256"] != digest:
                raise ValueError("archived prospective input bytes differ from freeze: " + name)
        for index, name in enumerate(sorted(manifest["members"]), 1):
            with bundle.extractfile(name) as stream:
                reader = ArchiveReader(stream)
                while reader.read(1024 * 1024):
                    pass
            if {"sha256": reader.digest.hexdigest(), "bytes": reader.bytes} != manifest["members"][name]:
                raise ValueError("archive member bytes differ: " + name)
            if index % 1000 == 0:
                emit("archive_verified_progress", members=index)
    atomic_json(report / "archive-verification.json", {
        "schema_version": 1, "status": "PASS", "campaign": CAMPAIGN,
        "archive_sha256": archive_record["sha256"], "archive_bytes": archive_record["bytes"],
        "all_members_individually_inspected": len(names), "exact_member_set": True,
        "manifest_sha256": inventory_record["manifest"]["sha256"],
        "certified_attempts": 175, "source_commit": SOURCE_COMMIT, "source_digest": SOURCE_DIGEST,
        "source_snapshot_files_verified": len(frozen["source"]["files"]),
        "prospective_input_files_verified": len(frozen["inputs_sha256"]),
        "new_training_updates": 0, "new_sampler_calls": 0, "qualification_changed": False,
        "availability": "LOCAL_ONLY", "remote_replication": "NOT_PERFORMED"})
    emit("archived_and_verified", members=len(names), bytes=archive_record["bytes"], sha256=archive_record["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path, default=ROOT / "runs/forge/bcap-dualnorm-pacing-v2-queue")
    parser.add_argument("--once", action="store_true", help="export available completed attempts without waiting")
    parser.add_argument("--verify-only", action="store_true", help="wait for existing exports and verify them without rendering")
    parser.add_argument("--archive", action="store_true", help="create the immutable archive after final software receipts close")
    parser.add_argument("--verify-archive", action="store_true", help="inspect every archived member against its exact manifest")
    args = parser.parse_args()
    root, queue_root = args.root.resolve(), args.queue_root.resolve()
    if args.archive or args.verify_archive:
        if args.archive and args.verify_archive:
            parser.error("choose archive creation or archive verification")
        (archive if args.archive else verify_archive)(root, queue_root)
        return
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
            atomic_json(root / "reports/forge/dualnorm-pacing-v2/media-index.json", {
                "schema_version": 1, "campaign": CAMPAIGN, "source_commit": SOURCE_COMMIT,
                "source_digest": "f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226",
                "qualification_input": False, "optimizer_updates_added": 0,
                "sampling_draws_added": 0, "attempt_count": len(identities), "items": items})
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

"""Publish compact outcomes, genuine training media and exact archive identities."""
from collections import Counter
from pathlib import Path
import argparse
import json
import tarfile

from prepare import QUEUE, REPORT, ROOT, emit
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.queue import Queue


def attempts_and_rows():
    state = Queue(QUEUE, report_root=ROOT / "reports/forge", on_completion=None).inspect()
    attempts, rows = {}, []
    for request_id, entry in sorted(state["submissions"].items()):
        request = entry["request"]
        names = {a["task"] for a in request["view"]["assignments"] if a["qualification_tier"] <= request["through_tier"]}
        measured = {}
        for job in request["jobs"]:
            if not Queue._authorized(request, job):
                continue
            saved = state["jobs"][job["compatibility_key"]]
            for attempt in saved["attempts"]:
                attempts[attempt["attempt_id"]] = Path(attempt["path"])
            for row in (saved.get("result") or {}).get("task_results", []):
                if row["task_id"] in names:
                    measured[row["task_id"]] = {"task_id": row["task_id"], "status": row["gate_status"],
                        "reason": row.get("reason"), "metrics": row.get("metrics", {}),
                        "cost": row.get("cost", {}),
                        "attempt_id": saved["result"]["attempt_id"],
                        "canonical_result_sha256": stable_hash(saved["result"])}
        tasks = [measured.get(name, {"task_id": name, "status": "UNKNOWN"}) for name in sorted(names)]
        rows.append({"candidate": request["candidate"]["id"], "request_id": request_id,
            "candidate_revision": request["candidate_revision"], "view": request["view"]["id"],
            "campaign": request["campaign_id"], "submission_status": entry["status"],
            "source_digest": request["source"]["digest"], "executed_commit": request["source"]["origin_commit"],
            "recipe_overrides": request["candidate"].get("recipe_overrides", {}),
            "counts": dict(Counter(t["status"] for t in tasks)), "tasks": tasks})
    return state, attempts, rows


def summarize():
    state, attempts, rows = attempts_and_rows()
    cost = {name: {k: value[k] for k in ("definition", "spent_seconds", "reserved_seconds")}
            for name, value in state["campaigns"].items()}
    report = {"schema_version": 1, "scope": "bcap_tier1_repair_readout", "qualification_input": False,
        "status": "COMPLETE" if all(r["submission_status"] not in {"queued", "running", "paused"}
            and all(t["status"] in {"PASS", "FAIL"} for t in r["tasks"]) for r in rows) else "IN_PROGRESS",
        "candidates": rows, "campaign_accounting": cost, "attempt_count": len(attempts),
        "total_new_paid_seconds": sum(v["spent_seconds"] for v in cost.values()),
        "logs": str(QUEUE / "events.jsonl")}
    atomic_json(REPORT / "results.json", report)
    emit("repair_readout", attempts=len(attempts), paid_seconds=report["total_new_paid_seconds"],
         counts=dict(Counter(t["status"] for r in rows for t in r["tasks"])))


def media():
    from experiments.forge.tier1_media import export_attempt
    _, attempts, _ = attempts_and_rows()
    items = []
    for attempt_id in sorted(attempts):
        directory = ROOT / "reports/forge/attempts" / attempt_id
        if not (directory / "evidence.json").exists():
            continue
        output = REPORT / "media" / attempt_id
        for receipt in export_attempt(directory, output):
            items.append({"attempt_id": attempt_id, **receipt,
                          "gif": str((output / (receipt["task_id"] + ".gif")).relative_to(ROOT))})
        emit("media_published", attempt=attempt_id)
    atomic_json(REPORT / "media.json", {"schema_version": 1, "qualification_input": False, "items": items})


def archive(destination):
    state, attempts, _ = attempts_and_rows()
    if (any(e["status"] in {"queued", "running", "paused"} for e in state["submissions"].values())
            or any(job["status"] == "running" for job in state["jobs"].values())
            or any(campaign["reserved_seconds"] for campaign in state["campaigns"].values())):
        raise ValueError("finish or cancel admitted work before archiving")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("archive already exists; preserve its published identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = {entry["request"]["source"]["digest"]: entry["request"]["source"]
               for entry in state["submissions"].values()}
    for attempt_id, local in attempts.items():
        durable = ROOT / "reports/forge/attempts" / attempt_id
        if not local.is_dir() or not durable.is_dir():
            raise ValueError("missing original attempt or durable envelope: " + attempt_id)
        for filename in ("request.json", "result.json", "evidence.json"):
            if not (durable / filename).is_file():
                raise ValueError("missing original durable " + filename + ": " + attempt_id)
    for digest, source in sources.items():
        snapshot = QUEUE / "snapshots" / digest
        for relative, expected in source["files"].items():
            if not (snapshot / relative).is_file() or file_hash(snapshot / relative) != expected:
                raise ValueError("missing or changed original source: " + relative)
    import tempfile
    staging = tempfile.TemporaryDirectory(prefix=".bcap-archive-", dir=destination.parent)
    temporary = Path(staging.name) / "artifacts.tar.gz"
    manifest = {}
    with tarfile.open(temporary, "w:gz") as bundle:
        def add_tree(directory, prefix):
            files = [directory] if directory.is_file() else sorted(p for p in directory.rglob("*") if p.is_file())
            for path in files:
                if path.is_symlink() or "__pycache__" in path.parts or path.suffix == ".lock":
                    continue
                name = str(Path(prefix) / path.relative_to(directory)) if directory.is_dir() else prefix
                bundle.add(path, arcname=name, recursive=False)
                manifest[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        for attempt_id, local in sorted(attempts.items()):
            add_tree(ROOT / "reports/forge/attempts" / attempt_id, "durable/" + attempt_id)
            add_tree(local, "attempts/" + attempt_id)
        for digest, source in sorted(sources.items()):
            snapshot = QUEUE / "snapshots" / digest
            for relative, expected in source["files"].items():
                if file_hash(snapshot / relative) != expected:
                    raise ValueError("archive source bytes differ from frozen manifest")
            add_tree(snapshot, "snapshots/" + digest)
        for path in sorted(QUEUE.rglob("*")):
            if path.is_file() and "snapshots" not in path.parts and not set(attempts).intersection(path.parts):
                add_tree(path, "queue/" + str(path.relative_to(QUEUE)))
    # Independently verify every archived file, not merely the tarball digest.
    import hashlib
    with tarfile.open(temporary, "r:gz") as bundle:
        checked = {}
        for member in bundle:
            if member.isfile():
                contents = bundle.extractfile(member).read()
                checked[member.name] = {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}
        if checked != manifest:
            raise ValueError("byte-exact archive verification failed")
    temporary.replace(destination)
    staging.cleanup()
    atomic_json(REPORT / "archive.json", {"schema_version": 1, "scope": "exact_original_execution_artifacts",
        "archive": {"path": str(destination), "sha256": file_hash(destination), "bytes": destination.stat().st_size},
        "attempt_ids": sorted(attempts), "source_digests": sorted(sources),
        "executed_commits": sorted({entry["request"]["source"]["origin_commit"]
            for entry in state["submissions"].values()}),
        "members": manifest, "verification": "all original member hashes independently verified",
        "qualification_input": False})
    emit("repair_archived", path=str(destination), attempts=len(attempts), bytes=destination.stat().st_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "media", "archive"))
    parser.add_argument("--archive-path", default="/mnt/ml7tb/experiments/ParticleGAN/bcap-tier1-repair-v1/artifacts.tar.gz")
    args = parser.parse_args()
    if args.stage == "report":
        summarize()
    elif args.stage == "media":
        media()
    else:
        archive(args.archive_path)

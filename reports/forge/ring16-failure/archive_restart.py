"""Archive this completed diagnostic byte-exactly; no scientific execution."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("existing_archiver", HERE.parent / "gaussian-smoke-inventory/archive_final.py")
archive_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_tools)


def archive():
    raw = ROOT / "runs/api/ring16-restart-diagnostic-v1"
    logs = ROOT / "runs/reports/ring16-failure"
    destination = ROOT / "artifacts/ring16-restart-diagnostic-v1.tar.gz"
    relative = archive_tools.ignored_destination(destination)
    files = sorted([p for p in raw.rglob("*") if p.is_file()] + list(logs.glob("restart-*.log")) +
                   [HERE / n for n in ("protocol.json", "restart-interruption.json", "reproduction-continuation.json")])
    if any(p.is_symlink() for p in files):
        raise ValueError("Archive inputs must be regular original files")
    inventory = {"schema_version": 1, "campaign": "ring16-restart-diagnostic-v1",
        "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": archive_tools.digest(p)} for p in files}}
    with destination.open("xb") as output:
        with archive_tools.gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
            with archive_tools.tarfile.open(fileobj=compressed, mode="w", format=archive_tools.tarfile.PAX_FORMAT) as tar:
                for name, expected in inventory["files"].items():
                    member = archive_tools.tarfile.TarInfo(name)
                    member.size, member.mode = expected["bytes"], 0o644
                    with (ROOT / name).open("rb") as handle:
                        tar.addfile(member, handle)
                payload = archive_tools.encoded(inventory)
                member = archive_tools.tarfile.TarInfo("archive-members.json")
                member.size, member.mode = len(payload), 0o644
                tar.addfile(member, archive_tools.io.BytesIO(payload))
    if archive_tools.verify(destination) != inventory:
        raise ValueError("Archived original byte verification failed")
    if any(archive_tools.digest(ROOT / name) != expected["sha256"] for name, expected in inventory["files"].items()):
        raise ValueError("Original bytes changed during archival")
    metrics = json.loads((HERE / "reproduction-results.json").read_text())
    receipt = {"schema_version": 1, "campaign": inventory["campaign"], "archive": relative,
        "sha256": archive_tools.digest(destination), "bytes": destination.stat().st_size,
        "original_files": len(files), "original_file_bytes": sum(p["bytes"] for p in inventory["files"].values()),
        "members_digest": hashlib.sha256(archive_tools.encoded(inventory)).hexdigest(), "byte_exact_verified": True,
        "source_commits": metrics["source_commits"], "cost": metrics["cost"], "qualification_input": False,
        "preserved_log_hashes": {p.relative_to(ROOT).as_posix(): archive_tools.digest(p) for p in logs.glob("restart-*.log")},
        "interruption_receipt_sha256": archive_tools.digest(HERE / "restart-interruption.json")}
    with (HERE / "archive-reproduction.json").open("x") as output:
        output.write(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"event": "archive_complete", "files": len(files), "bytes": receipt["bytes"], "sha256": receipt["sha256"]}))


if __name__ == "__main__":
    archive()

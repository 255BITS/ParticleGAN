"""Archive completed warm/hold continuation evidence without running training.

Every source byte is retained in a deterministic gzip member (mtime=0). The
manifest records both the source and archived hashes and byte counts. A
completed summary and matching CONTINUATION_DONE log are required for each
phase; a partial run cannot become a published evidence archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def completed_phase(directory: Path, log: Path, phase: str) -> tuple[dict, dict, list[Path]]:
    declaration = read_json(directory / "declaration.json")
    summary = read_json(directory / "summary.json")
    fork_summary = read_json(directory / "forks" / "summary.json")
    if declaration.get("phase") != phase or summary.get("phase") != phase:
        raise ValueError(f"phase mismatch in {directory}")
    if declaration.get("method") != summary.get("method"):
        raise ValueError(f"method mismatch in {directory}")
    if declaration.get("source") != summary.get("source"):
        raise ValueError(f"source receipt mismatch in {directory}")
    if summary.get("variants") != fork_summary.get("variants"):
        raise ValueError(f"fork summary mismatch in {directory}")
    if summary.get("cold_final_state_sha256") != fork_summary.get("cold_final_state_sha256"):
        raise ValueError(f"cold control mismatch in {directory}")
    for variant in summary["variants"]:
        receipt = read_json(directory / "forks" / f"{variant}.json")
        if receipt.get("variant") != variant:
            raise ValueError(f"wrong variant receipt: {variant}")
        if receipt.get("status") != summary["variants"][variant].get("status"):
            raise ValueError(f"status mismatch in {variant} receipt")
    read_json(directory / "forks" / "cold.json")
    for source_path, digest in declaration["source"].items():
        archived_source = directory / "source" / source_path
        if sha256(archived_source.read_bytes()) != digest:
            raise ValueError(f"source hash mismatch: {archived_source}")
    observer = directory / "source" / "generated_warm_observer.py"
    if sha256(observer.read_bytes()) != declaration["generated_observer_sha256"]:
        raise ValueError(f"generated observer hash mismatch: {observer}")
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if not events or events[-1].get("event") != "CONTINUATION_DONE":
        raise ValueError(f"no terminal completion event: {log}")
    if events[-1] != summary:
        # The log line has an event key in addition to the summary payload.
        logged_summary = dict(events[-1])
        logged_summary.pop("event")
        if logged_summary != summary:
            raise ValueError(f"completed log differs from summary: {log}")
    if any(event.get("event") in {"ERROR", "EXCEPTION"} for event in events):
        raise ValueError(f"error event in completed log: {log}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError(f"symlinks are not archive inputs: {directory}")
    if any(path.name.endswith((".tmp", ".partial")) for path in files):
        raise ValueError(f"unfinished file in {directory}")
    return declaration, summary, files


def archive_file(source: Path, destination: Path, relative: str) -> dict:
    raw = source.read_bytes()
    packed = gzip.compress(raw, mtime=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(packed)
    if gzip.decompress(destination.read_bytes()) != raw:
        raise AssertionError(f"gzip roundtrip failed: {relative}")
    # Re-read after compression to catch a source file modified during copy.
    if source.read_bytes() != raw:
        raise ValueError(f"source changed during archive: {source}")
    return {
        "path": relative + ".gz",
        "raw_sha256": sha256(raw),
        "raw_bytes": len(raw),
        "compressed_sha256": sha256(packed),
        "compressed_bytes": len(packed),
    }


def archive(warm: Path, hold: Path, warm_log: Path, hold_log: Path, output: Path) -> dict:
    phases = {"warm": (warm, warm_log), "hold": (hold, hold_log)}
    checked = {name: completed_phase(*args, name) for name, args in phases.items()}
    warm_decl, warm_summary, _ = checked["warm"]
    hold_decl, hold_summary, _ = checked["hold"]
    for key in ("method", "source", "nominal_rates", "seed", "noise_horizon"):
        if warm_decl.get(key) != hold_decl.get(key):
            raise ValueError(f"warm/hold declaration mismatch: {key}")
    if warm_summary.get("warm_state_sha256") != hold_summary.get("warm_state_sha256"):
        raise ValueError("warm/hold fork states differ")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        entries = []
        phase_receipts = {}
        for phase, (directory, log) in phases.items():
            declaration, summary, files = checked[phase]
            for source in files:
                relative = f"{phase}/{source.relative_to(directory).as_posix()}"
                entries.append(archive_file(source, staging / (relative + ".gz"), relative))
            relative = f"logs/{phase}.log"
            entries.append(archive_file(log, staging / (relative + ".gz"), relative))
            phase_receipts[phase] = {
                "phase": declaration["phase"],
                "summary_sha256": sha256((directory / "summary.json").read_bytes()),
                "steps": summary["steps"],
                "variants": {
                    variant: {
                        "status": row["status"],
                        "seconds": read_json(directory / "forks" / f"{variant}.json")["seconds"],
                    }
                    for variant, row in summary["variants"].items()
                },
            }
        manifest = {
            "schema": "continuation-evidence-archive-v1",
            "archiver_sha256": sha256(Path(__file__).read_bytes()),
            "method": warm_decl["method"],
            "scope": "completed warm and same-target long-hold continuation; no cold acquisition claim",
            "source_paths": {phase: str(directory) for phase, (directory, _) in phases.items()},
            "log_paths": {phase: str(log) for phase, (_, log) in phases.items()},
            "phase_receipts": phase_receipts,
            "files": sorted(entries, key=lambda row: row["path"]),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # The published directory appears only after every member is validated.
        os.rename(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--hold", type=Path, required=True)
    parser.add_argument("--warm-log", type=Path, required=True)
    parser.add_argument("--hold-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = archive(args.warm, args.hold, args.warm_log, args.hold_log, args.output)
    print(json.dumps({"output": str(args.output), "files": len(manifest["files"]),
                      "phase_receipts": manifest["phase_receipts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

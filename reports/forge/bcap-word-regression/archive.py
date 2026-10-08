"""Archive the concluded factorial's exact local bytes and executed source."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/forge/bcap-word-regression")
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/bcap-word-regression-factorial-v1.tar.gz")
    args = parser.parse_args()
    source = None
    for arm in ("truncated-serial", "full-serial", "truncated-threaded", "full-threaded"):
        receipt = read_json(args.runs / arm / "compact-receipt.json")
        if receipt["cost"]["completed_steps"] != 20001 or receipt["qualification_input"]:
            raise ValueError("all frozen arms must have concluded full-budget diagnostics")
        manifest = read_json(args.runs / arm / "source-manifest.json")
        if source is None:
            source = manifest
        elif manifest != source:
            raise ValueError("factorial mixed execution sources")
    members = {"raw/" + str(path.relative_to(args.runs)): path
               for path in args.runs.rglob("*") if path.is_file() and path.suffix != ".lock"}
    for relative, expected in source["files"].items():
        path = ROOT / relative
        if file_hash(path) != expected:
            raise ValueError(f"executed source changed: {relative}")
        members["execution-source/" + relative] = path
    # Full task JSON is also embedded in each raw request. Include its public card.
    task = ROOT / "configs/forge/tasks/five_word_joint_acquisition.json"
    members["execution-source/configs/forge/tasks/five_word_joint_acquisition.json"] = task
    expected = {name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
                for name, path in sorted(members.items())}
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "x:gz") as tar:
        for name, path in sorted(members.items()):
            tar.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(args.archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.name in actual:
                raise ValueError("unexpected archive member")
            digest = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual[member.name] = {"sha256": digest.hexdigest(), "bytes": member.size}
    if actual != expected:
        raise ValueError("archive bytes differ from originals")
    receipt = {"schema_version": 1, "scope": "exact_local_raw_evidence_and_execution_source",
               "archive": str(args.archive.relative_to(ROOT)), "sha256": file_hash(args.archive),
               "bytes": args.archive.stat().st_size, "member_count": len(actual),
               "member_manifest_sha256": stable_hash(actual), "source_commit": source["origin_commit"],
               "source_digest": source["digest"], "verification": "Every member reopened and hashed against original bytes",
               "bulk_artifacts_tracked": False, "qualification_input": False}
    atomic_json(Path(__file__).with_name("archive.json"), receipt)
    atomic_json(args.runs / "archive-member-manifest.json", actual)
    print(receipt)


if __name__ == "__main__":
    main()

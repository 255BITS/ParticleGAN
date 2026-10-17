"""Archive completed frozen receipts, queue logs and byte-exact source once."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import file_hash, read_json, stable_hash


def archive(root, snapshot, queue, output):
    report = read_json(snapshot)
    copy = dict(report)
    claimed = copy["provenance"].get("input_digest")
    copy["provenance"] = {k: v for k, v in copy["provenance"].items() if k != "input_digest"}
    if claimed != stable_hash(copy):
        raise ValueError("frozen publication digest differs")
    commit = report["frozen_source"]["commit"]
    attempts = sorted({a for row in report["rows"] for a in row.get("attempt_ids", [])})
    receipts, sources = {}, {}
    for attempt in attempts:
        directory = root / "reports/forge/attempts" / attempt
        request, evidence, result = (read_json(directory / (name + ".json")) for name in ("request", "evidence", "result"))
        actual = request.get("request", request)
        source = actual["source"]
        if (source["origin_commit"] != commit or source != evidence["source"]
                or actual["runtime"] != evidence["runtime"] or stable_hash(result) != evidence["result_hash"]
                or stable_hash(source["files"]) != source["digest"]):
            raise ValueError("original scientific receipt certificate differs")
        for name, sha in source["files"].items():
            if name in sources and sources[name] != sha:
                raise ValueError("conflicting byte-exact source files")
            sources[name] = sha
        for name in ("request", "evidence", "result"):
            path = directory / (name + ".json")
            receipts[path.relative_to(root).as_posix()] = file_hash(path)
    payloads = {}
    for name, sha in sources.items():
        content = subprocess.check_output(["git", "show", commit + ":" + name], cwd=root)
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError("Git cannot restore exact executed source bytes")
        payloads["frozen-scientific-source/" + name] = content
    manifest = {"schema_version": 1, "executed_commit": commit, "source_files_sha256": sources,
                "original_receipts_sha256": receipts, "attempts": len(attempts),
                "qualification_input": "original receipts regraded by the byte-exact source; compact summaries supply none",
                "snapshot_sha256": file_hash(snapshot)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as bundle:
        for name in sorted(receipts):
            bundle.add(root / name, arcname=name, recursive=False)
        bundle.add(queue, arcname="queue")
        for name, content in sorted(payloads.items()):
            entry = tarfile.TarInfo(name); entry.size = len(content); entry.mode = 0o644
            bundle.addfile(entry, io.BytesIO(content))
        content = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        entry = tarfile.TarInfo("restore-manifest.json"); entry.size = len(content)
        bundle.addfile(entry, io.BytesIO(content))
    with tarfile.open(output) as bundle:
        for name, expected in {**receipts, **{"frozen-scientific-source/" + k: v for k, v in sources.items()}}.items():
            if hashlib.sha256(bundle.extractfile(name).read()).hexdigest() != expected:
                raise ValueError("archived original bytes differ")
    return {"schema_version": 1, "kind": "completed-frozen-scientific-round-archive",
            "archive": {"path": str(output.resolve()), "sha256": file_hash(output), "bytes": output.stat().st_size},
            "manifest": manifest,
            "restoration": ["Fetch executed_commit and create an isolated checkout of it.",
                            "Restore reports/forge/attempts from this archive into that checkout.",
                            "Verify every original receipt and source hash against restore-manifest.json.",
                            "Use regenerate_technique_inventory.py --source-commit with that executed commit; no training is launched."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    args = parser.parse_args()
    result = archive(ROOT, args.snapshot, args.queue, args.output)
    args.card.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archive_bytes": result["archive"]["bytes"], "attempts": result["manifest"]["attempts"],
                      "source_files": len(result["manifest"]["source_files_sha256"])}))


if __name__ == "__main__":
    main()

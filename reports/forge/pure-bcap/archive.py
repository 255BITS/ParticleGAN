"""Archive byte-exact execution envelopes, raw logs and saved artifacts."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash

REPORT = Path("reports/forge/pure-bcap")


def retained_file(path):
    # Interpreter caches created by read-only regrading are not execution
    # evidence and need not be identical across restored checkout roots.
    return path.is_file() and not {"__pycache__", ".pytest_cache"} & set(path.parts)


def _retained_files(root, result):
    """Include all paid context and bind relocated original absolute paths."""
    rounds = result.get("execution_rounds") or [{"round": result["round"]}]
    attempts = sorted(record["attempt_id"] for record in result.get("paid_attempts", []))
    if not attempts:
        attempts = sorted({task["attempt_id"] for candidate in result["candidates"] for task in candidate["tasks"]}
                          | set(result.get("original_unselected_paid_attempt_ids", [])))
    if len(attempts) != len(set(attempts)) or len(attempts) != result["unique_paid_attempts"]:
        raise ValueError("Archive must retain every unique paid attempt, including original repair context")
    files, restoration = {}, {}
    def include(directory):
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Execution evidence directory is missing: {directory}")
        for path in directory.rglob("*"):
            if retained_file(path):
                files[path.relative_to(root).as_posix()] = path
    for cohort in rounds:
        include(root / "runs/forge" / cohort["round"])
    for attempt in attempts:
        directory = root / "reports/forge/attempts" / attempt
        include(directory)
        envelope, certificate = read_json(directory / "request.json"), read_json(directory / "evidence.json")
        request = envelope.get("request", envelope)
        source = request["source"]
        paths = [(certificate["local_artifact_root"],
                  root / "runs/forge" / request["campaign_id"] / "queue" / request["campaign_id"] / attempt),
                 (source["snapshot_path"],
                  root / "runs/forge" / request["campaign_id"] / "queue/snapshots" / source["digest"])]
        for original, relocated in paths:
            original = Path(original)
            retained = original if original.is_relative_to(root) else relocated
            if not retained.is_dir():
                raise ValueError(f"Restore the recorded artifact directory before archiving: {original}")
            include(retained)
            # A copied checkout keeps envelope bytes unchanged. Verify its raw
            # artifacts before recording the exact original restoration root.
            if original != retained and original.is_dir():
                original_files = {path.relative_to(original).as_posix(): file_hash(path)
                                  for path in original.rglob("*") if retained_file(path)}
                retained_files = {path.relative_to(retained).as_posix(): file_hash(path)
                                  for path in retained.rglob("*") if retained_file(path)}
                if original_files != retained_files:
                    raise ValueError(f"Relocated execution evidence differs from the original bytes: {original}")
            restoration[str(original)] = retained.relative_to(root).as_posix()
    return attempts, files, restoration


def archive_readout(root=ROOT, *, readout_path=REPORT / "readout.json", archive=None):
    root = Path(root).resolve()
    readout_path = Path(readout_path)
    result = read_json(root / readout_path)
    content = dict(result)
    if content.pop("input_digest", None) != stable_hash(content):
        raise ValueError("Archive readout digest differs")
    archive = Path(archive) if archive else root / "runs/archives/pure-bcap" / (readout_path.stem + ".tar.gz")
    if not archive.is_absolute():
        archive = root / archive
    attempts, files, restoration = _retained_files(root, result)
    files[readout_path.as_posix()] = root / readout_path
    for path in (result.get("plan"), "reports/forge/pure-bcap/audit-initial.json",
                 "reports/forge/pure-bcap/restoration.json"):
        if path and (root / path).is_file():
            files[path] = root / path
    if archive in files.values():
        raise ValueError("Artifact bundle cannot contain itself")
    inputs = {name: file_hash(path) for name, path in sorted(files.items())}
    cohorts = result.get("source_cohorts") or [{"source_commit": result["source_commit"],
        "source_digest": result["source_digest"]}]
    manifest = {"schema_version": 2, "readout": readout_path.as_posix(),
        "readout_sha256": file_hash(root / readout_path), "scope": result["scope"], "source_cohorts": cohorts,
        "attempt_ids": attempts, "files": inputs, "original_absolute_roots": restoration,
        "excluded_generated_caches": ["__pycache__", ".pytest_cache"]}
    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise ValueError("Archive destination already exists; do not overwrite execution evidence")
    with tarfile.open(archive, "w:gz") as bundle:
        for name in inputs:
            bundle.add(files[name], arcname=name, recursive=False)
        info = tarfile.TarInfo("bundle-manifest.json")
        info.size = len(encoded)
        bundle.addfile(info, io.BytesIO(encoded))
    with tarfile.open(archive, "r:gz") as bundle:
        if bundle.extractfile("bundle-manifest.json").read() != encoded:
            raise ValueError("Archived manifest differs")
        for name, expected in inputs.items():
            digest = hashlib.sha256()
            with bundle.extractfile(name) as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Archived execution evidence changed while copying: {name}")
    receipt = {"schema_version": 2, "round": result["round"], "scope": result["scope"],
        "readout": readout_path.as_posix(), "readout_sha256": file_hash(root / readout_path),
        "archive": str(archive), "sha256": file_hash(archive), "bytes": archive.stat().st_size,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(), "input_manifest_digest": stable_hash(inputs),
        "files_verified": len(inputs), "unique_attempts": len(attempts), "source_cohorts": cohorts,
        "qualification_input": False, "original_absolute_roots": restoration,
        "restoration": "Extract relative archive members under a checkout, then restore directories to the original_absolute_roots recorded in bundle-manifest.json when resolving unchanged absolute envelope links. Every file is SHA-256 verified. The archive retains bulk logs, queue state, frozen source, checkpoints, scored samples and original certified envelopes; no training or regrading is needed to read compact publications."}
    output = REPORT / ("archive-initial.json" if result["scope"] == "partial_pure_bcap_initial_readout" else "archive.json")
    atomic_json(root / output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--readout", type=Path, default=REPORT / "readout.json")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    receipt = archive_readout(args.root, readout_path=args.readout, archive=args.archive)
    print(f"Verified {receipt['files_verified']} files and {receipt['unique_attempts']} attempts in {receipt['archive']}", flush=True)


if __name__ == "__main__":
    main()

"""Archive certified bytes and reproduce saved-evidence reports without science.

Run archive only after ordinary and diagnostic work is terminal. Reproduce
uses a fresh source-compatible checkout, final audit/publication tooling, and
the existing raw artifacts at their recorded paths. No receipts are rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCE_COMMIT = "e1c7fcc22086c290868ea47b989a998a86e913e4"
SOURCE_DIGEST = "c68be4ae40db26959c1b2876ed33ec044aba9cf08cab84be71ed8a0264a2eb17"
SOURCE_FILES = 1205
ACTIVE = {"queued", "running", "paused"}
RECORDS = ("request.json", "result.json", "evidence.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read(path):
    return json.loads(path.read_bytes())


def identity(data):
    return {"sha256": sha(data), "bytes": len(data)}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def record(path, value):
    write(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def archive(repository, archive_root):
    destination = archive_root / "certificates"
    if destination.exists():
        raise ValueError("Preserve the existing certificate archive; use a fresh destination")
    inputs, attempts, sources = {}, set(), []
    for scope, queue, progress in (("ordinary", "queue", "progress.json"),
                                   ("research_diagnostic", "diagnostic-queue", "diagnostic-progress.json")):
        state_path = archive_root / queue / "queue/state.json"
        progress_path = archive_root / progress
        state_bytes, progress_bytes = state_path.read_bytes(), progress_path.read_bytes()
        state, progress_data = json.loads(state_bytes), json.loads(progress_bytes)
        request_ids = set(progress_data["requests"].values())
        submissions = [state["submissions"][rid] for rid in request_ids]
        jobs = [job for job in state["jobs"].values() if request_ids & set(job.get("subscribers", []))]
        if any(item["status"] in ACTIVE for item in [*submissions, *jobs]):
            raise ValueError(f"{scope}: paid work remains active; do not archive partial certificates")
        for submission in submissions:
            source = submission["request"]["source"]
            if len(source["files"]) != SOURCE_FILES or source["digest"] != SOURCE_DIGEST or canonical_hash(source["files"]) != SOURCE_DIGEST:
                raise ValueError(f"{scope}: source does not match the paid 1205-file science identity")
            sources.append(source)
        for job in jobs:
            attempts.update(attempt["attempt_id"] for attempt in job.get("attempts", []))
            if job.get("result"):
                attempts.add(job["result"]["attempt_id"])
        inputs[f"metadata/{scope}/queue/queue/state.json"] = (state_path, state_bytes)
        inputs[f"metadata/{scope}/progress.json"] = (progress_path, progress_bytes)
    protected = archive_root / "qualification-before.json"
    if not protected.is_file():
        raise ValueError("Original qualification-before.json is required for preservation verification")
    inputs["metadata/qualification-before.json"] = (protected, protected.read_bytes())
    for aid in sorted(attempts):
        if not aid or Path(aid).name != aid or aid in {".", ".."}:
            raise ValueError("Invalid certified attempt identity")
        for name in RECORDS:
            path = repository / "reports/forge/attempts" / aid / name
            inputs[f"attempts/{aid}/{name}"] = (path, path.read_bytes())
    # All prerequisites are present before creating the immutable archive.
    for relative, (_, data) in inputs.items():
        write(destination / relative, data)
    if any(path.read_bytes() != data for path, data in inputs.values()):
        raise ValueError("Original queue, progress or certificate bytes changed during archival")
    manifest = {"schema_version": 1, "source_compatible_commit": SOURCE_COMMIT,
                "scientific_source_digest": SOURCE_DIGEST, "scientific_source_files": SOURCE_FILES,
                "source_origins": sorted({source.get("origin_commit", "") for source in sources}),
                "certified_attempts": len(attempts), "files": {relative: identity(data) for relative, (_, data) in sorted(inputs.items())},
                "original_paths": {relative: str(path) for relative, (path, _) in sorted(inputs.items())},
                "optimizer_updates_added": 0, "sampling_draws_added": 0, "scores_recomputed": 0,
                "raw_artifacts": "Original absolute artifact and frozen-snapshot paths remain unchanged and must be available.",
                "reproduction_source_sha256": sha(Path(__file__).read_bytes())}
    record(destination / "manifest.json", manifest)
    print(json.dumps({"event": "certificates_archived", "attempts": len(attempts), "manifest": str(destination / "manifest.json")}), flush=True)


def reproduce(repository, archive_root, checkout, output):
    certificates = archive_root / "certificates"
    manifest_path = certificates / "manifest.json"
    manifest = read(manifest_path)
    if manifest["source_compatible_commit"] != SOURCE_COMMIT or manifest["scientific_source_digest"] != SOURCE_DIGEST:
        raise ValueError("Archived source identity differs from the declared paid experiment")
    for relative, expected in manifest["files"].items():
        if identity((certificates / relative).read_bytes()) != expected:
            raise ValueError(f"Archived certificate/metadata bytes changed: {relative}")
    if checkout.exists() or output.exists():
        raise ValueError("Use a fresh source checkout and fresh reproduction output")
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(checkout), SOURCE_COMMIT], cwd=repository, check=True)
    ordinary = certificates / "metadata/ordinary"
    diagnostic = certificates / "metadata/research_diagnostic"
    source = next(iter(read(ordinary / "queue/queue/state.json")["submissions"].values()))["request"]["source"]
    if len(source["files"]) != SOURCE_FILES or canonical_hash(source["files"]) != SOURCE_DIGEST:
        raise ValueError("Archived request source inventory differs")
    for relative, expected in source["files"].items():
        if sha((checkout / relative).read_bytes()) != expected:
            raise ValueError(f"Source-compatible checkout does not match paid scientific bytes: {relative}")
    for relative in manifest["files"]:
        if not relative.startswith("attempts/"):
            continue
        restored = Path("reports/forge") / relative
        ignored = subprocess.run(["git", "check-ignore", "--quiet", str(restored)], cwd=checkout)
        if ignored.returncode:
            raise ValueError(f"Restored certificate must remain ignored: {restored}")
        write(checkout / restored, (certificates / relative).read_bytes())
    output.mkdir(parents=True)
    tooling = repository / "reports/forge/bcap-develop-integration"
    shared = ["--repository", str(checkout), "--queue", str(ordinary / "queue"),
              "--progress", str(ordinary / "progress.json"), "--diagnostic-queue", str(diagnostic / "queue"),
              "--diagnostic-progress", str(diagnostic / "progress.json"), "--output", str(output)]
    environment = dict(os.environ, PYTHONPATH=str(repository), CUDA_VISIBLE_DEVICES="",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for name in ("audit.py", "publish.py"):
        command = [sys.executable, str(tooling / name), *shared]
        if name == "audit.py":
            command += ["--protected", str(certificates / "metadata/qualification-before.json")]
        log = output / f"{Path(name).stem}.log"
        print(f"tail -F {log}", flush=True)
        with log.open("w") as stdout:
            process = subprocess.run(command, cwd=checkout, env=environment, stdout=stdout, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"Saved-evidence {name} failed; inspect {log}. No scientific rerun is authorized.")
    for relative, expected in source["files"].items():
        if sha((checkout / relative).read_bytes()) != expected:
            raise ValueError(f"Reproduction changed scientific bytes: {relative}")
    for relative, expected in manifest["files"].items():
        if identity((certificates / relative).read_bytes()) != expected:
            raise ValueError(f"Reproduction changed archived metadata: {relative}")
        if relative.startswith("attempts/") and identity((checkout / "reports/forge" / relative).read_bytes()) != expected:
            raise ValueError(f"Reproduction changed restored certificate: {relative}")
    results = {name: read(output / name) for name in ("audit.json", "results.json")}
    if any(result.get(key) != 0 for result in results.values() for key in ("optimizer_updates_added", "sampling_draws_added")):
        raise ValueError("Saved-evidence tooling reported added scientific execution")
    record(output / "reproduction.json", {"schema_version": 1, "source_compatible_commit": SOURCE_COMMIT,
        "scientific_source_digest": SOURCE_DIGEST, "scientific_files_verified": SOURCE_FILES,
        "certificate_manifest_sha256": sha(manifest_path.read_bytes()), "certified_attempts": manifest["certified_attempts"],
        "tooling_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "tooling_sha256": {name: sha((tooling / name).read_bytes()) for name in ("audit.py", "publish.py")},
        "output_sha256": {name: sha((output / name).read_bytes()) for name in results},
        "optimizer_updates_added": 0, "sampling_draws_added": 0, "scores_recomputed": 0,
        "qualification_input": False, "source_checkout": str(checkout)})
    print(json.dumps({"event": "saved_evidence_reproduced", "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("archive", "reproduce"))
    parser.add_argument("--repository", type=Path, required=True, help="Final tooling and original ignored certificates")
    parser.add_argument("--archive", type=Path, required=True, help="Original outside-Git experiment archive A")
    parser.add_argument("--checkout", type=Path, help="Fresh artifact-drive checkout for reproduce")
    parser.add_argument("--output", type=Path, help="Fresh outside-Git reproduction output")
    args = parser.parse_args()
    repository, archive_root = args.repository.resolve(), args.archive.resolve()
    if archive_root.is_relative_to(repository):
        raise ValueError("Raw certificates must stay outside the source repository")
    if args.mode == "archive":
        archive(repository, archive_root)
    else:
        if args.checkout is None or args.output is None:
            parser.error("reproduce requires --checkout and --output")
        checkout, output = args.checkout.resolve(), args.output.resolve()
        if checkout.is_relative_to(repository) or output.is_relative_to(repository):
            raise ValueError("Reproduction checkout and artifacts must stay outside the source repository")
        reproduce(repository, archive_root, checkout, output)


if __name__ == "__main__":
    main()

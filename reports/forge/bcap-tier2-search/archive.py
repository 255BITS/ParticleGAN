"""Archive both completed campaigns and their original receipts byte for byte."""
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

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash

OUTPUT = ROOT / "reports/forge/bcap-tier2-search"
STUDIES = ("bcap-tier2-search-v1", "bcap-overnight-search-v1")
LOCAL = (ROOT / "runs/software/bcap-tier2-search", ROOT / "runs/software/bcap-overnight-search")


def archive(destination: Path):
    if destination.exists():
        raise ValueError("Archive destinations are immutable; choose an unused path")
    readouts = [read_json(OUTPUT / "readout.json"), read_json(OUTPUT / "overnight/readout.json")]
    combined = read_json(OUTPUT / "combined-readout.json")
    assert combined["configurations"] == sum(len(r["trials"]) for r in readouts) == 96
    assert combined["selection"]["selection_complete"]
    for directory in LOCAL:
        assert (directory / "completed.json").is_file()
        assert not (directory / "failed.json").exists()
    queues = [Path(r["queue_root"]).resolve() for r in readouts]
    attempts = sorted({attempt for r in readouts for trial in r["trials"] for attempt in trial["attempt_ids"]})

    # The original launcher was already loaded when its count was refactored.
    # Include exact Git producer versions alongside the final reporting code.
    reproduction = LOCAL[0] / "reproduction"
    for origin in sorted({commit for r in readouts for commit in r["source_origin_commits"]}):
        for name in ("run.py", "run_overnight.py", "prepare.py", "prepare_overnight.py"):
            relative = "reports/forge/bcap-tier2-search/" + name
            present = subprocess.run(["git", "cat-file", "-e", origin + ":" + relative],
                                     cwd=ROOT, capture_output=True).returncode == 0
            if present:
                target = reproduction / origin / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.check_output(["git", "show", origin + ":" + relative], cwd=ROOT))

    inputs = [("workspace", ROOT, path) for path in (
        OUTPUT, *LOCAL, ROOT / "configs/forge",
        ROOT / "reports/forge/word_checkpoint_publication.py",
        ROOT / "reports/forge/regenerate_technique_inventory.py",
        ROOT / "reports/forge/bcap-convolution/publish.py",
        ROOT / "reports/forge/gaussian-smoke-inventory/export_media.py",
        ROOT / "reports/forge/technique-evidence",
        ROOT / "reports/forge/technique-inventory.md",
        ROOT / "reports/forge/technique-inventory.json",
        *(ROOT / "reports/forge/technique-receipts" / (attempt + ".json") for attempt in attempts),
        *(ROOT / "reports/forge/attempts" / attempt for attempt in attempts))]
    inputs += [("queues/" + study, queue, queue) for study, queue in zip(STUDIES, queues)]
    files = {}
    for prefix, base, source in inputs:
        if not source.exists():
            raise ValueError("Missing required archive input: " + str(source))
        for path in source.rglob("*") if source.is_dir() else [source]:
            if path.is_symlink():
                raise ValueError("Refusing a symlink archive input: " + str(path))
            if path.is_file():
                files[prefix + "/" + path.relative_to(base).as_posix()] = path
    for path in (ROOT / "reports/forge/configuration-search").glob("*.json"):
        if any(path.stem.startswith(study + "--") for study in STUDIES):
            files["workspace/" + path.relative_to(ROOT).as_posix()] = path
    inventory = {name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                 for name, path in sorted(files.items())}
    bindings = {"workspace": str(ROOT), **{"queues/" + study: str(queue) for study, queue in zip(STUDIES, queues)}}
    metadata = {"schema_version": 1, "study_ids": list(STUDIES), "restore_roots": bindings,
                "note": "Restore each prefix to its recorded root before interpreting absolute checkpoint or artifact paths.",
                "files": inventory}
    payload = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "x:gz", compresslevel=3) as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
        member = tarfile.TarInfo("archive-members.json")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with tarfile.open(destination, "r:gz") as bundle:
        assert {m.name for m in bundle.getmembers()} == set(inventory) | {"archive-members.json"}
        for name, expected in inventory.items():
            member = bundle.getmember(name)
            assert member.isfile() and member.size == expected["bytes"]
            hasher = hashlib.sha256()
            with bundle.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            assert hasher.hexdigest() == expected["sha256"] == file_hash(files[name])
    receipt = {"schema_version": 1, "study_ids": list(STUDIES), "path": str(destination),
               "bytes": destination.stat().st_size, "sha256": file_hash(destination),
               "original_files": len(inventory), "members_digest": stable_hash(inventory),
               "restore_roots": bindings, "byte_exact_verified": True,
               "original_attempts": len(attempts), "original_files_removed": False,
               "qualification_input": False}
    atomic_json(OUTPUT / "archive.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    archive(parser.parse_args().destination.resolve())

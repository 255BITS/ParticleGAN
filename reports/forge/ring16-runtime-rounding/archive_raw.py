"""Archive local originals exclusively and verify every byte without training."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from experiments.forge.contracts import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(p for base in (args.root / "runs/api/ring16-runtime-rounding-v1",
                                  args.root / "runs/reports/ring16-runtime-rounding")
                   for p in base.rglob("*") if p.is_file())
    assert files and (args.root / "runs/api/ring16-runtime-rounding-v1/campaign-summary.json").exists()
    hashes = {str(p.relative_to(args.root)): file_hash(p) for p in files}
    manifest = {"source_files": hashes, "new_model_calls": 0, "qualification_input": False}
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for p in files:
            archive.add(p, arcname=str(p.relative_to(args.root)), recursive=False)
        info = tarfile.TarInfo("archive-manifest.json")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    import hashlib
    with tarfile.open(args.output, "r:gz") as archive:
        assert len(archive.getmembers()) == len(files) + 1
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
    assert all(file_hash(args.root / name) == digest for name, digest in hashes.items())
    atomic_json(args.receipt, {"scope": "verified_byte_exact_local_archive", "qualification_input": False,
        "archive_path": str(args.output.resolve()), "sha256": file_hash(args.output),
        "bytes": args.output.stat().st_size, "original_files": len(files), "archive_members": len(files) + 1,
        "every_member_hash_verified": True, "originals_unchanged": True, "new_model_calls": 0,
        "contains": "All four completed CUDA arm artifacts and campaign summary; local diagnostic/report logs existing at archive creation. Later publication logs remain separate.",
        "restore": "Extract into a new directory; archive paths preserve runs/api and runs/reports. Verify every file with archive-manifest.json before analysis."})
    print(json.dumps({"archive": str(args.output), "sha256": file_hash(args.output), "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()

"""Package completed solvability evidence, preserving original JSON bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def archive(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    files = [p for p in sorted(source.rglob("*")) if p.is_file()
             and "__pycache__" not in p.parts and ".git" not in p.parts]
    # Some handoffs already include compressed duplicates of the raw JSON.
    files = [p for p in files if not (p.name.endswith(".json.gz") and p.with_suffix("").exists())]
    compressed = {str(p.relative_to(source)) for p in files if p.suffix == ".json"}
    inventory = []
    for path in files:
        relative = str(path.relative_to(source))
        raw = path.read_bytes()
        target_name = relative + ".gz" if relative in compressed else relative
        target = destination / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(raw, mtime=0) if relative in compressed else raw
        if relative in compressed:
            assert gzip.decompress(data) == raw
        if path.suffix == ".md":
            for name in compressed:
                # Resolve Markdown links relative to their containing file.
                import os
                link = os.path.relpath(source / name, path.parent).encode()
                data = data.replace(b"](" + link + b")", b"](" + link + b".gz)")
        target.write_bytes(data)
        inventory.append(dict(path=target_name, original_path=relative,
                              original_sha256=hashlib.sha256(raw).hexdigest(),
                              archived_sha256=hashlib.sha256(data).hexdigest(),
                              gzip_added=relative in compressed,
                              markdown_links_adjusted=path.suffix == ".md" and data != raw))
    (destination / "archive_manifest.json").write_text(json.dumps(dict(files=inventory), indent=2) + "\n")
    print(f"Archived {len(inventory)} files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive(args.source, args.destination)

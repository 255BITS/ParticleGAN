"""Archive a completed local study without changing experiment JSON bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def archive(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    if not (source / "frozen.json").exists():
        raise ValueError("this archive command expects a completed frozen study")
    report = json.loads((source / "results.json").read_bytes())
    expected = sum(task["split"] == "reserved" for task in report["manifest"]["tasks"])
    if len(report.get("transfer", [])) != 3 or any(len(row["results"]) != expected for row in report["transfer"]):
        raise ValueError("reserved comparisons are incomplete")
    destination.mkdir(parents=True)
    inventory = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        raw = path.read_bytes()
        compress = path.suffix == ".json" and path.name not in ("manifest.json", "protocol.json", "frozen.json")
        target = destination / (str(relative) + ".gz" if compress else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(raw, mtime=0) if compress else raw
        if path.name == "README.md":
            data = raw.replace(b"](results.json)", b"](results.json.gz)")
        target.write_bytes(data)
        if compress:
            assert gzip.decompress(data) == raw
        inventory.append(dict(path=str(target.relative_to(destination)), original_path=str(relative),
                              original_sha256=hashlib.sha256(raw).hexdigest(),
                              archived_sha256=hashlib.sha256(data).hexdigest(), original_bytes=len(raw),
                              gzip_added=compress, markdown_links_adjusted=path.name == "README.md"))
    (destination / "policy.json").write_text(json.dumps(report["frozen"]["policy"], indent=2) + "\n")
    (destination / "archive_manifest.json").write_text(json.dumps(dict(files=inventory), indent=2) + "\n")
    print(f"Archived {len(inventory)} files in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive(args.source, args.destination)

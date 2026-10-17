"""Archive completed raw evidence and exact executed sources; no training.

Keep the output outside Git. The external card binds bytes without including
itself, or an audit proof that depends on those bytes, in the archive.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/forge/word-root-cause"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Archive output must be new; never overwrite an evidence identity")
    receipts = [json.loads(path.read_text()) for path in sorted((REPORT / "receipts").glob("*.json"))]
    manifests = {}
    for receipt in receipts:
        directory = Path(receipt["raw_directory"])
        manifest = json.loads((directory / "source-manifest.json").read_text())
        previous = manifests.setdefault(receipt["source_commit"], {"files": {}, "digests": set()})
        previous["digests"].add(manifest["digest"])
        for path, digest in manifest["files"].items():
            assert previous["files"].setdefault(path, digest) == digest
        for name, expected in receipt["raw_artifacts"].items():
            data = (directory / name).read_bytes()
            assert {"bytes": len(data), "sha256": sha(data)} == expected
    sources, trees = {}, {}
    for commit, manifest in manifests.items():
        entries = subprocess.check_output(["git", "ls-tree", "-rz", commit], cwd=ROOT)
        tree = {path.decode(): metadata.split()[2].decode()
                for entry in entries.split(b"\0") if entry
                for metadata, path in (entry.split(b"\t", 1),)}
        trees[commit] = tree
        needed = sorted({tree[path] for path in manifest["files"]} - sources.keys())
        process = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT,
                                 input=("\n".join(needed) + "\n").encode(),
                                 stdout=subprocess.PIPE, check=True)
        cursor = 0
        for oid in needed:
            end = process.stdout.index(b"\n", cursor)
            actual, kind, size = process.stdout[cursor:end].split()
            assert actual.decode() == oid and kind == b"blob"
            start, length = end + 1, int(size)
            sources[oid] = process.stdout[start:start + length]
            cursor = start + length + 1
        assert all(sha(sources[tree[path]]) == digest for path, digest in manifest["files"].items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for round_number in (1, 2, 3)
                   for path in (ROOT / f"runs/forge/word-root-cause-round{round_number}").rglob("*")
                   if path.is_file() and path.name != "archive.log")
    analysis = sorted(path for path in REPORT.glob("*")
                      if path.is_file() and path.suffix in (".py", ".json")
                      and path.name not in ("archive.json", "publication-audit.json"))
    inventory = []
    with args.output.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for path in files + analysis:
                data = path.read_bytes()
                name = str(path.relative_to(ROOT))
                inventory.append({"path": name, "bytes": len(data), "sha256": sha(data)})
                add_bytes(archive, name, data)
            for commit, manifest in sorted(manifests.items()):
                for path, digest in sorted(manifest["files"].items()):
                    data = sources[trees[commit][path]]
                    name = f"sources/{commit}/{path}"
                    inventory.append({"path": name, "bytes": len(data), "sha256": digest})
                    add_bytes(archive, name, data)
            add_bytes(archive, "inventory.json", (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode())
    digest = hashlib.sha256()
    with args.output.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    card = {"schema_version": 1, "id": "word-root-cause-18-arms-v1",
            "scope": "Raw task-only diagnostic evidence; no qualification input or reuse",
            "qualification_input": False, "qualification_reuse": False,
            "archive": {"path": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                        "sha256": digest.hexdigest(), "format": "tar.gz"},
            "primary_locator": "Main repository artifacts/forge/word-root-cause-18-arms-v1.tar.gz; ignored local artifact, not a remote-hosted download.",
            "raw_runs": 18, "members_in_inventory": len(inventory),
            "source_snapshots": [{"commit": commit, "request_source_digests": sorted(manifest["digests"]),
                                   "directory": f"sources/{commit}", "files_union": len(manifest["files"])}
                                  for commit, manifest in sorted(manifests.items())],
            "restore": "Extract into a new directory. Raw runs retain their runs/forge/word-root-cause-roundN layout. sources/<commit> contains every file in that executed source manifest; restore into a clean checkout of the matching commit for scientific replay. Compare inventory hashes and original source-manifest.json before execution. Existing raw absolute locators may be replaced in an external read-only audit adapter, never in frozen receipts.",
            "raw_directories_retained": sorted({str(Path(row["raw_directory"]).parent) for row in receipts}),
            "archive_builder": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(Path(__file__).read_bytes())},
            "audit": {"source": "reports/forge/word-root-cause/publication_audit.py",
                      "proof": "reports/forge/word-root-cause/publication-audit.json",
                      "binding": "External proof binds archive SHA and script SHA; it is not included in this archive."}}
    (REPORT / "archive.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    print(json.dumps(card["archive"]), flush=True)


if __name__ == "__main__":
    main()

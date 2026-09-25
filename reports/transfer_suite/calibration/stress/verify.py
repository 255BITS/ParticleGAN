"""Verify packaged bytes and every source member without external dependencies."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / "inventory.json").read_bytes())
    json_files = source_files = archives = 0
    for entry in inventory["files"]:
        data = (root / entry["path"]).read_bytes()
        assert len(data) == entry["packaged_bytes"], entry["path"]
        assert digest(data) == entry["packaged_sha256"], entry["path"]
        if entry["encoding"] == "gzip-json":
            raw = gzip.decompress(data)
            assert digest(raw) == entry["original_sha256"], entry["path"]
            assert len(raw) == entry["original_bytes"], entry["path"]
            json.loads(raw)
            json_files += 1
        elif entry["encoding"] == "tar-gzip-source":
            raw = gzip.decompress(data)
            assert digest(raw) == entry["uncompressed_tar_sha256"], entry["path"]
            manifest = json.loads(gzip.decompress((root / entry["manifest"]).read_bytes()))["source_sha256"]
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
                members = archive.getmembers()
                assert len(members) == len(manifest), entry["path"]
                assert {member.name for member in members} == manifest.keys(), entry["path"]
                for member in members:
                    assert member.isfile(), member.name
                    value = archive.extractfile(member).read()
                    assert digest(value) == manifest[member.name] == entry["members_sha256"][member.name], member.name
                    source_files += 1
            archives += 1
        else:
            assert digest(data) == entry["original_sha256"], entry["path"]
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        assert digest((root / filename).read_bytes()) == expected, filename
    print(f"Verified {json_files} exact JSON gzip files, {archives} source archives, "
          f"{source_files} source members, and all packaged SHA-256 values.")


if __name__ == "__main__":
    main()

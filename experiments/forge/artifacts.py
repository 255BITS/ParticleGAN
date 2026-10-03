"""Portable content identities for complete saved evaluator inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def _digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def manifest_artifacts(root: Path | str) -> dict:
    """Hash every regular file, using paths relative to the portable root.

    The manifest is returned to the caller, not written inside its own input
    tree. Symlinks and empty trees cannot represent self-contained evidence.
    """
    root = Path(root)
    if root.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    if not root.is_dir():
        raise ValueError("artifact root is missing or is not a directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is not portable evidence: {path.relative_to(root)}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact is not a regular file: {path.relative_to(root)}")
        files[path.relative_to(root).as_posix()] = _file_identity(path)
    if not files:
        raise ValueError("artifact tree is empty")
    return {"schema_version": 1, "files": files, "sha256": _digest(files),
            "file_count": len(files), "total_bytes": sum(row["size"] for row in files.values())}


def verify_artifacts(root: Path | str, manifest: dict) -> None:
    """Reject incomplete, changed, added, or malformed certified inputs."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("artifact manifest needs schema_version 1")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("artifact manifest must list its complete files")
    for name, row in files.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or PurePosixPath(name).as_posix() != name or name == "."):
            raise ValueError("artifact manifest paths must be normalized relative paths")
        digest = row.get("sha256") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {"size", "sha256"}
                or type(row.get("size")) is not int or row["size"] < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"invalid artifact identity: {name}")
    if (type(manifest.get("file_count")) is not int or manifest["file_count"] != len(files)
            or type(manifest.get("total_bytes")) is not int
            or manifest["total_bytes"] != sum(row["size"] for row in files.values())
            or manifest.get("sha256") != _digest(files)):
        raise ValueError("artifact manifest digest or totals disagree with its files")
    actual = manifest_artifacts(root)
    missing, added = sorted(set(files) - set(actual["files"])), sorted(set(actual["files"]) - set(files))
    if missing or added:
        raise ValueError(f"artifact file set changed: missing={missing}, added={added}")
    changed = [name for name, row in files.items() if actual["files"][name] != row]
    if changed:
        raise ValueError(f"artifact bytes changed: {changed}")

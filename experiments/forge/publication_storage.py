"""Storage-only reductions for current publications and newly retained media.

Recorded rows, content identities and per-attempt receipts stay unchanged.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit, urlunsplit


CURRENT_INVENTORY_LIMIT_BYTES = 25 * 1024 * 1024
CONTRACT_CATALOGS = ("recipe_contracts", "task_contracts", "protocol_contracts")


def compact_current_publication(publication):
    """Drop unreachable catalog entries, following keys and values transitively.

    Only the current inventory has this storage policy. Archived/raw exports
    retain their complete catalogs. Catalog values are copied without rewriting
    their hashes or contents; every non-catalog field is preserved.
    """
    if not isinstance(publication, dict) or publication.get("publication_scope") != "current_technique_inventory":
        return deepcopy(publication)
    catalogs = {name: publication[name] for name in CONTRACT_CATALOGS if name in publication}
    if any(not isinstance(catalog, dict) for catalog in catalogs.values()):
        raise ValueError("current publication contract catalogs must be mappings")
    locations = defaultdict(set)
    for name, catalog in catalogs.items():
        for identity in catalog:
            locations[identity].add((name, identity))
    retained = set()
    pending = deque()

    def references(value):
        if isinstance(value, str):
            for location in locations.get(value, ()):
                if location not in retained:
                    retained.add(location)
                    pending.append(location)
        elif isinstance(value, dict):
            for key, item in value.items():
                references(key)
                references(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                references(item)

    references({key: value for key, value in publication.items() if key not in catalogs})
    while pending:
        name, identity = pending.popleft()
        references(catalogs[name][identity])
    result = deepcopy(publication)
    for name, catalog in catalogs.items():
        result[name] = {identity: result[name][identity] for identity in catalog
                        if (name, identity) in retained}
    return result


def publication_json_text(value, *, max_bytes=CURRENT_INVENTORY_LIMIT_BYTES):
    """Serialize with a bounded current inventory and unchanged archival format."""
    current = isinstance(value, dict) and value.get("publication_scope") == "current_technique_inventory"
    if current:
        value = compact_current_publication(value)
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if current and len(content.encode("utf-8")) > max_bytes:
        raise ValueError(f"current technique inventory exceeds its {max_bytes}-byte publication budget")
    return content


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RetainedMediaStore:
    """Reuse only tracked report GIFs or canonical files explicitly published here."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self._by_hash = None

    def _index(self):
        if self._by_hash is None:
            self._by_hash = {}
            names = subprocess.check_output(
                ["git", "ls-files", "-z", "--", "reports"], cwd=self.root).decode().split("\0")
            for name in sorted(filter(None, names)):
                path = self.root / name
                if path.suffix.lower() != ".gif" or path.is_symlink() or not path.is_file():
                    continue
                if not path.resolve().is_relative_to(self.root):
                    continue
                self._by_hash.setdefault(_sha256(path), path)
        return self._by_hash

    def retain(self, source, *, expected_sha256=None):
        source = Path(source)
        digest = _sha256(source)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"media hash differs from retained receipt: {source}")
        index = self._index()
        existing = index.get(digest)
        if existing is not None:
            if _sha256(existing) != digest:
                raise ValueError(f"retained media changed during publication: {existing}")
            return existing
        target = self.root / "reports/forge/media" / f"{digest}.gif"
        if target.is_symlink() or not target.resolve().is_relative_to(self.root):
            raise ValueError(f"canonical publication media path must stay inside the repository: {target}")
        # An ignored local artifact can never become a publication dependency.
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(target)], cwd=self.root)
        if ignored.returncode == 0:
            raise ValueError(f"canonical publication media path is ignored: {target}")
        if ignored.returncode != 1:
            raise RuntimeError("could not check canonical media ignore policy")
        _write_new(target, source.read_bytes())
        index[digest] = target
        return target


def _write_new(path, content):
    """Never overwrite an existing archived artifact with different bytes."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"preserve existing publication symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"preserve existing publication artifact: {path}")


def rewrite_media_links(text, source_parent, destination_parent, media):
    """Rebind relative Markdown image/link targets to retained GIF locations."""
    source_parent, destination_parent = Path(source_parent), Path(destination_parent)

    def replace(match):
        href = match.group(2)
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path:
            return match.group(0)
        retained = media.get((source_parent / unquote(parts.path)).resolve())
        if retained is None:
            return match.group(0)
        relative = Path(os.path.relpath(retained, destination_parent)).as_posix()
        return f"[{match.group(1)}]({urlunsplit(('', '', relative, parts.query, parts.fragment))})"

    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", replace, text)


def publish_media_tree(source, destination, store):
    """Publish a fresh media index and byte-exact receipts without GIF copies.

    Existing archived trees are refused. Only the newly written index and
    Markdown links change; each per-attempt JSON receipt remains byte-identical.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"preserve archived media tree: {destination}")
    if not destination.is_relative_to(store.root):
        raise ValueError("media publication destination must stay inside the repository")
    index = json.loads((source / "index.json").read_text())
    planned = []
    for row in index["media"]:
        gif, receipt = (source / row[key] for key in ("gif", "receipt"))
        if any(not path.resolve().is_relative_to(source) for path in (gif, receipt)):
            raise ValueError("source media index must reference files inside its media tree")
        for path, key in ((gif, "gif_sha256"), (receipt, "receipt_sha256")):
            if _sha256(path) != row[key]:
                raise ValueError(f"source media index hash differs: {path}")
        planned.append((row, gif, receipt))
    indexed = {gif.resolve() for _, gif, _ in planned}
    unindexed = {path.resolve() for path in source.rglob("*")
                 if path.is_file() and path.suffix.lower() == ".gif"} - indexed
    if unindexed:
        raise ValueError("media tree contains GIFs missing from its certified index")
    media = {}
    for row, gif, _ in planned:
        retained = store.retain(gif, expected_sha256=row["gif_sha256"])
        media[gif.resolve()] = retained
        row["gif"] = Path(os.path.relpath(retained, destination)).as_posix()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path == source / "index.json" or path.suffix.lower() == ".gif":
            continue
        target = destination / path.relative_to(source)
        content = path.read_bytes()
        if path.suffix.lower() == ".md":
            content = rewrite_media_links(content.decode("utf-8"), path.parent, target.parent, media).encode("utf-8")
        _write_new(target, content)
    _write_new(destination / "index.json", (json.dumps(index, indent=2, sort_keys=True) + "\n").encode())
    return media

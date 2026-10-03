"""Verify retained old Tier 1 archive bytes, original receipts and frozen sources.

The older snapshot container hash is retained as provenance. Its complete file
mapping, archived frozen files and executed Git commit are verified separately.
No training, result regrading, or sampling is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from publication_audit import GitSources, sha, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    root = args.worktree.resolve()
    card_path = root / "reports/forge/tier1-refresh/archive.json"
    card = json.loads(card_path.read_text())
    metadata = card["manifest"]
    archive_path = Path(card["archive"]["path"])
    expected = dict(metadata["original_receipts_sha256"])
    expected.update({f"frozen-scientific-source/{path}": digest
                     for path, digest in metadata["source_files_sha256"].items()})
    word_states = {
        "queue/51169502a0e64466b5005e7c870b5592/state.pt":
            "b26893116fdad34972b6770c4587d3ee05b37b7a7847b8bbd3225bdbbf981a3a",
        "queue/8e0a35381d6543bd8a3ecb6056351c06/state.pt":
            "c28c3869eab705781ff594edb635d8016e6578c919ec354c9377efd5ea64d35b",
    }
    expected.update(word_states)
    actual, restore = {}, None
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name not in expected and member.name != "restore-manifest.json":
                continue
            handle = archive.extractfile(member)
            if member.name == "restore-manifest.json":
                restore = json.loads(handle.read())
                continue
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            actual[member.name] = digest.hexdigest()
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    source = GitSources(root).manifest_check({"origin_commit": metadata["executed_commit"],
        "files": metadata["source_files_sha256"], "digest": stable_hash(metadata["source_files_sha256"])})
    mismatched = [path for path, expected_digest in expected.items()
                  if actual.get(path) != expected_digest]
    result = {
        "schema_version": 1,
        "scope": "read-only retained historical archive identity audit; no regrading or training",
        "audit_source": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).read_bytes()),
                         "publication_audit_sha256": sha(Path(__file__).with_name("publication_audit.py").read_bytes())},
        "archive_card": {"path": str(card_path), "sha256": sha(card_path.read_bytes())},
        "archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size,
                    "sha256": digest.hexdigest()},
        "archive_identity_matches_card": (archive_path.stat().st_size == card["archive"]["bytes"]
                                          and digest.hexdigest() == card["archive"]["sha256"]),
        "restoration_manifest_equals_card": restore == metadata,
        "original_receipts_checked": len(metadata["original_receipts_sha256"]),
        "archived_frozen_source_files_checked": len(metadata["source_files_sha256"]),
        "source_files_sha256_mapping_digest": stable_hash(metadata["source_files_sha256"]),
        "git_source": source,
        "word_states_sha256": word_states,
        "mismatched_or_missing_archive_members": mismatched,
        "snapshot_container_provenance": {
            "retained_sha256": metadata["snapshot_sha256"],
            "verified_scope": "Complete archived frozen source files and executed Git commit; original snapshot container is not a member of this archive.",
        },
    }
    result["pass"] = (result["archive_identity_matches_card"] and result["restoration_manifest_equals_card"]
                      and not mismatched and not source["missing"] and not source["mismatched"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Read-only audit of diagnostic receipt/source bindings and preserved results.

No model execution, optimizer update, resampling, or requalification is done.
Git blobs are read from each executed source commit, not the current checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


class GitSources:
    def __init__(self, root):
        self.root, self.trees, self.blobs = root, {}, {}

    def manifest_check(self, manifest):
        commit = manifest["origin_commit"]
        if commit not in self.trees:
            raw = subprocess.check_output(["git", "ls-tree", "-rz", commit], cwd=self.root)
            self.trees[commit] = {path.decode(): metadata.split()[2].decode()
                                 for entry in raw.split(b"\0") if entry
                                 for metadata, path in (entry.split(b"\t", 1),)}
        tree = self.trees[commit]
        missing = sorted(set(manifest["files"]) - set(tree))
        needed = sorted({tree[path] for path in manifest["files"] if path in tree} - set(self.blobs))
        if needed:
            process = subprocess.run(["git", "cat-file", "--batch"], cwd=self.root,
                                     input=("\n".join(needed) + "\n").encode(),
                                     stdout=subprocess.PIPE, check=True)
            cursor = 0
            for blob in needed:
                end = process.stdout.index(b"\n", cursor)
                oid, kind, length = process.stdout[cursor:end].split()
                if oid.decode() != blob or kind != b"blob":
                    raise ValueError("unexpected source object")
                start, size = end + 1, int(length)
                self.blobs[blob] = sha(process.stdout[start:start + size])
                cursor = start + size + 1
        mismatched = [path for path, expected in manifest["files"].items()
                      if path in tree and self.blobs[tree[path]] != expected]
        return {"origin_commit": commit, "source_files_checked": len(manifest["files"]),
                "digest_matches_files": stable_hash(manifest["files"]) == manifest["digest"],
                "missing": missing, "mismatched": mismatched}


def preservation(root, base):
    scopes = ("reports/forge/leaderboards", "reports/forge/technique-receipts",
              "reports/forge/configuration-search", "configs/forge/configurations",
              "configs/forge/selections", "reports/forge/records")
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", base, *scopes],
                                    cwd=root, text=True).splitlines()
    old_hashes, new_hashes = {}, {}
    for path in paths:
        old = subprocess.check_output(["git", "show", f"{base}:{path}"], cwd=root)
        old_hashes[path] = sha(old)
        new_hashes[path] = sha((root / path).read_bytes()) if (root / path).exists() else None
    return {"base_commit": base, "scopes": scopes,
            "existing_file_counts_by_scope": {scope: sum(path.startswith(scope + "/") for path in paths)
                                              for scope in scopes},
            "exclusions": [
                "reports/forge/EXPERIMENT_MEMORY.md and reports/forge/compilation.json are generated outputs intentionally refreshed; they are outside this byte-preservation claim.",
                "New task-only diagnostic receipts, source-history records, and reports are additions; only files already present at base_commit are compared here.",
                "Task source pins and implementation/test files intentionally changed by the fix are outside these preservation scopes.",
            ], "existing_files_checked": len(paths),
            "base_sha256_of_path_hashes": stable_hash(old_hashes),
            "current_sha256_of_path_hashes": stable_hash(new_hashes),
            "changed": [path for path in paths if new_hashes[path] != old_hashes[path]]}


def inspect_archive(path, expected_sha256, arm_ids):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    identities, texts, members, regular_files, all_files = {}, {}, set(), 0, {}
    inventory_bytes = None
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name in members:
                raise ValueError("duplicate archive member")
            members.add(member.name)
            if not member.isfile():
                continue
            regular_files += 1
            member_path = Path(member.name)
            identity = hashlib.sha256()
            capture = (member.name == "inventory.json" or
                       (member_path.parent.name in arm_ids and member_path.name.endswith(".json")))
            chunks = [] if capture else None
            handle = archive.extractfile(member)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                identity.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            all_files[member.name] = {"bytes": member.size, "sha256": identity.hexdigest()}
            if member.name == "inventory.json":
                inventory_bytes = b"".join(chunks)
            if member_path.parent.name in arm_ids:
                key = (member_path.parent.name, member_path.name)
                if key in identities:
                    raise ValueError("ambiguous archived arm artifact")
                identities[key] = {"member": member.name, "bytes": member.size,
                                   "sha256": identity.hexdigest()}
                if chunks is not None:
                    texts[key] = b"".join(chunks)
    if inventory_bytes is None:
        raise ValueError("archive has no inventory.json")
    inventory_rows = json.loads(inventory_bytes)
    inventory = {row["path"]: {key: row[key] for key in ("bytes", "sha256")}
                 for row in inventory_rows}
    actual_inventory = {path: value for path, value in all_files.items() if path != "inventory.json"}
    inventory_check = {
        "entries_declared": len(inventory_rows),
        "unique_entries": len(inventory),
        "actual_entries": len(actual_inventory),
        "inventory_sha256": sha(inventory_bytes),
        "declared_path_identity_mapping_sha256": stable_hash(inventory),
        "actual_path_identity_mapping_sha256": stable_hash(actual_inventory),
        "missing": sorted(set(inventory) - set(actual_inventory)),
        "unexpected": sorted(set(actual_inventory) - set(inventory)),
        "mismatched": [path for path in inventory if path in actual_inventory
                       and inventory[path] != actual_inventory[path]],
    }
    inventory_check["pass"] = (len(inventory_rows) == len(inventory)
                               and inventory == actual_inventory)
    return ({"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": digest.hexdigest(), "expected_sha256": expected_sha256,
             "sha256_matches_expected": digest.hexdigest() == expected_sha256,
             "members_checked": len(members),
             "regular_files_hashed": regular_files,
             "inventory": inventory_check,
             "binding": "Proof binds immutable archive bytes and audit source, not a card that might include this proof's hash."},
            identities, texts, all_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--base", default="153c0dd4fc9b725785e133722c9194d849324f10")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    if bool(args.archive) != bool(args.archive_sha256):
        parser.error("--archive and --archive-sha256 must be supplied together")
    root = args.worktree.resolve()
    sources, results = GitSources(root), []
    paths = sorted((root / "reports/forge/word-root-cause/receipts").glob("*.json"))
    archive_info, archived_identities, archived_texts, archived_files = None, {}, {}, {}
    expected_source_unions = {}
    if args.archive:
        archive_info, archived_identities, archived_texts, archived_files = inspect_archive(
            args.archive, args.archive_sha256,
            {json.loads(path.read_text())["id"] for path in paths})
    for path in paths:
        receipt = json.loads(path.read_text())
        raw_directory = Path(receipt["raw_directory"])
        checked, failures = {}, []
        local_available = all((raw_directory / name).is_file()
                              for name in (*receipt["raw_artifacts"], "compact-receipt.json"))
        def raw_text(name):
            if archive_info is not None:
                data = archived_texts.get((receipt["id"], name))
                if data is None:
                    raise ValueError(f"{receipt['id']}: missing archived {name}")
                return data
            return (raw_directory / name).read_bytes()
        for name, identity in receipt["raw_artifacts"].items():
            target = raw_directory / name
            if target.is_file():
                data = target.read_bytes()
                actual = {"bytes": len(data), "sha256": sha(data)}
            elif archive_info is not None and (receipt["id"], name) in archived_identities:
                actual = {key: archived_identities[(receipt["id"], name)][key]
                          for key in ("bytes", "sha256")}
            else:
                raise ValueError(f"{receipt['id']}: raw artifact is unavailable locally and in archive")
            checked[name] = actual
            if actual != identity:
                failures.append(f"{name}: raw identity differs")
            if archive_info is not None:
                archived = archived_identities.get((receipt["id"], name))
                if archived is None or {key: archived[key] for key in ("bytes", "sha256")} != identity:
                    failures.append(f"{name}: archive identity differs or member is absent")
        if (raw_directory / "compact-receipt.json").is_file() and path.read_bytes() != (raw_directory / "compact-receipt.json").read_bytes():
            failures.append("published receipt differs from frozen raw receipt")
        if archive_info is not None and path.read_bytes() != archived_texts.get((receipt["id"], "compact-receipt.json")):
            failures.append("published receipt differs from archived raw receipt")
        manifest = json.loads(raw_text("source-manifest.json"))
        source = sources.manifest_check(manifest)
        expected_union = expected_source_unions.setdefault(manifest["origin_commit"], {})
        for name, digest in manifest["files"].items():
            if name in expected_union and expected_union[name] != digest:
                raise ValueError("one executed commit has conflicting request source files")
            expected_union[name] = digest
        if archive_info is not None:
            source_prefix = f"sources/{manifest['origin_commit']}/"
            source["archived_copy_missing"] = [name for name in manifest["files"]
                                               if source_prefix + name not in archived_files]
            source["archived_copy_mismatched"] = [name for name, digest in manifest["files"].items()
                if source_prefix + name in archived_files and
                archived_files[source_prefix + name]["sha256"] != digest]
            if source["archived_copy_missing"] or source["archived_copy_mismatched"]:
                failures.append("archived executed source copy differs")
        if (source["missing"] or source["mismatched"] or not source["digest_matches_files"]
                or manifest["digest"] != receipt["source_digest"]
                or manifest["origin_commit"] != receipt["source_commit"]):
            failures.append("executed source binding differs")
        request = json.loads(raw_text("request.json"))
        protocol = request["protocol"]
        protocol_path = root / f"reports/forge/word-root-cause/round{protocol['id'].split('-round')[1].split('-')[0]}-protocol.json"
        if (protocol != json.loads(protocol_path.read_text())
                or sha(protocol_path.read_bytes()) != receipt["protocol_sha256"]
                or stable_hash(request["candidate"]) != receipt["candidate_revision"]
                or request["arm"]["recipe_delta"] != receipt["recipe_delta"]):
            failures.append("request/protocol/candidate binding differs")
        if (receipt["qualification_input"] is not False
                or receipt["eligible_for_default"] is not False
                or receipt["cost"]["completed_steps"] != protocol["task_updates"]):
            failures.append("diagnostic scope or budget differs")
        results.append({"id": receipt["id"], "gate": receipt["grade"]["gate_status"],
                        "published_receipt_sha256": sha(path.read_bytes()),
                        "original_raw_directory_available": local_available,
                        "raw_identity_scope": "archive plus available local original files" if archive_info else "local original files",
                        "raw_artifacts": checked,
                        "archived_artifacts": {name: archived_identities[(receipt["id"], name)]
                            for name in (*checked, "compact-receipt.json")
                            if (receipt["id"], name) in archived_identities},
                        "source": source, "failures": failures})
    source_snapshot_checks = {}
    if archive_info is not None:
        for commit, expected_union in expected_source_unions.items():
            prefix = f"sources/{commit}/"
            actual_union = {path[len(prefix):]: identity["sha256"]
                            for path, identity in archived_files.items() if path.startswith(prefix)}
            source_snapshot_checks[commit] = {
                "request_union_files": len(expected_union), "archived_union_files": len(actual_union),
                "request_union_mapping_sha256": stable_hash(expected_union),
                "archive_union_mapping_sha256": stable_hash(actual_union),
                "missing": sorted(set(expected_union) - set(actual_union)),
                "unexpected": sorted(set(actual_union) - set(expected_union)),
                "mismatched": [name for name in expected_union if name in actual_union
                               and expected_union[name] != actual_union[name]],
                "pass": expected_union == actual_union,
            }
    result = {"schema_version": 1,
              "scope": "read-only identity and historical preservation audit; no training or gate resampling",
              "worktree": str(root),
              "audit_source": {"path": str(Path(__file__).resolve()),
                               "sha256": sha(Path(__file__).read_bytes())},
              "archive": archive_info,
              "archived_source_snapshots": source_snapshot_checks,
              "preservation": preservation(root, args.base),
              "receipts_checked": len(results), "receipts": results}
    result["pass"] = (not result["preservation"]["changed"] and not any(r["failures"] for r in results)
                      and (archive_info is None or (archive_info["sha256_matches_expected"]
                                                   and archive_info["inventory"]["pass"]))
                      and all(check["pass"] for check in source_snapshot_checks.values()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

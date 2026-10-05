"""Preserve exact legacy unmeasured declaration rows during display refresh.

Earlier publication could render live unmeasured configuration alternatives
without registering them as scientific evidence. This immutable Git binding
retains those display declarations; it never admits measurement or selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, stable_hash
from experiments.forge.trainer_families import scientific_row_hash

REGISTRY = Path("reports/forge/technique-evidence/legacy-display-roster.json")
PUBLICATION = "reports/forge/technique-inventory.json"
POLICY = ("view", "view_revision", "policy_fingerprint", "tier_requirements")


class LegacyObjectUnavailable(ValueError):
    """The committed provenance receipt remains usable in a shallow checkout."""


def unmeasured_alternative(row):
    return (row.get("attempt_ids") == [] and row.get("selected_configuration") is False and
            row.get("alternative_scope") == "archived_alternative" and row.get("qualified_tier") == 0 and
            row.get("qualification_input") is False and row.get("qualification_reuse") is False and
            row.get("status") in {"INCOMPLETE", "UNKNOWN", "NOT_RUN", "BLOCKED"} and
            all(cell.get("passed") == 0 for cell in row.get("tiers", {}).values()) and
            all(task.get("status") in {"UNKNOWN", "NOT_RUN", "BLOCKED"}
                for task in row.get("tasks", []) + row.get("nonrequired_tasks", [])) and
            row.get("cost", {}).get("measured_tasks", 0) == 0 and
            row.get("cost", {}).get("wall_seconds") in {None, 0} and
            all(row.get("cost", {}).get(key) in {None, 0} for key in (
                "new_paid_wall_seconds", "evidence_wall_seconds", "execution_seconds", "paid_seconds")))


def _git(root, *arguments):
    try:
        return subprocess.check_output(["git", *arguments], cwd=root, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise LegacyObjectUnavailable("exact legacy display-publication Git object is unavailable") from error


def _source(root, commit):
    commit = _git(root, "rev-parse", "--verify", "--end-of-options", str(commit) + "^{commit}").decode().strip()
    identity = commit + ":" + PUBLICATION
    blob = _git(root, "rev-parse", "--verify", "--end-of-options", identity).decode().strip()
    content = _git(root, "show", identity)
    publication = json.loads(content)
    copied = deepcopy(publication)
    claimed = copied.get("provenance", {}).pop("input_digest", None)
    if (claimed != stable_hash(copied) or publication.get("publication_scope") != "current_technique_inventory" or
            publication.get("qualification_input") is not False or publication.get("qualification_reuse") is not False):
        raise ValueError("legacy display source must be an intact prior current publication")
    pointer = {"commit": commit, "path": PUBLICATION, "git_blob": blob,
               "sha256": hashlib.sha256(content).hexdigest(), "input_digest": claimed}
    return publication, pointer


def register(root, source_commit):
    root = Path(root).resolve()
    source, pointer = _source(root, source_commit)
    spec = importlib.util.spec_from_file_location("forge_legacy_roster_publisher", root / "reports/forge/regenerate_technique_inventory.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    manifest = publisher.read_json(root / publisher.EVIDENCE_MANIFEST)
    snapshot_hashes = set()
    for entry in manifest["cohorts"]:
        report, _ = publisher._snapshot(root, entry, manifest)
        snapshot_hashes.update(scientific_row_hash(row) for row in report["rows"])
    hashes = sorted({scientific_row_hash(row) for row in source.get("configuration_rows", [])
                     if unmeasured_alternative(row)} - snapshot_hashes)
    if not hashes:
        raise ValueError("prior publication contains no unmeasured legacy alternatives")
    registry = {"schema_version": 1, "scope": "legacy_unmeasured_display_declarations", "qualification_input": False,
                "source_publication": pointer, "policy": {key: source[key] for key in POLICY},
                "scientific_row_sha256": hashes}
    atomic_json(root / REGISTRY, registry)
    return registry


def verified_hashes(root, current, snapshot_hashes=()):
    root = Path(root).resolve()
    if not (root / REGISTRY).is_file():
        return set()
    registry = json.loads((root / REGISTRY).read_text())
    if (registry.get("schema_version") != 1 or registry.get("scope") != "legacy_unmeasured_display_declarations" or
            registry.get("qualification_input") is not False):
        raise ValueError("invalid legacy display roster")
    pointer = registry.get("source_publication", {})
    if (pointer.get("path") != PUBLICATION or
            any(not isinstance(pointer.get(key), str) or len(pointer[key]) != length or
                any(character not in "0123456789abcdef" for character in pointer[key])
                for key, length in (("commit", 40), ("git_blob", 40), ("sha256", 64), ("input_digest", 64)))):
        raise ValueError("invalid legacy display-publication provenance receipt")
    policy = registry["policy"]
    if any(current.get(key) != policy[key] for key in POLICY):
        raise ValueError("legacy display declarations cannot cross qualification policies")
    hashes = registry.get("scientific_row_sha256")
    if (not isinstance(hashes, list) or hashes != sorted(set(hashes)) or
            any(not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value) for value in hashes)):
        raise ValueError("invalid legacy display scientific-row identities")
    try:
        source, actual = _source(root, pointer["commit"])
    except LegacyObjectUnavailable:
        # This compact committed receipt is sufficient for display in fresh
        # shallow checkouts, just as compact technique receipts are. It cannot
        # establish any measured outcome or substitute qualification evidence.
        return set(hashes)
    if actual != pointer or {key: source[key] for key in POLICY} != policy:
        raise ValueError("legacy display-publication Git identity mismatch")
    expected = sorted({scientific_row_hash(row) for row in source.get("configuration_rows", [])
                       if unmeasured_alternative(row)} - set(snapshot_hashes))
    if expected != hashes:
        raise ValueError("legacy display roster differs from its exact prior scientific rows")
    return set(hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = register(args.root, args.source_commit)
    print(json.dumps({"registry": str(REGISTRY), "declaration_rows": len(result["scientific_row_sha256"]),
                      "source_publication": result["source_publication"]}, sort_keys=True), flush=True)

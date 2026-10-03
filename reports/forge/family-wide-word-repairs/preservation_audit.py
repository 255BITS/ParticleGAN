"""Read-only byte preservation proof for preexisting scientific history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


GROUPS = {
    "idea_cards": "configs/forge/ideas/",
    "task_cards": "configs/forge/tasks/",
    "search_specs": "configs/forge/searches/",
    "configuration_cards": "configs/forge/configurations/",
    "selection_exports": "configs/forge/selections/",
    "ordinary_receipts": "reports/forge/attempts/",
    "compact_receipts": "reports/forge/technique-receipts/",
    "search_receipts": "reports/forge/configuration-search/",
    "goal_leaderboards": "reports/forge/leaderboards/",
    "scientific_snapshots": "reports/forge/technique-evidence/",
    "source_history_records": "reports/forge/records/",
    "historical_word_evidence": "reports/forge/word-root-cause/",
    "original_word_example": "reports/forge/five-word-joint/",
}
EXCLUDED = {
    "reports/forge/technique-evidence/manifest.json": "Additive evidence registration.",
    "reports/forge/word-root-cause/README.md": "Current routing narrative; historical receipts and reproduction sources stay exact.",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def audit(root: Path, baseline: str) -> dict:
    baseline = git(root, "rev-parse", baseline).decode().strip()
    tree = git(root, "ls-tree", "-r", "-z", baseline).split(b"\0")
    groups = {name: {"root": prefix, "files": 0, "preserved": 0, "identities": []}
              for name, prefix in GROUPS.items()}
    failures = []
    for entry in tree:
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        path = name.decode()
        matches = [group for group, prefix in GROUPS.items() if path.startswith(prefix)]
        if not matches or path in EXCLUDED:
            continue
        if len(matches) != 1:
            raise ValueError("overlapping preservation roots")
        group = groups[matches[0]]
        blob = metadata.split()[2].decode()
        original = git(root, "cat-file", "blob", blob)
        target = root / path
        preserved = target.is_file() and target.read_bytes() == original
        group["files"] += 1
        group["preserved"] += preserved
        group["identities"].append({"path": path, "git_blob": blob, "sha256": sha256(original)})
        if not preserved:
            failures.append(path)
    for group in groups.values():
        identities = group.pop("identities")
        group["baseline_file_identities_sha256"] = sha256(json.dumps(
            identities, sort_keys=True, separators=(",", ":")).encode())
    return {"schema_version": 1, "scope": "source_history_byte_preservation",
            "qualification_input": False, "qualification_reuse": False,
            "baseline_commit": baseline, "worktree": str(root),
            "groups": groups, "files": sum(group["files"] for group in groups.values()),
            "preserved": sum(group["preserved"] for group in groups.values()),
            "status": "PASS" if not failures else "FAIL", "failures": failures,
            "exact_exclusions_within_roots": EXCLUDED,
            "generated_outputs_outside_preservation_roots": [
                "reports/forge/EXPERIMENT_MEMORY.md", "reports/forge/compilation.json",
                "reports/forge/technique-inventory.json", "reports/forge/technique-inventory.md"],
            "note": "Only files existing at the baseline are checked; additions do not rewrite history. No training, grading or qualification changes are performed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--baseline", default="53f2d423")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.worktree.resolve(), args.baseline)
    data = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    print(json.dumps({key: result[key] for key in ("status", "files", "preserved", "failures")}))
    raise SystemExit(result["status"] != "PASS")


if __name__ == "__main__":
    main()

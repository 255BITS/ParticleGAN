"""Reject new Ruff findings while retaining reviewed legacy fixtures and exports.

Use --update-baseline after reviewing additions or resolving existing findings.
The compact baseline stores messages and counts, without volatile line numbers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHS = ("particlegan", "particlegan_forge", "examples", "lib", "experiments",
         "benchmarks", "scripts", "ops", "tests")
RULES = "F,E9"


def findings(root: Path) -> Counter:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", RULES,
         "--output-format", "json", *PATHS],
        cwd=root, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "Ruff failed")
    counts = Counter()
    for item in json.loads(result.stdout):
        path = Path(item["filename"]).resolve().relative_to(root.resolve()).as_posix()
        message = re.sub(r"from line \d+", "from an earlier line", item["message"])
        counts[(path, item["code"], message)] += 1
    return counts


def unpack(document: dict) -> Counter:
    if document.get("schema_version") != 1 or document.get("rules") != RULES:
        raise ValueError("unsupported lint baseline; review and regenerate it")
    return Counter({(path, code, message): count
                    for path, rules in document["findings"].items()
                    for code, messages in rules.items()
                    for message, count in messages.items()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    baseline = root / ".ruff-baseline.json"
    current = findings(root)
    if args.update_baseline:
        grouped = {}
        for (path, code, message), count in sorted(current.items()):
            grouped.setdefault(path, {}).setdefault(code, {})[message] = count
        baseline.write_text(json.dumps({"schema_version": 1, "rules": RULES,
                                       "findings": grouped}, indent=2, sort_keys=True) + "\n")
        print(f"Recorded {sum(current.values())} reviewed Ruff findings in {baseline.name}")
        return 0
    expected = unpack(json.loads(baseline.read_text()))
    added, removed = current - expected, expected - current
    for label, delta in (("NEW", added), ("RESOLVED", removed)):
        for (path, code, message), count in sorted(delta.items()):
            print(f"{label}: {path}: {code} {message} ({count})")
    if added or removed:
        print("Review the changes, then run python scripts/check_lint.py --update-baseline")
        return 1
    print(f"Lint baseline current: {sum(current.values())} existing findings; no additions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

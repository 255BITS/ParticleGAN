"""Rank the family D CPU screen from screen.jsonl. Stdout is the table."""
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3p-det-d/screen.jsonl")
rows = []
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        continue
runs = [r for r in rows if not r.get("init_only") and r.get("status") in ("PASS", "FAIL")]
by = defaultdict(lambda: defaultdict(dict))
for row in runs:
    by[row["variant"]][row["gate"]][row["seed_offset"]] = row

priority = ("ring", "hold", "unequal", "stripes", "blobs", "grid100", "rotated100", "shift")


def count(variant, gate):
    items = by[variant].get(gate, {})
    return sum(1 for row in items.values() if row.get("status") == "PASS"), len(items)


def cell(variant, gate):
    passed, total = count(variant, gate)
    if total == 0:
        return ""
    if gate in ("ring", "unequal") and total > 1:
        return f"{passed}/{total}"
    return "P" if passed else "F"


order = sorted(by, key=lambda name: (-(count(name, "ring")[0] + count(name, "unequal")[0]),
                                      -count(name, "ring")[0], -count(name, "unequal")[0], name))
print(f"{'variant':<14} {'ring':>7} {'unequal':>8}  priority")
for name in order:
    gates = " ".join(f"{gate[0]}:{cell(name, gate) or '-'}" for gate in priority if cell(name, gate))
    print(f"{name:<14} {cell(name, 'ring'):>7} {cell(name, 'unequal'):>8}  {gates}")

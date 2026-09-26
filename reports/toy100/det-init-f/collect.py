"""Build the family-F leaderboard from progress.jsonl.

    python reports/toy100/det-init-f/collect.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROGRESS = ROOT / "results" / "det-init-f" / "progress.jsonl"
OUT = ROOT / "results" / "det-init-f" / "LEADERBOARD.md"
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)


def load():
    rows = []
    if not PROGRESS.exists():
        return rows
    for line in PROGRESS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "done":
            rows.append(row)
    return rows


def cell(row):
    if row is None:
        return "."
    if row.get("status") == "ERROR":
        return "E"
    if row.get("init_only"):
        return "D" if row.get("param_sha256") else "?"
    status = row.get("status")
    live = row.get("live") or {}
    mark = "P" if status == "PASS" else "F"
    if "modes" in live and live["modes"] is not None:
        mark += str(int(live["modes"]))
    return mark


def main():
    rows = load()
    by = {}
    for row in rows:
        by[(row.get("phase"), row.get("init"), row.get("gate"), row.get("seed_offset"))] = row
    inits = []
    for row in rows:
        if row.get("phase", "").startswith("wave") or row.get("init") in ("hid_q", "qr_pb_pq"):
            name = row.get("init")
            if name not in inits and row.get("gate") in ("ring", "unequal") and not row.get("init_only"):
                inits.append(name)
    # Keep calibration even if it is not a wave.
    lines = ["# Family F particle-prior screen", ""]
    base_ring = [by.get(("baseline", "k3p", "ring", offset)) for offset in OFFSETS]
    base_uneq = [by.get(("baseline", "k3p", "unequal", offset)) for offset in OFFSETS]
    def passes(items):
        known = [item for item in items if item and not item.get("init_only")]
        return sum(item.get("status") == "PASS" for item in known), len(known)
    br, bn = passes(base_ring)
    ur, un = passes(base_uneq)
    lines.append("## Calibration (random K3P init on this CPU build)")
    lines.append("")
    lines.append(f"| init | ring | unequal | ring seeds | unequal seeds |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(f"| k3p | {br}/{bn} | {ur}/{un} | {' '.join(cell(item) for item in base_ring)} | {' '.join(cell(item) for item in base_uneq)} |")
    lines.append("")
    ranked = []
    for name in inits:
        ring = []
        uneq = []
        phase = None
        for offset in OFFSETS:
            for candidate in by:
                if candidate[1] == name and candidate[2] == "ring" and candidate[3] == offset and not by[candidate].get("init_only"):
                    ring.append(by[candidate])
                    phase = candidate[0]
                    break
            else:
                ring.append(None)
            for candidate in by:
                if candidate[1] == name and candidate[2] == "unequal" and candidate[3] == offset and not by[candidate].get("init_only"):
                    uneq.append(by[candidate])
                    break
            else:
                uneq.append(None)
        rp, rn = passes(ring)
        up, unn = passes(uneq)
        ranked.append((rp, up, name, phase, ring, uneq, rn, unn))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    lines.append("## Ranked screen")
    lines.append("")
    lines.append("Rank is ring passes, then unequal-mass passes. `P`/`F` plus ring mode count. `.` is not finished.")
    lines.append("")
    lines.append("| rank | init | ring | unequal | ring seeds | unequal seeds |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for i, (rp, up, name, phase, ring, uneq, rn, unn) in enumerate(ranked, 1):
        lines.append(f"| {i} | {name} | {rp}/{rn} | {up}/{unn} | {' '.join(cell(item) for item in ring)} | {' '.join(cell(item) for item in uneq)} |")
    lines.append("")
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

"""Build the family-F leaderboard from progress.jsonl.

    python reports/toy100/det-init-f/collect.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROGRESS = ROOT / "results" / "det-init-f" / "progress.jsonl"
OUT = ROOT / "results" / "det-init-f" / "LEADERBOARD.md"
SUMMARY = ROOT / "results" / "det-init-f" / "SUMMARY.json"
OFFSETS = (0, 101, 202, 303, 404, 505, 606, 707)
PRIORITY_GATES = ("ring", "unequal", "stripes", "blobs", "hold", "shift", "grid100", "rotated100")


def load():
    rows = []
    if not PROGRESS.exists():
        return rows
    for line in PROGRESS.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
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


def passes(items):
    known = [item for item in items if item and not item.get("init_only")]
    return sum(item.get("status") == "PASS" for item in known), len(known)


def main():
    rows = load()
    done = [row for row in rows if row.get("event") == "done" and not row.get("init_only")]
    by = {}
    for row in done:
        by[(row.get("phase"), row.get("init"), row.get("gate"), row.get("seed_offset"))] = row
    builds = [row for row in rows if row.get("event") == "build" and row.get("torch")]
    determ = next((row for row in reversed(rows) if row.get("event") == "determinism_summary"), None)
    lines = ["# Family F particle-prior screen", ""]
    if builds:
        build = builds[-1]
        lines.append(
            f"Torch `{build.get('torch')}` git `{build.get('torch_git')}` file `{build.get('torch_file')}`."
        )
        lines.append("")
    if determ:
        lines.append(
            f"Determinism: {determ.get('checked')} pairs checked, {determ.get('bad')} mismatches."
        )
        lines.append("")

    base_ring = [by.get(("baseline", "k3p", "ring", offset)) for offset in OFFSETS]
    base_uneq = [by.get(("baseline", "k3p", "unequal", offset)) for offset in OFFSETS]
    br, bn = passes(base_ring)
    ur, un = passes(base_uneq)
    lines.append("## Calibration (random K3P init on this CPU build)")
    lines.append("")
    lines.append("| init | ring | unequal | ring seeds | unequal seeds |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| k3p | {br}/{bn} | {ur}/{un} | {' '.join(cell(item) for item in base_ring)} | "
        f"{' '.join(cell(item) for item in base_uneq)} |"
    )
    lines.append("")

    names = []
    for row in done:
        phase = row.get("phase") or ""
        if phase.startswith("wave") and row.get("gate") in ("ring", "unequal"):
            name = row.get("init")
            if name not in names:
                names.append(name)
    ranked = []
    for name in names:
        ring, uneq = [], []
        for offset in OFFSETS:
            ring.append(next((by[key] for key in by if key[0].startswith("wave") and key[1] == name and key[2] == "ring" and key[3] == offset), None))
            uneq.append(next((by[key] for key in by if key[0].startswith("wave") and key[1] == name and key[2] == "unequal" and key[3] == offset), None))
        rp, rn = passes(ring)
        up, unn = passes(uneq)
        ranked.append({
            "init": name,
            "ring_pass": rp,
            "ring_n": rn,
            "unequal_pass": up,
            "unequal_n": unn,
            "complete": rn == len(OFFSETS) and unn == len(OFFSETS),
            "ring_seeds": [cell(item) for item in ring],
            "unequal_seeds": [cell(item) for item in uneq],
        })
    ranked.sort(key=lambda item: (-item["ring_pass"], -item["unequal_pass"], item["init"]))
    lines.append("## Ranked screen")
    lines.append("")
    lines.append("Rank is ring passes, then unequal-mass passes, then name. `P`/`F` plus ring mode count. `.` is not finished.")
    lines.append("")
    lines.append("| rank | init | ring | unequal | ring seeds | unequal seeds |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for i, item in enumerate(ranked, 1):
        item["rank"] = i
        lines.append(
            f"| {i} | {item['init']} | {item['ring_pass']}/{item['ring_n']} | "
            f"{item['unequal_pass']}/{item['unequal_n']} | {' '.join(item['ring_seeds'])} | "
            f"{' '.join(item['unequal_seeds'])} |"
        )
    lines.append("")

    priority_names = []
    for row in done:
        if row.get("phase") == "priority" and row.get("init") not in priority_names:
            priority_names.append(row.get("init"))
    priority = []
    if priority_names:
        lines.append("## Priority gates (repo seed)")
        lines.append("")
        header = "| init | " + " | ".join(PRIORITY_GATES) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---" for _ in PRIORITY_GATES) + " |")
        for name in priority_names:
            marks = []
            detail = {"init": name, "gates": {}}
            for gate in PRIORITY_GATES:
                row = by.get(("priority", name, gate, 0))
                if not row:
                    marks.append(".")
                elif row.get("status") == "PASS":
                    marks.append("P")
                elif row.get("status") == "POST_CONVERGENCE_FAIL":
                    marks.append("post-fail")
                elif row.get("status") == "ERROR":
                    marks.append("E")
                else:
                    marks.append("F")
                if row:
                    detail["gates"][gate] = row.get("status")
            lines.append(f"| {name} | " + " | ".join(marks) + " |")
            priority.append(detail)
        lines.append("")

    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    complete = [item for item in ranked if item["complete"]]
    summary = {
        "torch": builds[-1] if builds else None,
        "determinism": determ,
        "calibration": {"ring": f"{br}/{bn}", "unequal": f"{ur}/{un}",
                        "ring_seeds": [cell(item) for item in base_ring],
                        "unequal_seeds": [cell(item) for item in base_uneq]},
        "ranked": ranked,
        "top2": [item["init"] for item in complete[:2]],
        "priority": priority,
        "errors": sum(row.get("status") == "ERROR" for row in done),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(text)


if __name__ == "__main__":
    main()

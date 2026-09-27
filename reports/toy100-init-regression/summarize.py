"""Leaderboard from runs/<arm>/<problem>/result.json. Usage: python summarize.py [runs_dir]"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "runs"
P = ("grid100", "rotated100", "staggered100")
rows = {}
for f in sorted(root.glob("*/*/result.json")):
    r = json.loads(f.read_text()); rows.setdefault(r["arm"], {})[r["problem"]] = r
print("| arm | gate g/r/s | modes g/r/s | final HQ g/r/s | first 100 modes g/r/s | max grad g/r/s | D init |")
print("|---|---|---|---|---|---|---|")
fmt = lambda arm, k, f: "/".join(f(rows[arm][p][k]) if p in rows[arm] and rows[arm][p][k] is not None else "-" for p in P)
for arm in sorted(rows, key=lambda a: (-sum(r["gate"] == "PASS" for r in rows[a].values()),
                                      -sum(r["hq"] or 0 for r in rows[a].values()) / len(rows[a]))):
    any_r = next(iter(rows[arm].values()))
    print(f"| {arm} | {fmt(arm, 'gate', lambda v: 'P' if v == 'PASS' else 'F')} | {fmt(arm, 'modes', str)} | "
          f"{fmt(arm, 'hq', lambda v: f'{v:.3f}')} | {fmt(arm, 'first_100_modes_step', str)} | "
          f"{fmt(arm, 'max_grad', lambda v: f'{v:.1f}')} | {any_r['desc']} |")

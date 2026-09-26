import json, os, sys
from pathlib import Path
gate = sys.argv[1] if len(sys.argv) > 1 else "toy-mode_hold"
for var in sorted(os.listdir("out")):
    cells = []; n = p = 0
    for o in (0, 101, 202, 303, 404, 505, 606, 707):
        f = Path("out", var, f"s{o}", gate, "result.json")
        if not f.exists():
            cells.append("." if not f.parent.exists() else "r"); continue
        d = json.loads(f.read_text()); ok = d.get("status") == "PASS"; n += 1; p += ok
        m = {x["metric"]: x["value"] for x in (d.get("verdict") or {}).get("metrics", [])}
        cells.append(("P" if ok else "F") + (str(m.get("modes")) if "modes" in m else ""))
    if n: print(f"{var:18s} {p}/{n}  " + " ".join(cells))

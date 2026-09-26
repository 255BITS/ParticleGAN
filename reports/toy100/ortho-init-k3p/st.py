import json, os, sys
from pathlib import Path
for var in sorted(os.listdir("out")):
    for sd in sorted(p for p in Path("out", var).glob("s*") if p.is_dir()):
        row = []
        for g in sorted(sd.iterdir()):
            f = g / "result.json"
            if not f.exists(): row.append(f"{g.name}=RUNNING"); continue
            d = json.loads(f.read_text()); st = d.get("status")
            ex = ""
            if g.name.startswith("toy-"):
                ms = (d.get("verdict") or {}).get("metrics", [])
                ex = ",".join(f"{m['metric']}={round(m['value'],3) if isinstance(m['value'],float) else m['value']}" for m in ms if m["status"] != "PASS" or m["metric"] == "modes")
            if g.name.startswith("shift-"):
                ex = "stay=%s/120" % (d.get("continued_hold") or {}).get("passing_checks")
            if g.name.startswith("native-"):
                ex = "cov=%s,acc=%s" % ((d.get("coverage") or {}).get("status"), (d.get("accuracy") or {}).get("status"))
            row.append(f"{g.name}={st}" + (f"({ex})" if ex else ""))
        print(var, sd.name, " | ".join(row))

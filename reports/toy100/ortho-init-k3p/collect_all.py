import json, os
from pathlib import Path
out = {}
for var in sorted(os.listdir("out")):
    V = out.setdefault(var, {"runs": {}, "determinism": {}})
    for sd in sorted(p for p in Path("out", var).iterdir() if p.is_dir() and p.name.startswith("s") and p.name[1:].isdigit()):
        for g in sorted(p for p in sd.iterdir() if p.is_dir()):
            f = g / "result.json"
            if not f.exists(): continue
            d = json.loads(f.read_text()); r = {"status": d.get("status")}
            if g.name.startswith("toy-"):
                r["metrics"] = {m["metric"]: [m["value"], m["status"]] for m in (d.get("verdict") or {}).get("metrics", [])}
            elif g.name.startswith("shift-"):
                r["stay"] = (d.get("continued_hold") or {}).get("passing_checks")
            elif g.name.startswith("native-"):
                r["coverage"] = (d.get("coverage") or {}).get("status"); r["accuracy"] = (d.get("accuracy") or {}).get("status")
            elif g.name.startswith("hold-"):
                r["converged_step"] = (d.get("gate") or {}).get("converged_step")
            oi = g / "ortho-init.json"
            if oi.exists():
                r["init_sha256"] = json.loads(oi.read_text())["all_params_sha256"]
            V["runs"].setdefault(sd.name, {})[g.name] = r
    a, b = Path("out", var, "init-s0"), Path("out", var, "init-s101")
    if a.exists() and b.exists():
        for g in sorted(p.name for p in a.iterdir()):
            fa, fb = a / g / "ortho-init.json", b / g / "ortho-init.json"
            if fa.exists() and fb.exists():
                ja, jb = json.loads(fa.read_text()), json.loads(fb.read_text())
                V["determinism"][g] = {"n": ja["n"], "sha_s0": ja["all_params_sha256"], "sha_s101": jb["all_params_sha256"],
                                       "match": ja["all_params_sha256"] == jb["all_params_sha256"],
                                       "rules": sorted({p["rule"] for p in ja["params"]}), "spec": ja.get("spec")}
print(json.dumps(out))

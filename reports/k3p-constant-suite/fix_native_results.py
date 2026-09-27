"""Re-derive native-* result.json status from bench/gate-<problem>.json (runs made before run_suite's 22:40 file-name fix read gate.json, which does not exist). Idempotent."""
import json, sys
from pathlib import Path
runs = Path(__file__).resolve().parent / "runs"
for r in sorted(runs.glob("*/native-*/result.json")):
    res = json.loads(r.read_text()); p = r.parent.name.split("-", 1)[1]; b = r.parent / "bench"
    g = b / f"gate-{p}.json"; a = b / f"accuracy-gate-{p}.json"
    if not (g.exists() and a.exists()): continue
    gs, as_ = json.loads(g.read_text()).get("status"), json.loads(a.read_text()).get("status")
    passed = gs == "PASS" and as_ == "PASS"
    new = res["detail"].replace("coverage None accuracy None", f"coverage {gs} accuracy {as_}")
    if new != res["detail"] or res["passed"] != passed:
        res.update(status="PASS" if passed else "FAIL", passed=passed, detail=new, harness_fix="status re-derived from gate-<problem>.json")
        r.write_text(json.dumps(res, indent=1, default=str) + "\n"); print("fixed", r.parent.parent.name, r.parent.name, res["status"], new)

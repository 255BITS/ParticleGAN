"""Convert a KA2 reference run (raw/ from the unmodified KA2 worker) into a summarize.py result.json.

Usage: ka2_ref_result.py runs/ka2_stock_ref   (reads runs/ka2_stock_ref/raw/, writes runs/ka2_stock_ref/result.json)
Points are the live observations (step, modes, hq); the init receipt is copied from raw/declaration.json.
"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]).resolve()
raw = run / "raw"
points = [{"step": r["step"], "modes": r["modes"], "hq": r["hq"]}
          for r in map(json.loads, (raw / "metrics.jsonl").read_text().splitlines())
          if r.get("event") == "observation"]
declaration = json.loads((raw / "declaration.json").read_text())
result = {"arm": run.name,
          "formulation": "REF: stock KA2 worker rerun (RpGAN+KA2 penalty/controller, noise on)",
          "source": f"runs/{run.name}/raw (reports/ka2-default-candidate/constant-lr-api/worker.py)",
          "init_receipt": declaration.get("init_receipt"),
          "points": points}
(run / "result.json").write_text(json.dumps(result) + "\n")
print(f"{run.name}: {len(points)} points, init={result['init_receipt'] and result['init_receipt']['initialization']}")

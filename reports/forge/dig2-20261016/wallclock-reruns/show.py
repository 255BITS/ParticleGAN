"""Print a compact per-cell line from analyze.py output. Usage: show.py table.json"""
import json
import sys

KEYS = ("root", "cell", "status", "worker_seconds", "last_step", "solo_spu", "earlier_spu_low", "earlier_spu_high",
        "slowdown_low", "slowdown_high", "gpu", "gpu_util_mean", "nonresident_pids_on_gpu", "confirmed_checks")
for row in json.load(open(sys.argv[1])):
    print({k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in KEYS if k in row},
          row.get("wallclock") or "", (row.get("reason") or "")[:80] if row.get("status") not in ("PASS",) else "")

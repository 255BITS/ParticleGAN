"""Real/fake input-gradient RMS on the ring (logged every 10 updates in ring8-multishift points; needs the
kernel's phase-A diagnostics, i.e. runs after the reg_real_mode change). Prints a markdown table.
Columns: median and max over updates 1210-2400 (prehold) and over the whole run, of the per-batch mean and max
real RMS ||grad D(real)||/sqrt(d); plus the whole-run max fake RMS.
Usage: grid_gradstats.py arm[,arm...]   (runs_grid/, then runs_grid_diag/ for the reference arms)"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(arm):
    for root in ("runs_grid", "runs_grid_diag"):
        f = HERE / root / arm / "ring8-multishift" / "result.json"
        if f.exists():
            pts = json.loads(f.read_text())["raw"]["points"]
            if "real_rms_max" in pts[-1]:
                return pts
    return None


print("| arm | prehold real RMS mean (median) | prehold real RMS max (median / max) | run real RMS max (max) | "
      "run fake RMS max (max) |")
print("|---|---|---|---|---|")
for arm in sys.argv[1].split(","):
    pts = load(arm)
    if pts is None:
        print(f"| {arm} | not logged | | | |")
        continue
    pre = [p for p in pts if 1210 <= p["step"] <= 2400]
    med = lambda k, ps: statistics.median(p[k] for p in ps)  # noqa: E731
    print(f"| {arm} | {med('real_rms_mean', pre):.3f} | {med('real_rms_max', pre):.2f} / "
          f"{max(p['real_rms_max'] for p in pre):.2f} | {max(p['real_rms_max'] for p in pts):.2f} | "
          f"{max(p['fake_rms_max'] for p in pts):.2f} |")

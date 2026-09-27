"""Round-5 ring table: summarize.score + curvature medians. Usage: round5_table.py ARM..."""
import json
import statistics
import sys
from pathlib import Path

from summarize import score

HERE = Path(__file__).resolve().parent
print("| arm | formulation | prehold | arrival | post-arrival | departures | longest fail streak | fails outside transit "
      "| final HQ | max abs D(real) | max grad | median curv (wide) | median g(real) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for arm in sys.argv[1:]:
    p = HERE / "runs" / arm / "result.json"
    if not p.exists():
        p = HERE / "diag" / "runs" / arm / "result.json"
    r = json.loads(p.read_text())
    s, c = score(r["points"]), r.get("curvature", {})
    gr = statistics.median(q["g_real"] for q in r["points"])
    f = r["formulation"].replace(" {lr c×0.5 g×1}", "").replace(" [Dβ2=0.9, A2=0]", "")
    print(f"| {arm} | {f} | {s['pre_pass']}/{s['pre_n']} | {s['arrival'] if s['arrival'] is not None else 'none'} "
          f"| {s['post_pass']}/{s['post_n']} | {s['departures']} | {s['streak']} | {s['fails']} | {s['final_hq']:.3f} "
          f"| {s['dmax']:.2f} | {s['gmax']:.2f} | {c.get('median_curv', float('nan')):.2f} "
          f"({c.get('median_curv_wide', float('nan')):.2f}) | {gr:.3f} |")

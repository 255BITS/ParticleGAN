"""Compact per-eval timeline for one run dir: step, modes, hq (live), probe g_max/g_real/Dr-Df nearest step."""
import json, sys
from pathlib import Path
run, prob = Path(sys.argv[1]), sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 500
probes = {}
p = run / "diag" / f"{prob}.jsonl"
if not p.exists():  # this study's layout: runs/<arm>/<problem>/{diag.jsonl,bench/<problem>/events.jsonl}
    run = run / prob
    p = run / "diag.jsonl"
if p.exists():
    for l in p.read_text().splitlines():
        r = json.loads(l); probes[r["step"]] = r
ev = run / "bench" / prob / "events.jsonl"
for l in ev.read_text().splitlines():
    r = json.loads(l)
    if r.get("event") != "eval" or r.get("model") != "live":
        continue
    s = r["step"]
    if s % every and s not in (1, 10, 50, 100):
        continue
    m = r["metrics"]
    q = probes.get(s) or probes.get(s - s % 50) or {}
    print(f"{s:5d} modes={m.get('modes'):3} hq={m.get('hq'):.3f} pass={int(bool(m.get('passed')))} "
          f"gmax={q.get('g_max', float('nan')):.2f} gr={q.get('g_real', float('nan')):.2f} "
          f"gf={q.get('g_fake', float('nan')):.2f} Dr-Df={q.get('dr_mean', q.get('dr', 0)) - q.get('df_mean', q.get('df', 0)):+.3f} "
          f"lr_d={q.get('lr_d', float('nan')):.2g}")

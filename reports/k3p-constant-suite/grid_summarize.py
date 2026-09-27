"""Grid-search tables for GRID.md: runs_grid/<arm> plus the reference arms k3p_simple / k3p_stock from runs/.

Usage: grid_summarize.py [--arms a,b,...] [--full]   (default: every arm in runs_grid/ with a result)
Native cell: pass | cov-gate terminal checks /5 | accuracy terminal checks /5 | final HQ | per-mode cov eig ratio
min-max (gate .4-1.7) | worst terminal center RMS/sigma (gate .2). Toy cell: passing suffix /24.
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from run_suite import TASKS  # noqa: E402

NATIVE = ("grid100", "rotated100", "staggered100")
TARGET = [f"native-{p}" for p in NATIVE] + ["toy-img_bars4", "toy-two_pole"]
REFS = ("k3p_simple", "k3p_stock")


def results(arm):
    root = HERE / ("runs" if arm in REFS else "runs_grid") / arm
    out = {}
    for f in root.glob("*/result.json"):
        out[f.parent.name] = json.loads(f.read_text())
    return out


def native(arm, p):
    b = HERE / ("runs" if arm in REFS else "runs_grid") / arm / f"native-{p}" / "bench"
    try:
        g = json.loads((b / f"gate-{p}.json").read_text())["problems"][p]
        a = json.loads((b / f"accuracy-gate-{p}.json").read_text())["problems"][p]
    except (FileNotFoundError, KeyError):
        return None
    m = re.search(r"only (\d+)/5", g.get("reason", ""))
    cov = 5 if g["status"] == "PASS" else int(m.group(1)) if m else 0
    f = g["final_metrics"]
    tc = a["terminal_checks"]
    return dict(passed=g["status"] == "PASS" and a["status"] == "PASS", cov=cov,
                acc=sum(c["passed"] for c in tc), hq=f["hq"], emin=f["min_cov_eig_ratio"], emax=f["max_cov_eig_ratio"],
                center=max((c["metrics"]["center_rms_sigma"] or float("inf")) for c in tc) if tc else float("nan"),
                tv=f["mass_tv"], modes=f["modes"])


def ncell(n):
    if n is None:
        return "–"
    return (f"{'P' if n['passed'] else 'F'} {n['cov']}/{n['acc']} {n['hq']:.3f} "
            f"{n['emin']:.2f}-{n['emax']:.2f} c{n['center']:.2f}")


def ring(r):
    s = ((r or {}).get("raw") or {}).get("score") or {}
    if not s:
        return "–" if r is None else r.get("status", "?")
    arr = "/".join("x" if g["arrival"] is None else f"+{g['arrival']}" for g in s["segments"])
    return f"{'P' if r['passed'] else 'F'} f{s['fails_outside_transit']} {arr} d{s['departures']}"


def row5(arm):
    res = results(arm)
    ns = [native(arm, p) for p in NATIVE]
    toy = lambda t: (f"{'P' if res[t]['passed'] else 'F'} {res[t]['metric']['value']}" if t in res else "–")  # noqa
    passes = sum(res[t].get("passed") is True for t in TARGET if t in res)
    done = sum(t in res for t in TARGET)
    ok = [n for n in ns if n]
    emin = sum(n["emin"] for n in ok) / len(ok) if ok else float("nan")
    cen = sum(n["center"] for n in ok) / len(ok) if ok else float("nan")
    hq = sum(n["hq"] for n in ok) / len(ok) if ok else float("nan")
    return dict(arm=arm, passes=passes, done=done, cells=[ncell(n) for n in ns] + [toy("toy-img_bars4"), toy("toy-two_pole")]
                + [ring(res.get("ring8-shift")), ring(res.get("ring8-multishift"))],
                emin=emin, center=cen, hq=hq, covchecks=sum(n["cov"] for n in ok), accchecks=sum(n["acc"] for n in ok))


def stage_table(arms):
    rows = [row5(a) for a in arms]
    rows.sort(key=lambda r: (-r["passes"], -(r["covchecks"] + r["accchecks"]), r["center"]))
    lines = ["| arm | 5-task passes | native checks cov+acc /30 | mean HQ | mean min eig | mean worst center | grid100 | "
             "rotated100 | staggered100 | img_bars4 | two_pole | ring8-shift | ring8-multishift |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['passes']}/{r['done']} | {r['covchecks'] + r['accchecks']} | {r['hq']:.3f} | "
                     f"{r['emin']:.2f} | {r['center']:.2f} | " + " | ".join(r["cells"]) + " |")
    return lines


def full_table(arms):
    lines = ["| arm | passes | native | transfer /19 | ring8-shift | ring8-multishift |", "|---|---|---|---|---|---|"]
    per = {}
    for a in arms:
        res = results(a)
        per[a] = res
        passes = sum(r.get("passed") is True for r in res.values())
        nat = sum(res.get(f"native-{p}", {}).get("passed") is True for p in NATIVE)
        tr = sum(res.get(t, {}).get("passed") is True for t in TASKS if t.startswith("toy-"))
        lines.append(f"| {a} | {passes}/{len(res)} | {nat}/3 | {tr} | {ring(res.get('ring8-shift'))} | "
                     f"{ring(res.get('ring8-multishift'))} |")
    lines += ["", "| task | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for t in TASKS:
        cells = []
        for a in arms:
            r = per[a].get(t)
            if r is None:
                cells.append("–")
            elif t.startswith("ring8"):
                cells.append(ring(r))
            else:
                cells.append(f"{'P' if r.get('passed') else 'F'} {(r.get('metric') or {}).get('value')}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    arms = (a.arms.split(",") if a.arms else
            list(REFS) + sorted(p.name for p in (HERE / "runs_grid").glob("*/") if any(p.glob("*/result.json"))))
    print("\n".join(full_table(arms) if a.full else stage_table(arms)))

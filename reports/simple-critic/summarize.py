"""Leaderboard for simple-critic arms: runs/*/result.json (or metrics.jsonl while running).

Scoring (observations every 10 updates; pass = 8 modes and HQ >= .90):
  prehold      passing checks in 1210..2400 (x/120)
  arrival      updates from the shift (after 2400) to the first passing check
  post         passing checks from first arrival to the end (x/y)
  departures   pass->fail transitions after the first pre-shift pass, excluding
               the shift transit (2400 -> first arrival) = the seesaw count
  streak       longest failing run (checks) outside the shift transit
Rank: arrived; fewest fails outside transit (prehold + post-arrival); fewest
departures; earliest arrival. The archived KA2 constant run (noise on, KA2
controller) is a labeled reference row.

Usage: python summarize.py [--write | --diag]   (--write also saves leaderboard.md; --diag prints probe medians)
"""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIFT, PRE = 2400, (1210, 2400)
REF = HERE.parent / "ka2-default-candidate/constant-lr-api/evidence/constant/metrics.jsonl.gz"


def good(p):
    return p["modes"] == 8 and .90 <= p["hq"] <= 1.0


def score(points):
    pts = sorted((p for p in points if p["step"] % 10 == 0), key=lambda p: p["step"])
    ok = [good(p) for p in pts]
    first_acq = next((p["step"] for p, g in zip(pts, ok) if g and p["step"] <= SHIFT), None)
    arrival = next((p["step"] for p, g in zip(pts, ok) if g and p["step"] > SHIFT), None)
    pre = [(p, g) for p, g in zip(pts, ok) if PRE[0] <= p["step"] <= PRE[1]]
    post = [(p, g) for p, g in zip(pts, ok) if arrival is not None and p["step"] >= arrival]
    segments = [[g for p, g in zip(pts, ok) if first_acq is not None and first_acq <= p["step"] <= SHIFT],
                [g for _, g in post]]
    departures = sum(a and not b for seg in segments for a, b in zip(seg, seg[1:]))
    streak = 0
    for seg in segments:
        run = 0
        for g in seg:
            run = 0 if g else run + 1
            streak = max(streak, run)
    suffix = 0
    for g in reversed(ok):
        if not g:
            break
        suffix += 1
    pre_fail = sum(not g for _, g in pre)
    post_fail = sum(not g for _, g in post)
    last = pts[-1] if pts else {"step": 0, "hq": float("nan")}
    return dict(last_step=last["step"], pre_pass=len(pre) - pre_fail, pre_n=len(pre), first_acq=first_acq,
                arrival=None if arrival is None else arrival - SHIFT,
                post_pass=len(post) - post_fail, post_n=len(post), departures=departures, streak=streak,
                fails=pre_fail + post_fail, suffix=suffix,
                suffix_from=None if not suffix else pts[-suffix]["step"], final_hq=last["hq"],
                dmax=max((p["dr_absmax"] for p in pts if "dr_absmax" in p), default=None),
                gmax=max((p["g_max"] for p in pts if "g_max" in p), default=None))


def load_arms():
    rows = []
    for run in sorted((HERE / "runs").glob("*/")):
        res, met = run / "result.json", run / "metrics.jsonl"
        if res.exists():
            r = json.loads(res.read_text())
            # A rerun of the KA2 reference (noise + controller) is a labeled reference row, not a ranked arm.
            status = "reference" if r["formulation"].startswith("REF") else "done"
            rows.append(dict(arm=r["arm"], formulation=r["formulation"], status=status, **score(r["points"])))
        elif met.exists():
            pts = [json.loads(line) for line in met.read_text().splitlines() if line.strip()]
            decl = run / "declaration.json"
            form = json.loads(decl.read_text())["formulation"] if decl.exists() else "?"
            if pts:
                s = score(pts)
                rows.append(dict(arm=run.name, formulation=form, status=f"running@{s['last_step']}", **s))
    return rows


def load_reference():
    if not REF.exists():
        return None
    with gzip.open(REF, "rt") as f:
        pts = [r for r in map(json.loads, f) if r.get("event") == "observation"]
    return dict(arm="ref:ka2-constant", formulation="RpGAN+KA2 penalty/controller, noise on (archived)",
                status="reference", **score(pts))


def fmt(v, spec="{}", none="-"):
    return none if v is None else spec.format(v)


def table(rows):
    rows.sort(key=lambda r: (r["arrival"] is None, r["fails"], r["departures"],
                             r["arrival"] if r["arrival"] is not None else 1e9))
    head = ("| # | arm | formulation | prehold | arrival | post-arrival | departures | longest fail streak "
            "| final suffix | final HQ | max abs D(real) | max grad-norm | fails outside transit |")
    lines = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    rank = 0
    for r in rows:
        if r["status"] == "reference":
            label = "ref"
        else:
            rank += 1
            label = str(rank) if r["status"] == "done" else f"{rank}*"
        lines.append("| " + " | ".join([
            label, r["arm"], r["formulation"], f"{r['pre_pass']}/{r['pre_n']}",
            fmt(r["arrival"], "{}", "none"), f"{r['post_pass']}/{r['post_n']}", str(r["departures"]),
            str(r["streak"]), f"{r['suffix']}" + (f" (from {r['suffix_from']})" if r["suffix"] else ""),
            f"{r['final_hq']:.3f}", fmt(r["dmax"], "{:.2f}", "n/a"), fmt(r["gmax"], "{:.2f}", "n/a"),
            str(r["fails"])]) + " |")
    return "\n".join(lines)


def diag():
    """Median probe diagnostics per finished arm (python summarize.py --diag)."""
    import statistics as st
    head = ("| arm | best HQ (step) | mean HQ 1210-2400 | mean HQ 2410-3600 | mean HQ 3600-4600 | pass 2410-4600 "
            "| g(real) | g(path) | gmax | obs gmax>2 | max gmax |")
    lines = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    rows = []
    for run in sorted((HERE / "runs").glob("*/")):
        f = run / "result.json"
        if not f.exists():
            continue
        pts = json.loads(f.read_text())["points"]
        if "g_max" not in pts[0]:
            continue
        mean = lambda a, b: st.mean(p["hq"] for p in pts if a <= p["step"] <= b)
        best = max(pts, key=lambda p: p["hq"])
        post = [p for p in pts if p["step"] > 2400]
        g = [p["g_max"] for p in pts]
        rows.append((-mean(2410, 4600), "| " + " | ".join([
            run.name, f"{best['hq']:.3f} ({best['step']})", f"{mean(1210, 2400):.3f}", f"{mean(2410, 3600):.3f}",
            f"{mean(3600, 4600):.3f}", f"{sum(p['pass'] for p in post)}/{len(post)}",
            f"{st.median(p['g_real'] for p in pts):.2f}", f"{st.median(p['g_path'] for p in pts):.2f}",
            f"{st.median(g):.2f}", f"{sum(x > 2 for x in g)}/{len(g)}", f"{max(g):.3g}"]) + " |"))
    return "\n".join(lines + [r for _, r in sorted(rows)])


def main():
    if "--diag" in sys.argv:
        print(diag())
        return
    rows = load_arms()
    ref = load_reference()
    if ref:
        rows.append(ref)
    out = table(rows)
    note = ("\n\n`*` = still running (scored through its last observation). Departures and streaks exclude "
            "the shift transit (2400 to first arrival). ref = archived KA2 constant-LR run (not a simple arm).")
    print(out + note)
    if "--write" in sys.argv:
        (HERE / "leaderboard.md").write_text("# Simple-critic leaderboard\n\n" + out + note + "\n")


if __name__ == "__main__":
    main()

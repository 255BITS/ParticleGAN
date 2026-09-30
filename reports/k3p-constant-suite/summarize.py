"""Leaderboard for the K3P constant-LR suite: runs/<arm>/<task>/result.json -> LEADERBOARD.md.

Rank: total passes over the 26 tasks, then ring fails outside transit (shift + multishift), then
multishift arrivals, then native passes. Also checks that every arm used the same init per task.
Usage: python summarize.py [--runs-dir DIR] [--write]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from run_suite import TASKS  # noqa: E402

SHORT = {t: t.replace("native-", "").replace("toy-", "").replace("vector_", "v_").replace("img_", "i_")
         .replace("-mode_hold", "") for t in TASKS}


def load(runs):
    arms = json.loads((HERE / "arms.json").read_text())["arms"]
    rows = {}
    for arm_dir in sorted(p for p in runs.glob("*/") if p.is_dir()):
        res = {}
        for task_dir in arm_dir.glob("*/"):
            f = task_dir / "result.json"
            if f.exists():
                res[task_dir.name] = json.loads(f.read_text())
        if res:
            rows[arm_dir.name] = dict(results=res, formulation=arms.get(arm_dir.name, {}).get("formulation", "?"))
    return rows


def ring(r):
    s = ((r or {}).get("raw") or {}).get("score") or {}
    return s


def fmt_ring(r):
    s = ring(r)
    if not s:
        return "–" if r is None else r.get("status", "?")
    arr = "/".join("x" if g["arrival"] is None else f"+{g['arrival']}" for g in s["segments"])
    return f"{s['prehold']} {arr} f{s['fails_outside_transit']} d{s['departures']}"


def table(rows):
    lines = []
    ranked = []
    for arm, row in rows.items():
        res = row["results"]
        done = [t for t in TASKS if t in res]
        passes = sum(res[t].get("passed") is True for t in done)
        na = sum(res[t].get("status") == "N/A" for t in done)
        rs, rm = ring(res.get("ring8-shift")), ring(res.get("ring8-multishift"))
        rfail = sum(s.get("fails_outside_transit", 10 ** 6) for s in (rs, rm)) if rs and rm else 10 ** 6
        marr = sum(g["arrival"] is not None for g in rm.get("segments", [])) if rm else 0
        native = sum(res.get(f"native-{p}", {}).get("passed") is True for p in ("grid100", "rotated100", "staggered100"))
        ranked.append((-passes, rfail, -marr, -native, arm, passes, len(done), na))
    ranked.sort()
    lines.append("| # | arm | passes | native | transfer 19 | hold | shift | ring8-shift (prehold arrival fails dep) "
                 "| ring8-multishift (prehold arrivals fails dep) | formulation |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, _, _, native, arm, passes, done, na) in enumerate(ranked, 1):
        res = rows[arm]["results"]
        tr = sum(res.get(f"toy-{t}", {}).get("passed") is True for t in
                 [t[4:] for t in TASKS if t.startswith("toy-")])
        trn = sum(1 for t in TASKS if t.startswith("toy-") and t in res)
        st = lambda t: res[t]["status"] if t in res else "–"  # noqa: E731
        nn = sum(f"native-{p}" in res for p in ("grid100", "rotated100", "staggered100"))
        copied = sum(bool(r.get("inferred")) for r in res.values())
        name = f"{arm} (copied: {copied}/{done} results)" if copied else arm
        lines.append(f"| {i} | {name} | {passes}/{done}" + (f" ({na} N/A)" if na else "") +
                     f" | {f'{-native}/{nn}' if nn else '–'} | {tr}/{trn} | {st('hold-mode_hold')} | {st('shift-mode_hold')} | "
                     f"{fmt_ring(res.get('ring8-shift'))} | {fmt_ring(res.get('ring8-multishift'))} | "
                     f"{rows[arm]['formulation']} |")
    return lines, [r[4] for r in ranked]


def per_task(rows, order):
    lines = ["| task | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for t in TASKS:
        cells = []
        for arm in order:
            r = rows[arm]["results"].get(t)
            if r is None:
                cells.append("–")
                continue
            m = (r.get("metric") or {}).get("value")
            mark = {"PASS": "P", "FAIL": "F", "ERROR": "ERR", "N/A": "N/A"}.get(r["status"], r["status"])
            cells.append(f"{mark} {m}" if m is not None else mark)
        lines.append(f"| {SHORT[t]} | " + " | ".join(cells) + " |")
    return lines


def runtime(rows, order):
    lines = ["| task | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    total = {a: 0.0 for a in order}
    for t in TASKS:
        cells = []
        for arm in order:
            r = rows[arm]["results"].get(t)
            cells.append("–" if r is None else f"{r.get('seconds', 0):.0f}" + ("*" if r.get("inferred") else ""))
            total[arm] += 0 if r is None else r.get("seconds", 0)
        lines.append(f"| {SHORT[t]} | " + " | ".join(cells) + " |")
    lines.append("| **sum (s)** | " + " | ".join(f"{total[a]:.0f}" for a in order) + " |")
    return lines


def checks(rows):
    """Init identical across arms per task; receipts for noise/LR/AMSGrad."""
    out = []
    for t in TASKS:
        hashes = {a: r["results"][t]["init"]["init_sha256"] for a, r in rows.items() if t in r["results"]}
        arms = json.loads((HERE / "arms.json").read_text())["arms"]
        declared = {a for a in hashes if arms.get(a, {}).get("init") == "constructor"}
        rest = {a: h for a, h in hashes.items() if a not in declared}
        if len(set(rest.values())) > 1:
            out.append(f"- INIT MISMATCH on {t}: " + ", ".join(f"{a}={h[:8]}" for a, h in hashes.items()))
        elif declared and any(hashes[a] not in rest.values() for a in declared):
            out.append(f"- {t}: {', '.join(sorted(declared))} uses constructor init by declaration (calibration); "
                       f"all other arms share init {next(iter(rest.values()))[:8]}")
    arms = json.loads((HERE / "arms.json").read_text())["arms"]
    for a, r in rows.items():
        spec = arms.get(a, {})
        want_const = spec.get("config", {}).get("lr_floor") == 1.0
        want_ams = bool(spec.get("recipe", {}).get("amsgrad"))
        for t, res in r["results"].items():
            rc = res.get("receipts") or {}
            if want_const and rc.get("lr_constant") is False:
                out.append(f"- {a}/{t}: LR not constant")
            if want_ams and rc.get("amsgrad_all") is False:
                out.append(f"- {a}/{t}: AMSGrad missing on some group")
            if not want_ams and any((v or {}).get("amsgrad") for v in (rc.get("lr") or {}).values()):
                out.append(f"- {a}/{t}: AMSGrad on unexpectedly")
    return out or ["- init identical across arms on every task; LR/AMSGrad receipts as declared"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = load(args.runs_dir)
    if not rows:
        print("no results")
        return
    board, order = table(rows)
    text = "\n".join(["# K3P constant-LR suite leaderboard", "",
                      "Pass per task as defined in run_suite.py; ring cells: prehold x/120, arrival per shift "
                      "(x = never), f = fails outside transit, d = departures. \"copied\" rows were not trained "
                      "separately: result.json copied from k3p_const_ams after a 400-update bit-exact check, or from runs_smoke/a where "
                      "the hashes differed (ab_nodirect two_pole); "
                      "k3p_simple (both removals) is the full-budget run that supports them.", "",
                      *board, "",
                      "## Per task (status, key metric)", "", *per_task(rows, order), "",
                      "## Runtime per task (s, wall under concurrency; * = result copied from another arm, "
                      "see result.json 'inferred')", "", *runtime(rows, order), "",
                      "## Receipt checks", "", *checks(rows), ""])
    print(text)
    if args.write:
        (HERE / "LEADERBOARD.md").write_text(text)


if __name__ == "__main__":
    main()

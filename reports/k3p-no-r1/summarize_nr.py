"""Leaderboard for the no-R1 factorial: runs/<arm>/<task>/result.json plus the read-only reference arms.

Usage: summarize_nr.py [--runs-dir runs] [--tasks screen|all] [--write]   (--write -> LEADERBOARD.md)
Rank: passes on the chosen tasks, then ring fails outside transit (shift + multishift), then native gate checks
(coverage + accuracy terminal checks /30), then native within-mode covariance error.
Native cell: P/F cov/acc checks | final HQ | per-mode cov eig ratio min-max (gate .4-1.7) | worst center RMS/sigma.
Covariance error = mean over natives of max(|ln min eig ratio|, |ln max eig ratio|) (0 = exact within-mode shape).
Receipt gate: an nr cell whose nr_receipt shows the arm's override did not engage is marked 'X <reason>' and does not
count as a pass (reference arms have no nr receipt and are not gated).
"""
import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_ROOT = Path("/home/martyn/dev/ParticleGAN/.claude/worktrees/k3p-constant-fix/reports/k3p-constant-suite")
REFS = {"gs2_c03_lr2_d05": REF_ROOT / "runs_grid", "k3p_simple": REF_ROOT / "runs", "k3p_stock": REF_ROOT / "runs"}
NATIVE = ("grid100", "rotated100", "staggered100")
ARMS = json.loads((HERE / "arms.json").read_text())["arms"]
if (HERE / "arms_e1.json").exists():
    ARMS.update(json.loads((HERE / "arms_e1.json").read_text())["arms"])
SCREEN = ["native-grid100", "native-rotated100", "native-staggered100", "toy-img_bars4", "toy-two_pole",
          "ring8-shift", "ring8-multishift"]


def results(root, arm):
    return {f.parent.name: json.loads(f.read_text()) for f in (root / arm).glob("*/result.json")}


def receipt_ok(arm, r):
    """Reasons the task's nr_receipt contradicts arms.json[arm] (empty list = the arm engaged as specified)."""
    rec = (r or {}).get("nr_receipt")
    if not rec:
        return ["no nr_receipt"]
    nr, bad = ARMS[arm]["nr"], []
    spike, settle = nr["spike"], nr["settle"]
    if (rec.get("spike"), rec.get("settle")) != (spike, settle):
        bad.append(f"arm {rec.get('spike')}/{rec.get('settle')}")
    if not rec.get("penalty_patched") or not rec.get("nr_penalty_calls"):
        bad.append("nr penalty not called")
    if rec.get("r1_evals", -1) != 0 or rec.get("stock_k3p_penalty_calls", -1) != 0:
        bad.append(f"R1 evals {rec.get('r1_evals')} stock {rec.get('stock_k3p_penalty_calls')}")
    if set(rec.get("d_loss_calls") or {}) != {"rp_hinge" if settle == "hinge" else "rp_softplus"}:
        bad.append(f"d_loss {rec.get('d_loss_calls')}")
    if (rec.get("oadam_steps", 0) > 0) != (settle == "oadam") or rec.get("oadam_steps_non_critic", -1) != 0:
        bad.append(f"oadam steps {rec.get('oadam_steps')} (non-critic {rec.get('oadam_steps_non_critic')})")
    want_rule = "lib.oadam.OptimisticAdam.step" if settle == "oadam" else "torch.optim.Adam.step"
    rules = [c.get("update_rule") for c in rec.get("critic_optimizers") or []]
    if not rules or any(u != want_rule for u in rules):
        bad.append(f"update rule {rules}")
    if set(rec.get("guard_reads") or {}) - {nr["guard_buffer"]}:
        bad.append(f"guard read {rec.get('guard_reads')}")
    if settle == "emar1":
        e1, recipe = rec.get("emar1") or {}, ARMS[arm]["recipe"]
        want = dict(beta=nr["beta"], decay=[recipe["reg_anchor_decay"]], weight=[recipe["reg_anchor_weight"]])
        got = {k: e1.get(k) for k in want}
        if got != want:
            bad.append(f"emar1 {got} != {want}")
        if not e1.get("anchor_started") or not all(e1["anchor_started"]):
            bad.append(f"anchor_started {e1.get('anchor_started')}")
    if bool(rec.get("anchor_active")) != (settle in ("anchor", "emar1")):
        bad.append(f"anchor_active {rec.get('anchor_active')}")
    if rec.get("pathcap_pair_mismatch", 0) != 0:
        bad.append(f"pathcap mismatch {rec['pathcap_pair_mismatch']}")
    if (rec.get("lr") or {}).get("lr_constant") is not True:
        bad.append("lr not constant")
    if rec.get("nonfinite_terms"):
        bad.append("nonfinite terms")
    return bad


def native(root, arm, p):
    b = root / arm / f"native-{p}" / "bench"
    try:
        g = json.loads((b / f"gate-{p}.json").read_text())["problems"][p]
        a = json.loads((b / f"accuracy-gate-{p}.json").read_text())["problems"][p]
    except (FileNotFoundError, KeyError):
        return None
    m = re.search(r"only (\d+)/5", g.get("reason", ""))
    f, tc = g.get("final_metrics") or {}, a.get("terminal_checks") or []
    if not f:  # shortened (smoke) run: the gate never reached its terminal window
        f = dict(hq=float("nan"), min_cov_eig_ratio=float("nan"), max_cov_eig_ratio=float("nan"))
    return dict(passed=g["status"] == "PASS" and a["status"] == "PASS",
                cov=5 if g["status"] == "PASS" else int(m.group(1)) if m else 0,
                acc=sum(c["passed"] for c in tc), hq=f["hq"], emin=f["min_cov_eig_ratio"], emax=f["max_cov_eig_ratio"],
                center=max((c["metrics"]["center_rms_sigma"] or float("inf")) for c in tc) if tc else float("nan"))


def ncell(n):
    if n is None:
        return "–"
    return f"{'P' if n['passed'] else 'F'} {n['cov']}/{n['acc']} {n['hq']:.3f} {n['emin']:.2f}-{n['emax']:.2f} c{n['center']:.2f}"


def ring(r):
    s = ((r or {}).get("raw") or {}).get("score") or {}
    if not s:
        return "–" if r is None else r.get("status", "?")
    arr = "/".join("x" if g["arrival"] is None else f"+{g['arrival']}" for g in s["segments"])
    return f"{'P' if r['passed'] else 'F'} f{s['fails_outside_transit']} {arr} d{s['departures']}"


def row(root, arm, tasks):
    res = results(root, arm)
    gate = {t: receipt_ok(arm, res[t]) for t in tasks if t in res} if arm in ARMS else {}
    ns = {p: native(root, arm, p) for p in NATIVE if f"native-{p}" in tasks and not gate.get(f"native-{p}")}
    ok = [n for n in ns.values() if n]
    rfail = sum((((res.get(t) or {}).get("raw") or {}).get("score") or {}).get("fails_outside_transit", 10 ** 4)
                for t in ("ring8-shift", "ring8-multishift") if t in tasks and not gate.get(t))
    coverr = (sum(max(abs(math.log(max(n["emin"], 1e-9))), abs(math.log(max(n["emax"], 1e-9)))) for n in ok) / len(ok)
              if ok and all(n["emin"] == n["emin"] for n in ok) else float("nan"))
    cells = []
    for t in tasks:
        if gate.get(t):
            cells.append("X " + "; ".join(gate[t])[:60])
            continue
        if t.startswith("native-"):
            cells.append(ncell(ns.get(t.split("-", 1)[1])))
        elif t.startswith("ring8"):
            cells.append(ring(res.get(t)))
        else:
            r = res.get(t)
            cells.append("–" if r is None else f"{'P' if r.get('passed') else 'F'} {(r.get('metric') or {}).get('value')}")
    inert = [t for t in tasks if ((res.get(t) or {}).get("nr_receipt") or {}).get("pairsec_no_neighbor")]
    return dict(arm=arm, passes=sum(res[t].get("passed") is True and not gate.get(t) for t in tasks if t in res),
                inert=inert, gated=sum(bool(v) for v in gate.values()),
                done=sum(t in res for t in tasks), rfail=rfail, checks=sum(n["cov"] + n["acc"] for n in ok),
                coverr=coverr, cells=cells)


def table(runs, tasks):
    arms = sorted(p.name for p in runs.glob("*/") if any(p.glob("*/result.json")))
    rows = [row(runs, a, tasks) for a in arms] + [row(REFS[a], a, tasks) | {"arm": f"{a} (ref)"} for a in REFS]
    rows.sort(key=lambda r: (-r["passes"], r["rfail"], -r["checks"], r["coverr"] if r["coverr"] == r["coverr"] else 9))
    short = [t.replace("native-", "").replace("toy-", "") for t in tasks]
    lines = [f"| # | arm | passes | ring fails | native checks /30 | cov err | " + " | ".join(short) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(tasks)]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['arm']} | {r['passes']}/{r['done']} | {r['rfail'] if r['rfail'] < 10 ** 4 else '–'} | "
                     f"{r['checks']} | {r['coverr']:.2f} | " + " | ".join(r["cells"]) + " |")
    gated = [f"{r['arm']} ({r['gated']})" for r in rows if r.get("gated")]
    if gated:
        lines.append(f"\nReceipt-gated cells (X, not counted as passes): {', '.join(gated)}")
    inert = [f"{r['arm']}: {', '.join(r['inert'])}" for r in rows if r.get("inert")]
    if inert:
        lines.append(f"\npairsec dropped rows with no distinct real neighbour (see pairsec_no_neighbor): {'; '.join(inert)}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--tasks", default="screen")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(HERE.parent / "k3p-constant-suite"))  # this worktree's copy (REF_ROOT is read-only)
    from run_suite import TASKS
    tasks = SCREEN if a.tasks == "screen" else list(TASKS) if a.tasks == "all" else a.tasks.split(",")
    text = "\n".join(table(Path(a.runs_dir).resolve(), tasks))
    print(text)
    if a.write:
        (HERE / "LEADERBOARD.md").write_text("# No-R1 leaderboard\n\nGenerated by `summarize_nr.py`. " + __doc__.split(
            "\n", 3)[3] + "\n" + text + "\n")


if __name__ == "__main__":
    main()

"""EMA-centred R1 (e1_*) screen table vs gs2 / nr_none_anchor / k3p_stock, from result.json + nr_receipt.json.

Usage: summarize_e1.py [--tasks screen|all]
Columns: native final HQ per task | worst per-mode cov eig ratio over the 3 natives (gate >= .4) | native passes |
img_bars4 and two_pole passing-suffix cells /24 | ring fails outside transit (shift+multishift), first arrivals,
departures | max real RMS slope ||grad D(r)||/sqrt d seen by the penalty (receipt; natives / rings) | receipt gate.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "k3p-constant-suite")]
import summarize_nr as sm  # noqa: E402

NAT = ("grid100", "rotated100", "staggered100")
REF = {"gs2_c03_lr2_d05": sm.REFS["gs2_c03_lr2_d05"], "k3p_stock": sm.REFS["k3p_stock"],
       "nr_none_anchor": HERE / "runs"}


def rms_max(res, tasks):
    v = [((res.get(t) or {}).get("nr_receipt") or {}).get("emar1", {}).get("real_rms_max") for t in tasks]
    v = [x for x in v if x is not None]
    return f"{max(v):.2f}" if v else "n/a"


def arm_row(root, arm, tasks):
    res = sm.results(root, arm)
    gated = {t: sm.receipt_ok(arm, res[t]) for t in res if t in tasks} if arm in sm.ARMS else {}
    nat = {p: sm.native(root, arm, p) for p in NAT}
    hq = "/".join("–" if n is None else f"{n['hq']:.3f}" for n in nat.values())
    ok = [n for n in nat.values() if n]
    worst = min(n["emin"] for n in ok) if ok else float("nan")
    maxe = max(n["emax"] for n in ok) if ok else float("nan")
    npass = sum(n["passed"] for n in ok)

    def toy(t):
        r = res.get(t)
        return "–" if r is None else f"{'P' if r.get('passed') else 'F'}{(r.get('metric') or {}).get('value')}"
    fails, arr, dep = 0, [], 0
    for t in ("ring8-shift", "ring8-multishift"):
        s = (((res.get(t) or {}).get("raw") or {}).get("score")) or {}
        if not s:
            fails, arr = None, ["?"]
            break
        fails += s["fails_outside_transit"]
        dep += s["departures"]
        arr.append("/".join("x" if g["arrival"] is None else str(g["arrival"]) for g in s["segments"]))
    passes = sum(res[t].get("passed") is True and not gated.get(t) for t in tasks if t in res)
    return dict(arm=arm, done=sum(t in res for t in tasks), passes=passes, hq=hq, worst=worst, maxe=maxe, npass=npass,
                bars=toy("toy-img_bars4"), pole=toy("toy-two_pole"), fails=fails, arr=" ".join(arr), dep=dep,
                rms_nat=rms_max(res, [f"native-{p}" for p in NAT]), rms_ring=rms_max(res, ["ring8-shift", "ring8-multishift"]),
                gate=sum(bool(v) for v in gated.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="screen")
    a = ap.parse_args()
    from run_suite import TASKS
    tasks = sm.SCREEN if a.tasks == "screen" else list(TASKS)
    arms = sorted(p.name for p in (HERE / "runs").glob("e1_*/") if any(p.glob("*/result.json")))
    rows = [arm_row(REF[r], r, tasks) | {"arm": f"{r} (ref)"} for r in REF] + [arm_row(HERE / "runs", x, tasks)
                                                                               for x in arms]
    print(f"| arm | passes | native HQ g/r/s | worst cov (max) | native P | img_bars4 | two_pole | ring fails | "
          f"arrivals shift ; multi | dep | max real RMS nat/ring | gated |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['arm']} | {r['passes']}/{r['done']} | {r['hq']} | {r['worst']:.3f} ({r['maxe']:.2f}) | {r['npass']}/3 | "
              f"{r['bars']} | {r['pole']} | {r['fails']} | {r['arr']} | {r['dep']} | {r['rms_nat']} / {r['rms_ring']} | "
              f"{r['gate']} |")


if __name__ == "__main__":
    main()

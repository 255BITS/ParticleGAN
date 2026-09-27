#!/usr/bin/env python
"""Leaderboard for results/sparse-secant/<arm>/ (study metrics + D diagnostics). Prints markdown."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "sparse-secant"
REF = ROOT.parent / "sparse-ucd" / "results" / "sparse" / "runs" / "champion" / "l0p02_gw_sp0p003_s1"

FORM = {
    "champ_ref_anneal": "REF: champion s1 as archived (RpGAN + g_interp_cap(1), cosine LR anneal from 60% to 5%)",
    "champ_matched": "RpGAN + g_interp_cap(1) [constant LRs]",
    "sec_nodamp": "wgan + r1(1) + secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] (= secant_r1_b2 here: no A2 damping in this harness)",
    "sec_rpbase": "RpGAN + r1(1) + secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9]",
    "sec_lazy4": "wgan + r1(1) + secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, lazy_k=4]",
}


def load(d: Path):
    s = json.loads((d / "summary.json").read_text())
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return s, rows


def fmt(v, p=3):
    return "n/a" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def main() -> None:
    arms = [("champ_ref_anneal", REF)] + [(a, RES / a) for a in FORM if a != "champ_ref_anneal"]
    out = []
    for name, d in arms:
        if not (d / "summary.json").exists():
            continue
        s, rows = load(d)
        f = s["final"]
        on_bar = sum(bool(r.get("bar_all")) for r in rows)
        after = [r for r in rows if s["bar_step"] is not None and r["step"] >= s["bar_step"]]
        held = sum(bool(r.get("bar_all")) for r in after)
        last = rows[-1]
        dd = s.get("d_diag", {})
        out.append({
            "arm": name, "bar": "PASS" if s["bar_held"] else "fail", "bar_step": s["bar_step"],
            "held": f"{held}/{len(after)}" if after else "0/0", "on_bar": on_bar,
            "modes": f["modes"], "hq": f["hq"], "cond": f["cond_acc"], "sep": f["cond_sep_ratio"],
            "sym": f["sym_acc_mode"], "joint": f["joint_acc"], "sp2": f["sparse_prec@0p01"],
            "sp3": f["sparse_prec@0p001"], "zero": f["exact_zero_frac"], "core": f.get("core_ratio"),
            "w1": f["sliced_w1"], "ucdF": f.get("ucd_acc_fake"),
            "Dabs": dd.get("max_abs_D_real"), "gnmax": dd.get("max_grad_norm_eval"),
            "gap": last.get("D_gap"), "gni": last.get("gni_med"), "sps": s["steps_per_sec"],
            "min_modes_2nd_half": min(r["modes"] for r in rows if r["step"] >= s["steps"] // 2),
        })
    ref = [r for r in out if r["arm"].startswith("champ_ref")]
    rest = sorted((r for r in out if not r["arm"].startswith("champ_ref")),
                  key=lambda r: (r["bar"] != "PASS", -r["on_bar"], -r["modes"], -r["joint"] * r["hq"]))
    print("# Sparse-UCD: simple-critic transfer (seed 1, 5000 steps, constant LRs, no instance noise)\n")
    print("Ranked by: bar held at end, then evals on bar (of 50), then modes, then joint*hq. "
          "`held` = evals on bar / evals since first crossing. D diagnostics: max |d(real)[c]| over every "
          "training step; max ||grad D|| over real/fake/interp at the 50 eval steps; final D gap = mean d(r) - d(f); "
          "gn_i = final median ||grad D|| on interpolates.\n")
    cols = ["#", "arm", "formulation", "bar", "bar_step", "held", "on_bar", "modes", "min modes 2nd half", "hq", "cond",
            "sep", "sym", "joint", "sp@1e-2", "sp@1e-3", "zero", "core", "w1", "ucdF",
            "max abs D(real)", "max grad-norm", "final D gap", "gn_i", "steps/s"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for i, r in enumerate(ref + rest):
        rank = "ref" if r in ref else str(i + 1 - len(ref))
        vals = [rank, r["arm"], FORM[r["arm"]], r["bar"], fmt(r["bar_step"]), r["held"], str(r["on_bar"]),
                str(r["modes"]), str(r["min_modes_2nd_half"]), fmt(r["hq"]), fmt(r["cond"]), fmt(r["sep"], 2),
                fmt(r["sym"]), fmt(r["joint"]), fmt(r["sp2"]), fmt(r["sp3"]), fmt(r["zero"], 2), fmt(r["core"], 2),
                fmt(r["w1"]), fmt(r["ucdF"], 2), fmt(r["Dabs"], 2), fmt(r["gnmax"], 2), fmt(r["gap"], 2),
                fmt(r["gni"], 2), fmt(r["sps"], 1)]
        print("| " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()

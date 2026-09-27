"""Leaderboard for the 100-Gaussian transfer study: python3 summarize.py [--md] [--runs-dir runs_oldinit]."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = ("grid100", "rotated100", "staggered100")
ARCHIVE = HERE.parents[2] / "reports/toy100/simpler22/toy100"
RUNS = HERE / (sys.argv[sys.argv.index("--runs-dir") + 1] if "--runs-dir" in sys.argv else "runs")


def load():
    rows = []
    for f in sorted(RUNS.glob("*/result.json")):
        r = json.loads(f.read_text())
        rows.append(r)
    return rows


def median(v):
    v = sorted(v)
    return v[len(v) // 2]


def late(arm, problem, key, agg):
    rows = [json.loads(l) for l in (RUNS / arm / "diag" / f"{problem}.jsonl").open()]
    return agg([r[key] for r in rows if r["step"] >= 1000])


def fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def main():
    rows = load()
    def key(r):
        ps = r["problems"].values()
        return (r["arm"].startswith("ref"),
                -sum(p["final"].get("passed") is True for p in ps),
                -sum(p["evals_passed_after_1000"] for p in ps),
                -sum(p["final"].get("hq") or 0 for p in ps))
    rows.sort(key=key)
    out = ["| # | arm | formulation | gate | accuracy | final live pass | final modes (g/r/s) | final HQ (g/r/s) | "
           "first 100 modes | passing live evals >=1000 | final min-max cov eig ratio | max abs D(real) | max grad-norm (all / >=1000) | median gmax | g(real) median >=1000 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rank = 0
    for r in rows:
        ps = [r["problems"][p] for p in P]
        ref = r["arm"].startswith("ref")
        if not ref:
            rank += 1
        out.append("| " + " | ".join([
            "ref" if ref else str(rank), r["arm"], r["formulation"], str(r["gate_status"]), str(r["accuracy_status"]),
            f"{sum(p['final'].get('passed') is True for p in ps)}/3",
            "/".join(str(p["final"].get("modes")) for p in ps),
            "/".join(fmt(p["final"].get("hq")) for p in ps),
            "/".join(str(p["first_full_coverage"]) for p in ps),
            ", ".join(f"{p['evals_passed_after_1000']}/{p['evals_after_1000']}" for p in ps),
            ", ".join(f"{p['final'].get('min_cov_eig_ratio'):.2f}-{p['final'].get('max_cov_eig_ratio'):.2f}" for p in ps),
            fmt(max(p["max_abs_D_real"] for p in ps), ".2f"),
            fmt(max(p["max_grad_norm"] for p in ps), ".2f") + " / " + fmt(max(late(r["arm"], q, "g_max", max) for q in P), ".2f"),
            "/".join(fmt(p["median_grad_norm_max"], ".2f") for p in ps),
            "/".join(fmt(late(r["arm"], q, "g_real", median), ".3f") for q in P),
        ]) + " |")
    # archived CPU reference
    arch = []
    for p in P:
        s = json.loads((ARCHIVE / p / "summary.json").read_text())
        arch.append(s)
    out.append("| ref | ref:archived-cpu | REF: shipped default, archived CPU run (reports/toy100/simpler22) | PASS | PASS | 3/3 | "
               + "/".join(str(s["final"]["live"]["modes"]) for s in arch) + " | "
               + "/".join(f"{s['final']['live']['hq']:.3f}" for s in arch) + " | "
               + "/".join(str(s["first_full_coverage_step"]["live"]) for s in arch) + " | n/a | "
               + ", ".join(f"{s['final']['live']['min_cov_eig_ratio']:.2f}-{s['final']['live']['max_cov_eig_ratio']:.2f}" for s in arch)
               + " | n/a | n/a | n/a | n/a |")
    print("\n".join(out))
    if "--detail" in sys.argv:
        for r in rows:
            for p in P:
                d = r["problems"][p]
                f = d["final"]
                print(r["arm"], p, d["status"], "stable", d["stable_pass"], "tv", fmt(f.get("mass_tv")),
                      "cov", fmt(f.get("min_cov_eig_ratio"), ".2f"), fmt(f.get("max_cov_eig_ratio"), ".2f"),
                      "rad", fmt(f.get("min_radial_median_ratio"), ".2f"), fmt(f.get("max_radial_median_ratio"), ".2f"),
                      "train_s", fmt(d["train_seconds"], ".0f"), d["error"] or "")


if __name__ == "__main__":
    main()

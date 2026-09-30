#!/usr/bin/env python
"""Leaderboards for the MisGAN toy (experiments/misgan_toy.py).

One table per mechanism, ranked by imputation mode accuracy (with its gap to
the Bayes-optimal sampled imputer), then generation 3-sigma %, then sliced W1.
The untrained baselines (Bayes-optimal posterior sampler, kNN, mean) are
computed here on the same fixed test rows. `ppost:<run>` rows and the per-source
ppost tables come from experiments/misgan_ppost.py (posthoc.json). Writes the tables and automatic
comparisons into reports/misgan-toy/FINDINGS.md between the AUTO markers and
keeps any hand-written text outside them.

    python experiments/analyze_misgan.py [--runs results/misgan/runs]
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from lib.misgan import MECHANISMS, Problem, baselines  # noqa: E402

START, END = "<!-- AUTO:start -->", "<!-- AUTO:end -->"
COLUMNS = ("arm", "acc", "gap", "acc_lo", "itv", "istd", "rmse", "ms_1k", "ihq", "modes", "hq", "swd", "off",
           "m_mae", "m_tv", "m_soft", "acc_peak")
PPOST_MAIN = "M4096"  # the ppost configuration shown in the main leaderboard (declared before the runs)
PPOST_CONFIGS = ("M256", "M256r", "M4096", "M4096r")


def fmt(value, key):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, str):
        return value
    if key in ("modes", "imodes"):
        return str(int(value))
    if key in ("hq", "ihq", "ms_1k"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def load_runs(runs_dir):
    runs = {}
    for path in sorted(Path(runs_dir).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        posthoc = path.parent / "posthoc.json"
        summary["posthoc"] = json.loads(posthoc.read_text()) if posthoc.exists() else {}
        label = path.parent.name.split("__", 1)[-1]  # the arm, or a follow-up label
        runs.setdefault(summary["config"]["mechanism"], {})[label] = summary
    return runs


def markdown(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(row.get(c), c) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def table(mech, arms, base):
    bayes_acc = base["bayes"]["acc"]
    tv_key = "m_tv" if mech == "block" else "m_tvk"
    rows = []
    for arm, summary in arms.items():
        final = dict(summary["final"])
        final["arm"], final["gap"] = arm, bayes_acc - final["acc"]
        final["m_tv"] = final.get(tv_key, final.get("m_tv"))
        final["acc_peak"] = max(h["acc"] for h in summary["history"])
        final["ms_1k"] = summary["posthoc"].get("cost_ms_1k")
        rows.append(final)
        ppost = summary["posthoc"].get("ppost", {}).get(PPOST_MAIN)
        if ppost:  # G_x imputes by particle posterior; generation columns are the source G_x's
            gen = {k: final[k] for k in ("modes", "hq", "swd", "off")}
            rows.append({**ppost, **gen, "arm": f"ppost:{arm}", "gap": bayes_acc - ppost["acc"]})
    rows.sort(key=lambda r: (-round(r["acc"], 3), -r["hq"], r["swd"]))
    for name in ("bayes", "knn", "mean"):
        rows.append({**base[name], "arm": f"*{name}*", "gap": bayes_acc - base[name]["acc"]})
    floor = base["floor"]
    rows.append({"arm": "*clean sample*", "modes": floor["modes"], "hq": floor["hq"], "swd": floor["swd"],
                 "off": floor["off"]})
    note = (f"Bayes MAP accuracy {base['bayes']['acc_map']:.3f}. `ppost:<run>` rows impute with that run's "
            f"G_x by particle posterior ({PPOST_MAIN}, no refinement). ms_1k = wall-clock ms per 1k rows for "
            "16 draws. Mask TV column = "
            + ("TV over the 4 block patterns (256-pattern space)." if mech == "block"
               else "TV of the observed-count histogram vs Binomial(8, 1-p)."))
    return markdown(rows, COLUMNS), note, {r["arm"]: r for r in rows}


def ppost_table(mech, arms, base):
    """One row per source G_x: acc / acc_lo / itv for each (M, refine) configuration."""
    rows = []
    for arm, summary in arms.items():
        pp = summary["posthoc"].get("ppost")
        if not pp:
            continue
        row = {"source G_x": arm, "hq": summary["final"]["hq"], "modes": summary["final"]["modes"],
               "sigma": pp[PPOST_MAIN]["sigma"], "istd": pp[PPOST_MAIN]["istd"]}
        for name in PPOST_CONFIGS:
            row[name] = f"{pp[name]['acc']:.3f} / {pp[name]['acc_lo']:.3f} / {pp[name]['itv']:.3f}"
        row["ms_1k M4096 / r"] = f"{pp['M4096']['ms_1k']:.1f} / {pp['M4096r']['ms_1k']:.1f}"
        row["own imputer"] = (f"{summary['final']['acc']:.3f} / {summary['final']['acc_lo']:.3f} / "
                              f"{summary['final']['itv']:.3f}")
        rows.append(row)
    if not rows:
        return ""
    b = base["bayes"]
    rows.append({"source G_x": "*bayes*", **{name: f"{b['acc']:.3f} / {b['acc_lo']:.3f} / {b['itv']:.3f}"
                                             for name in PPOST_CONFIGS}, "istd": b["istd"]})
    columns = ("source G_x", "modes", "hq", "sigma", *PPOST_CONFIGS, "istd", "ms_1k M4096 / r", "own imputer")
    return markdown(rows, columns)


def comparisons(tables):
    """Automatic one-line contrasts; the narrative is written by hand."""
    out = []
    for mech, rows in tables.items():
        if "misgan" not in rows:
            continue
        ref = rows["misgan"]
        bayes = rows["*bayes*"]["acc"]
        parts = [f"misgan acc {ref['acc']:.3f} vs Bayes {bayes:.3f} (gap {bayes - ref['acc']:+.3f})"]
        for arm in ("oracle", "zerofill", "misgan_realmask", "misgan_paired", "misgan_gauss", "misgan_hard",
                    "misgan_detach", "misgan_long", "aegan_recon_w0.1", "aegan_recon_w1", "aegan_ce"):
            if arm in rows:
                r = rows[arm]
                parts.append(f"{arm} dacc {r['acc'] - ref['acc']:+.3f} ditv {r['itv'] - ref['itv']:+.3f} "
                             f"dhq {r['hq'] - ref['hq']:+.1f}")
        out.append(f"- **{mech}**: " + "; ".join(parts))
    return "\n".join(out)


def predictions(runs, tables):
    """The numbers each stated prediction is judged on (verdicts are written by hand)."""
    out = []

    def pp(mech, label, name=PPOST_MAIN):
        return runs[mech].get(label, {}).get("posthoc", {}).get("ppost", {}).get(name)

    for mech, rows in tables.items():
        b = rows["*bayes*"]
        for w in ("aegan_recon_w0.1", "aegan_recon_w1"):
            if w not in rows:
                continue
            r, own = rows[w], pp(mech, w)
            line = (f"- {mech} {w}: (a) acc_lo {r['acc_lo']:.3f} vs Bayes {b['acc_lo']:.3f}; "
                    f"(b) itv/istd {r['itv']:.3f}/{r['istd']:.3f}")
            if own:
                line += f" vs ppost on its G_x {own['itv']:.3f}/{own['istd']:.3f}"
            line += f"; (c) G_x hq {r['hq']:.1f}, modes {r['modes']}"
            out.append(line)
        for src in ("oracle", "misgan_detach", "misgan_realmask", "misgan"):
            if src in rows and pp(mech, src):
                gi, p = rows[src], pp(mech, src)
                closed = ((p["acc_lo"] - gi["acc_lo"]) / (b["acc_lo"] - gi["acc_lo"])
                          if b["acc_lo"] != gi["acc_lo"] else float("nan"))
                out.append(f"- {mech} (d) ppost on {src}: acc_lo {p['acc_lo']:.3f} (own G_i {gi['acc_lo']:.3f}, "
                           f"Bayes {b['acc_lo']:.3f}; share of gap closed {closed:.2f}); itv {p['itv']:.3f} "
                           f"vs {gi['itv']:.3f}")
        if "aegan_ce" in rows and pp(mech, "aegan_ce"):
            r, p = rows["aegan_ce"], pp(mech, "aegan_ce")
            out.append(f"- {mech} (e) aegan_ce acc/acc_lo/itv {r['acc']:.3f}/{r['acc_lo']:.3f}/{r['itv']:.3f} "
                       f"at {r['ms_1k']:.1f} ms/1k vs ppost on its G_x {p['acc']:.3f}/{p['acc_lo']:.3f}/"
                       f"{p['itv']:.3f} at {p['ms_1k']:.1f} ms/1k")
        for src in ("misgan", "misgan_gauss", "oracle"):
            small, large = pp(mech, src, "M256"), pp(mech, src, "M4096")
            if small and large:
                out.append(f"- {mech} (f) {src}: ppost acc M256 {small['acc']:.3f} -> M4096 {large['acc']:.3f}; "
                           f"acc_lo {small['acc_lo']:.3f} -> {large['acc_lo']:.3f}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="results/misgan/runs")
    parser.add_argument("--findings", default="reports/misgan-toy/FINDINGS.md")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        parser.error(f"no summaries under {args.runs}")
    sections, tables = [], {}
    for mech in MECHANISMS:
        if mech not in runs:
            continue
        cfg = next(iter(runs[mech].values()))["config"]
        problem = Problem(mech, cfg["n_train"], cfg["n_test"], cfg["data_seed"], args.device)
        base = baselines(problem, cfg["impute_draws"])
        text, note, rows = table(mech, runs[mech], base)
        tables[mech] = rows
        extra = ppost_table(mech, runs[mech], base)
        if extra:
            extra = ("\nParticle-posterior imputation by source G_x. Cells: acc / acc_lo / itv; `r` = with "
                     f"latent refinement; sigma and istd at {PPOST_MAIN}.\n\n" + extra + "\n")
        print(f"\n== {mech} ==\n{text}\n{note}\n{extra}")
        sections.append(f"### {mech}\n\n{text}\n\n{note}\n{extra}")
    auto, checks = comparisons(tables), predictions(runs, tables)
    print("\n" + auto + "\n\n" + checks)
    body = (f"{START}\n## Leaderboards\n\nFinal EMA weights at the last update. Ranked by imputation "
            "mode accuracy, then generation 3-sigma %, then sliced W1. Italic rows are untrained "
            "references on the same test rows.\n\n" + "\n".join(sections)
            + f"\n## Automatic comparisons (vs `misgan`)\n\n{auto}\n"
            + f"\n## Prediction checks (numbers)\n\n{checks}\n{END}")
    path = Path(args.findings)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and START in (old := path.read_text()) and END in old:
        text = old[: old.index(START)] + body + old[old.index(END) + len(END):]
    else:
        text = "# MisGAN toy: findings\n\nProtocol: [README.md](README.md).\n\n" + body + "\n"
    path.write_text(text)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

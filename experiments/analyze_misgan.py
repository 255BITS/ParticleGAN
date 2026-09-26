#!/usr/bin/env python
"""Leaderboards for the MisGAN toy (experiments/misgan_toy.py).

One table per mechanism, ranked by imputation mode accuracy (with its gap to
the Bayes-optimal sampled imputer), then generation 3-sigma %, then sliced W1.
The untrained baselines (Bayes-optimal posterior sampler, kNN, mean) are
computed here on the same fixed test rows. Writes the tables and automatic
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
COLUMNS = ("arm", "acc", "gap", "acc_lo", "itv", "istd", "rmse", "ihq", "modes", "hq", "swd", "off",
           "m_mae", "m_tv", "m_soft", "acc_peak")


def fmt(value, key):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, str):
        return value
    if key in ("modes", "imodes"):
        return str(int(value))
    if key in ("hq", "ihq"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def load_runs(runs_dir):
    runs = {}
    for path in sorted(Path(runs_dir).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        cfg = summary["config"]
        label = path.parent.name.split("__", 1)[-1]  # the arm, or a follow-up label
        runs.setdefault(cfg["mechanism"], {})[label] = summary
    return runs


def table(mech, arms, base):
    bayes_acc = base["bayes"]["acc"]
    tv_key = "m_tv" if mech == "block" else "m_tvk"
    rows = []
    for arm, summary in arms.items():
        final = dict(summary["final"])
        final["arm"], final["gap"] = arm, bayes_acc - final["acc"]
        final["m_tv"] = final.get(tv_key, final.get("m_tv"))
        final["acc_peak"] = max(h["acc"] for h in summary["history"])
        rows.append(final)
    rows.sort(key=lambda r: (-round(r["acc"], 3), -r["hq"], r["swd"]))
    for name in ("bayes", "knn", "mean"):
        row = {**base[name], "arm": f"*{name}*", "gap": bayes_acc - base[name]["acc"]}
        rows.append(row)
    floor = base["floor"]
    rows.append({"arm": "*clean sample*", "modes": floor["modes"], "hq": floor["hq"], "swd": floor["swd"],
                 "off": floor["off"]})
    head = "| " + " | ".join(COLUMNS) + " |"
    lines = [head, "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c), c) for c in COLUMNS) + " |")
    note = (f"Bayes MAP accuracy {base['bayes']['acc_map']:.3f}. Mask TV column = "
            + ("TV over the 4 block patterns (256-pattern space)." if mech == "block"
               else "TV of the observed-count histogram vs Binomial(8, 1-p)."))
    return "\n".join(lines), note, {r["arm"]: r for r in rows}


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
                    "misgan_detach", "misgan_long"):
            if arm in rows:
                r = rows[arm]
                parts.append(f"{arm} dacc {r['acc'] - ref['acc']:+.3f} ditv {r['itv'] - ref['itv']:+.3f} "
                             f"dhq {r['hq'] - ref['hq']:+.1f}")
        out.append(f"- **{mech}**: " + "; ".join(parts))
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
        text, note, rows = table(mech, runs[mech], baselines(problem, cfg["impute_draws"]))
        tables[mech] = rows
        print(f"\n== {mech} ==\n{text}\n{note}")
        sections.append(f"### {mech}\n\n{text}\n\n{note}\n")
    auto = comparisons(tables)
    print("\n" + auto)
    body = (f"{START}\n## Leaderboards\n\nFinal EMA weights at the last update. Ranked by imputation "
            "mode accuracy, then generation 3-sigma %, then sliced W1. Italic rows are untrained "
            "references on the same test rows.\n\n" + "\n".join(sections)
            + f"\n## Automatic comparisons (vs `misgan`)\n\n{auto}\n{END}")
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

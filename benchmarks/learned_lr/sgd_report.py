"""Render the raw-SGD research artifacts without changing selection."""
import argparse
import json
from pathlib import Path


def render(source):
    source = Path(source)
    sweep = json.loads((source / "sweep.json").read_text())
    def values(row):
        live = row.get("live", {})
        return (f"{live.get('modes', '—')}/{live.get('n_modes', '—')} / "
                + (f"{live['hq']:.2%}" if "hq" in live else "—"))
    lines = ["# Plain SGD and gradient-feedback feasibility", "",
             "The inner optimizer is raw SGD: no momentum, weight decay, clipping, or coordinate scaling. "
             "Independent G/prior and D learning rates are tuned on the existing ring4 and grid9 training distributions. "
             "Every episode uses seed 0 and 1,200 updates. Live weights determine every score; EMA is retained separately in the episode files.", "",
             "The objective puts sustained full coverage/HQ first: 20×no sustained pass, plus final/last-five missing-mode and HQ deficits, "
             "a small normalized SW1 term, and observed convergence progress. A sustained pass requires every mode, HQ ≥90%, "
             "and at least five consecutive passing observations through the final checkpoint. Errors/incomplete runs score 1000.", "",
             "| Best configuration | G LR | D LR | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Sustained tasks | Errors |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    selections = [("SGD constant", sweep.get("best_constant")), ("SGD delayed cosine", sweep.get("best_cosine"))]
    if len(sweep.get("controls", [])) == 2:
        selections.append(("Adam delayed cosine", dict(g_lr=.0017, d_lr=.0017,
            objective=sum(e["objective"] for e in sweep["controls"]) / 2, episodes=sweep["controls"])))
    fit = json.loads((source / "fit.json").read_text()) if (source / "fit.json").exists() else None
    if fit and fit.get("best"):
        nonzero = any(v != 0 for row in fit["best"]["weights"] for v in row)
        selections.append(("SGD learned feedback" if nonzero else "SGD search winner (constant)",
                           {**fit["best"], "g_lr": fit["base_rates"]["g"], "d_lr": fit["base_rates"]["d"]}))
    for name, row in selections:
        if row is None:
            continue
        a, b = row["episodes"]
        stable = sum(e["convergence"]["stable_from_step"] is not None for e in row["episodes"])
        errors = sum(e["status"] != "COMPLETE" for e in row["episodes"])
        lines.append(f"| {name} | {row['g_lr']:g} | {row['d_lr']:g} | {row['objective']:.4f} | {values(a)} | {values(b)} | {stable}/2 | {errors} |")
    rows = sweep["rows"]
    errors = sum(e["status"] != "COMPLETE" for r in rows for e in r["episodes"])
    seconds = sum(e["seconds"] for r in rows for e in r["episodes"])
    lines += ["", f"The sweep completed {len(rows)} schedule/rate settings ({2 * len(rows)} episodes); "
              f"{errors} episodes failed with recorded exceptions. Observed episode time totals {seconds:.1f} CPU seconds. "
              "No failed configuration is omitted from the ranking below.", "",
              "| Configuration | G LR | D LR | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Error episodes |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda r: r["objective"]):
        a, b = row["episodes"]
        error_count = sum(e["status"] != "COMPLETE" for e in row["episodes"])
        lines.append(f"| {row['name']} | {row['g_lr']:g} | {row['d_lr']:g} | {row['objective']:.4f} | {values(a)} | {values(b)} | {error_count} |")
    if fit:
        proposals = fit["proposals"]
        errors = sum(e["status"] != "COMPLETE" for r in proposals for e in r["episodes"])
        lines += ["", f"The scalar-feedback search evaluated {len(proposals)} policies on both training tasks; "
                  f"{errors} fitting episodes failed. Every policy and episode is retained in `fit.json` and `episodes/`."]
    if (source / "layer_diagnostics.json").exists():
        lines += ["", "A read-only layer diagnostic exactly reproduces every metric of the selected constant SGD runs. "
                  "At initialization, the generator's first weight matrix changes by about 0.0095% of its RMS, while "
                  "its output bias changes by 25.8% and its output weights by 1.24%. A common scalar LR preserves these "
                  "relative disparities for a given gradient. This motivates the separate per-tensor study; it is not "
                  "a proof of what caused collapse. Full measurements are in `layer_diagnostics.json`."]
    decision_file = source / "decision.json"
    if decision_file.exists():
        decision = json.loads(decision_file.read_text())
        lines += ["", "## Decision", "", decision["summary"], "", decision["next_step"]]
    lines += ["", "## Artifacts and limits", "",
              "- `sweep.json` contains all rate pairs, full configuration/runtime/source fingerprints, selection scores, and summaries.",
              "- `episodes/` contains every feature/action/metric trace, separate final EMA results, and complete error tracebacks.",
              "- `fit.json` and `sgd_policy.json`, when present, record all controller proposals and the frozen policy.",
              "- Tests analytically verify `parameter_next = parameter - LR * gradient`, empty optimizer state, and scalar-only controller effects.",
              "", "This is a limited rate-grid experiment on one architecture and formulation. It does not establish that SGD can never work. "
              "Previously observed ring8 and full-suite results are validation data, not fresh held-out tests. No new distribution or architecture is used to tune this study.", ""]
    (source / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    render(parser.parse_args().input)

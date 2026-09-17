#!/usr/bin/env python
"""Single-seed Hopfield study leaderboard, decision report, and figures.

    python experiments/analyze_hopfield.py --runs-dir results/hopfield/runs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("modes", "hq", "sigma_ratio", "tv", "kl", "steps_to_tv",
           "max_w", "eff_n", "dead_frac", "interp_hq", "log_beta")


def horizon(config):
    return int(config.get("steps", config.get("epochs", 7) * config.get("steps_per_epoch", 1000)))


def read_series(path, summary):
    metrics = path / "metrics.jsonl"
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    else:
        rows = summary.get("history", [])
    return sorted((r for r in rows if "step" in r and r.get("tv") is not None), key=lambda r: r["step"])


def sustained_tv_step(series):
    """Only the final uninterrupted suffix below the threshold counts."""
    step = None
    for point in reversed(series):
        if not float(point["tv"]) < 0.03:
            break
        step = point["step"]
    return step


def load_runs(runs_dir):
    rows = []
    for file in sorted(runs_dir.glob("*/summary.json")):
        summary = json.loads(file.read_text())
        c, f = summary["config"], summary["final"]
        series = read_series(file.parent, summary)
        row = {"run": file.parent.name, "root": str(file.parent),
               "dataset": c["dataset"], "read": c["read"],
               "num_particles": c["num_particles"], "seed": c["seed"],
               "beta": c.get("beta"), "learn_beta": c.get("learn_beta", False),
               "steps": horizon(c), **{key: f.get(key) for key in METRICS}}
        if series:
            row["steps_to_tv"] = sustained_tv_step(series)
        row["quality_pass"] = (f.get("modes", 0) == 100 and (f.get("hq") or 0) >= 0.98
                               and (f.get("sigma_ratio") or 0) >= 0.80)
        row["fidelity_pass"] = f.get("tv", math.inf) < 0.03
        row["_series"], row["_summary"] = series, summary
        rows.append(row)
    return sorted(rows, key=lambda r: (r["dataset"], r["num_particles"], r["tv"]))


def label(row):
    if row["read"] == "uniform":
        return "uniform"
    return "Hopfield β learned" if row["learn_beta"] else f"Hopfield β={row['beta']:g}"


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.4f}"


def kill_decision(rows):
    candidates = [r for r in rows if r["dataset"] == "imbalanced" and r["num_particles"] == 100 and r["steps"] == 7000]
    baseline = next((r for r in candidates if r["read"] == "uniform"), None)
    hopfield = next((r for r in candidates if r["read"] == "hopfield" and not r["learn_beta"] and r["beta"] == 16 and (baseline is None or r["seed"] == baseline["seed"])), None)
    if baseline is None or hopfield is None:
        return {"status": "pending", "reason": "The matched 7,000-step kill-test pair is not complete."}
    improvement = baseline["tv"] - hopfield["tv"]
    return {"status": "pass" if improvement >= 0.20 else "stop", "tv_improvement": improvement,
            "uniform_run": baseline["run"], "hopfield_run": hopfield["run"],
            "uniform_tv": baseline["tv"], "hopfield_tv": hopfield["tv"], "required_improvement": 0.20}


def compare_grid(rows):
    """Apply the proposal's comparisons without claiming multi-seed success."""
    core = [r for r in rows if r["steps"] == 7000]
    identities = {(r["dataset"], r["num_particles"], r["read"],
                   r["beta"] if r["read"] == "hopfield" else None,
                   r["learn_beta"]) for r in core}
    comparisons = []
    for arm in core:
        if arm["dataset"] != "imbalanced" or arm["read"] != "hopfield" or arm["num_particles"] != 1000:
            continue
        matched = [r for r in core if r["seed"] == arm["seed"]]
        baseline = next((r for r in matched if r["dataset"] == "imbalanced" and r["read"] == "uniform" and r["num_particles"] == 20000), None)
        control = next((r for r in matched if r["dataset"] == "uniform" and r["read"] == "hopfield" and r["num_particles"] == 1000 and r["beta"] == arm["beta"] and r["learn_beta"] == arm["learn_beta"]), None)
        uniform_control = next((r for r in matched if r["dataset"] == "uniform" and r["read"] == "uniform" and r["num_particles"] == 1000), None)
        quality_fidelity = arm["quality_pass"] and arm["fidelity_pass"]
        timing = None
        if baseline is not None:
            timing = arm["steps_to_tv"] is not None and (baseline["steps_to_tv"] is None or baseline["steps_to_tv"] > 2 * arm["steps_to_tv"])
        no_regression = None
        if control is not None and uniform_control is not None:
            keys = ("hq", "sigma_ratio")
            if all(r[k] is not None for r in (control, uniform_control) for k in keys):
                no_regression = (control["hq"] >= uniform_control["hq"] - 0.01 and control["sigma_ratio"] >= uniform_control["sigma_ratio"] - 0.05)
        all_comparisons = baseline is not None and no_regression is not None
        screen_pass = quality_fidelity and timing is True and no_regression is True
        efficiency_partial = (all_comparisons and quality_fidelity and timing is False
                              and baseline["fidelity_pass"] and baseline["quality_pass"]
                              and no_regression is True)
        comparisons.append({"run": arm["run"], "quality_and_fidelity": quality_fidelity,
                            "baseline_run": baseline["run"] if baseline else None,
                            "baseline_over_2x_slower_or_censored": timing,
                            "uniform_control_no_regression": no_regression,
                            "single_seed_full_criteria": screen_pass,
                            "particle_efficiency_partial": efficiency_partial,
                            "interpolation_pass": arm["interp_hq"] is not None and arm["interp_hq"] >= 0.90})
    if any(c["single_seed_full_criteria"] for c in comparisons):
        outcome = "single_seed_full_criteria"
    elif any(c["particle_efficiency_partial"] for c in comparisons):
        outcome = "single_seed_particle_efficiency_partial"
    else:
        outcome = "no_qualifying_result" if len(identities) >= 30 else "incomplete"
    return {"core_arms_complete": len(identities), "expected_core_arms": 30,
            "outcome": outcome, "comparisons": comparisons}


def write_tables(rows, out):
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (out / "leaderboard.json").write_text(json.dumps(clean, indent=2) + "\n")
    columns = ["run", "dataset", "read", "num_particles", "seed", "beta", "learn_beta", "steps", *METRICS, "quality_pass", "fidelity_pass"]
    with (out / "leaderboard.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean)
    lines = ["# Hopfield read leaderboard", "", "One seed per arm; no seed averaging or uncertainty estimate. Ranked by final TV within each dataset and particle count.", "",
             "| Dataset | M | Read | Seed | Modes | HQ | σ ratio | TV ↓ | KL ↓ | Steps to TV | max_w | eff_n | dead_frac | interp_hq | log β | Quality floor |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["dataset"], str(row["num_particles"]), label(row), str(row["seed"])]
        for key in METRICS:
            values.append(f">{row['steps']}" if key == "steps_to_tv" and row[key] is None else fmt(row[key]))
        values.append("pass" if row["quality_pass"] else "fail")
        lines.append("| " + " | ".join(values) + " |")
    text = "\n".join(lines) + "\n"
    (out / "TABLE.md").write_text(text)
    (out / "LEADERBOARD.md").write_text(text)


def plot_histories(rows, out, key, filename):
    selected = [r for r in rows if key != "eff_n" or r["read"] == "hopfield"]
    panels = sorted({(r["dataset"], r["num_particles"]) for r in selected})
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.4 * len(panels)), squeeze=False)
    for axis, panel in zip(axes[:, 0], panels):
        for row in selected:
            if (row["dataset"], row["num_particles"]) != panel:
                continue
            points = [p for p in row["_series"] if p.get(key) is not None]
            if points:
                axis.plot([p["step"] for p in points], [p[key] for p in points], label=f"{label(row)} ({row['steps']} steps)")
        if key == "tv":
            axis.axhline(0.03, color="black", linestyle="--", alpha=0.5, label="TV target 0.03")
        elif key == "eff_n":
            axis.axhline(panel[1], color="black", linestyle="--", alpha=0.5, label="M")
            axis.set_yscale("log")
        axis.set(title=f"{panel[0]}, M={panel[1]}", xlabel="Training step", ylabel=key)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)


def target_weights(row):
    root = Path(row["root"])
    if (root / "target_weights.npy").exists():
        return np.load(root / "target_weights.npy")
    if (root / "target_weights.json").exists():
        value = json.loads((root / "target_weights.json").read_text())
        return np.asarray(value.get("weights", value) if isinstance(value, dict) else value)
    value = row["_summary"].get("target_weights")
    return None if value is None else np.asarray(value)


def plot_mode_weights(rows, out):
    hopfields = [r for r in rows if r["read"] == "hopfield" and r["dataset"] == "imbalanced"]
    if not hopfields:
        return None
    best = min(hopfields, key=lambda r: r["tv"])
    baseline = next((r for r in rows if r["read"] == "uniform" and r["dataset"] == best["dataset"] and r["num_particles"] == best["num_particles"] and r["seed"] == best["seed"] and r["steps"] == best["steps"]), None)
    target = target_weights(best)
    if target is None:
        return best
    order = np.argsort(target)
    fig, axis = plt.subplots(figsize=(14, 4.5))
    axis.bar(np.arange(100) - 0.27, target[order], 0.27, label="True weights")
    for offset, row in [(0, best), (0.27, baseline)]:
        if row is None:
            continue
        values = row["_summary"]["final"].get("mode_weights")
        if values is not None:
            axis.bar(np.arange(100) + offset, np.asarray(values)[order], 0.27, label=label(row))
    axis.set(yscale="log", xlabel="Mode (sorted by true weight)", ylabel="HQ-conditional probability",
             title=f"Imbalanced mode weights, M={best['num_particles']}; zero bars are absent on log scale")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "mode_weights.png", dpi=150)
    plt.close(fig)
    return best


def make_gifs(rows, best, decision, out):
    from PIL import Image
    chosen = {decision.get("uniform_run"), decision.get("hopfield_run")}
    if best:
        chosen.add(best["run"])
    written = []
    for row in rows:
        if row["run"] not in chosen:
            continue
        root = Path(row["root"])
        files = sorted(root.glob("samples_step_*.png")) or sorted(root.glob("snapshots/*.png"))
        if not files:
            continue
        # A thumbnail retains the scatter structure and keeps README assets manageable.
        frames = []
        for file in files:
            with Image.open(file) as img:
                img.thumbnail((900, 600))
                frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
        name = f"{row['run']}.gif"
        frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=250, loop=0)
        for frame in frames:
            frame.close()
        written.append(name)
    return written


def write_report(rows, decision, best, gifs, out):
    comparisons = compare_grid(rows)
    (out / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    lines = ["# Hopfield particle-prior study", "", "This is a one-seed-per-arm screening study. The original ≥2-of-3-seeds success criterion cannot be assessed; no seed mean ± sd is reported.", ""]
    if decision["status"] == "pending":
        lines.append(decision["reason"])
    else:
        lines += [f"**Kill test: {'PASS' if decision['status'] == 'pass' else 'STOP'}.** Uniform final TV = {decision['uniform_tv']:.4f}; Hopfield β=16 final TV = {decision['hopfield_tv']:.4f}. Improvement = {decision['tv_improvement']:.4f}, required ≥0.2000.", ""]
        if decision["status"] == "stop":
            lines.append("Recommendation: stop before the full grid. At this recipe, M=100 and this seed, the proposed retrieval fails its predeclared reweighting gate. This does not rule out other temperatures, larger memories, or eventual baseline convergence.")
        elif comparisons["core_arms_complete"] < 30:
            lines.append("Recommendation: proceed through the single-seed grid; assess quality and interpolation as well as TV before interpreting a particle-efficiency benefit.")
        else:
            lines += [f"Completed {comparisons['core_arms_complete']}/30 core arms. Comparative screening outcome: `{comparisons['outcome']}`.", ""]
            if comparisons["outcome"] == "single_seed_full_criteria":
                lines.append("An M=1,000 Hopfield arm satisfies fidelity/quality, baseline timing and matched uniform-control criteria for this seed. This is single-seed evidence, not the proposed multi-seed success claim.")
            elif comparisons["outcome"] == "single_seed_particle_efficiency_partial":
                lines.append("An M=1,000 Hopfield arm matches the qualifying M=20,000 uniform baseline without the required >2× timing advantage. The supported result is particle efficiency; assess interpolation separately. There is no unique weight-fidelity advantage here.")
            else:
                lines.append("No M=1,000 Hopfield arm satisfies the combined fidelity, quality and uniform-control criteria. Retain the existing recipe. A regularization follow-up is warranted only if its TV plateaus while read-health remains healthy.")
    if comparisons["comparisons"]:
        lines += ["", "Matched M=1,000 Hopfield comparisons (baseline M=20,000; uniform-control dataset at M=1,000):", "",
                  "| Run | Quality + TV | >2× baseline timing advantage | Control no regression | interp_hq ≥0.90 |", "|---|---|---|---|---|"]
        for item in comparisons["comparisons"]:
            flags = ["pending" if item[key] is None else ("pass" if item[key] else "fail") for key in ("quality_and_fidelity", "baseline_over_2x_slower_or_censored", "uniform_control_no_regression", "interpolation_pass")]
            lines.append("| " + " | ".join([item["run"], *flags]) + " |")
    for row in rows:
        noise = row["_summary"].get("noise_floor")
        if noise:
            lines += ["", f"Measured true-mixture TV noise floor for `{row['run']}`: {noise['mean']:.5f} ± {noise['sd']:.5f} ({noise['repeats']} draws of {noise['n_eval']:,} samples). This repeated-sampling uncertainty is distinct from training-seed uncertainty."]
        if row["read"] == "uniform" and row["sigma_ratio"] is not None and row["sigma_ratio"] < 0.1:
            lines += ["", f"`{row['run']}` has near-delta within-mode spread (σ ratio {row['sigma_ratio']:.4f}); its better TV does not make it a successful generative model."]
        if row["read"] == "hopfield":
            flags = []
            if row["sigma_ratio"] is not None and row["sigma_ratio"] < 0.8:
                flags.append("within-mode spread misses the 0.80 floor")
            if row["sigma_ratio"] is not None and row["sigma_ratio"] > 2:
                flags.append("the large core σ ratio does not establish healthy Gaussian spread: it averages nearest-center widths over modes, including diffuse low-mass assignments; interpret it with coverage and HQ")
            if row["eff_n"] is not None and row["eff_n"] <= 1.5:
                flags.append("effective retrieval is close to one particle per query")
            if row["dead_frac"] is not None and row["dead_frac"] >= 0.5:
                flags.append(f"{row['dead_frac']:.1%} of particles never win an argmax in evaluation")
            if row["eff_n"] is not None and row["eff_n"] > row["num_particles"] * 0.5:
                flags.append("diffuse retrieval uses more than half the table per query on average")
            if row["max_w"] is not None and row["max_w"] > 0.99:
                flags.append("retrieval is nearly hard")
            if row["learn_beta"] and row["log_beta"] is not None and row["log_beta"] >= math.log(256) - 1e-3:
                flags.append("learned β reaches its upper clamp")
            if row["interp_hq"] is not None and row["interp_hq"] < 0.90:
                flags.append("query interpolation misses the 0.90 HQ target")
            if flags:
                lines += ["", f"`{row['run']}`: " + "; ".join(flags) + "."]
    if best:
        lines += ["", f"Lowest-TV measured imbalanced Hopfield arm: `{best['run']}` (TV {best['tv']:.4f}; quality floor {'passes' if best['quality_pass'] else 'fails'}). Ranking by TV alone does not establish study success."]
    lines += ["", "`steps_to_tv` is the start of the final uninterrupted sequence of evaluation points with TV <0.03. `>7000` means no sustained convergence was observed within that horizon; it does not imply convergence is impossible.", "", "All read-health and quality metrics refer to the EMA read. Uniform read-health and interpolation values are N/A. HQ-conditional TV can conceal dropped low-quality mass, so always read it together with HQ, coverage and σ ratio.", "", "Artifacts: [leaderboard](TABLE.md), [TV curves](tv_vs_step.png), [read entropy diagnostic](eff_n_vs_step.png), [mode weights](mode_weights.png)."]
    if gifs:
        lines += ["", "Scatter animations: " + ", ".join(f"[{name}]({name})" for name in gifs) + "."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", "--runs_dir", type=Path, default=Path("results/hopfield/runs"))
    parser.add_argument("--out-dir", "--out_dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_dir or args.runs_dir.parent
    out.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.runs_dir)
    if not rows:
        parser.error(f"No completed summaries in {args.runs_dir}")
    write_tables(rows, out)
    decision = kill_decision(rows)
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    plot_histories(rows, out, "tv", "tv_vs_step.png")
    plot_histories(rows, out, "eff_n", "eff_n_vs_step.png")
    best = plot_mode_weights(rows, out)
    gifs = make_gifs(rows, best, decision, out)
    write_report(rows, decision, best, gifs, out)
    print(f"Wrote {len(rows)} arms to {out}/TABLE.md; kill gate: {decision['status']}")


if __name__ == "__main__":
    main()

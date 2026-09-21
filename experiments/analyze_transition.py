#!/usr/bin/env python
"""Build a matched-budget transition leaderboard and portable visual report."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def preference_audit(path, run):
    """Observed y-side frequency near the midpoint; not a support-validity test."""
    with np.load(path/"test_samples.npz") as data:
        length = run["config"]["length"]
        tick = min(np.unique(data["tick"]), key=lambda t: abs(t/(length-1)-.5))
        time = tick/(length-1)
        if np.sin(np.pi*time)**2 < .25:
            return None
        geom = data["geom"]
        center = geom[:, 0]*(1-time)+geom[:, 1]*np.sin(np.pi*time)**2
        rows = []
        for c in (0, 1):
            mask = (data["tick"] == tick) & (data["c"] == c)
            rows.append(dict(c=c, count=int(mask.sum()),
                             real_upper=float((data["real"][mask, 1] > center[mask]).mean()),
                             generated_upper=float((data["x"][mask, 1] > center[mask]).mean())))
        return dict(tick=int(tick), time=float(time), classes=rows)


def analyze(paths, out):
    runs = [(Path(path), json.loads((Path(path)/"summary.json").read_text())) for path in paths]
    ignored = {"architecture", "width", "critic_mode", "out_dir", "live_log"}
    def label(run):
        return run["config"]["architecture"]+"_"+run["config"].get("critic_mode", "joint")
    def contract(path, run):
        return dict(config={k: v for k, v in run["config"].items() if k not in ignored},
                    recipe=run["recipe"], sources=run["provenance"]["sources"],
                    normalization=json.loads((path/"normalization.json").read_text()),
                    prior_config=json.loads((path/"prior.json").read_text()) if (path/"prior.json").exists() else None,
                    real_draws=run["real_draws"], D_joint=run.get("critic_parameters", {}).get("joint", run["parameters"]["D"]),
                    prior=run["parameters"]["prior"])
    reference = contract(*runs[0])
    if any(contract(path, run) != reference for path, run in runs[1:]):
        raise ValueError("Runs differ beyond generator architecture/width or added critics; use separate leaderboards")
    sizes = [run["parameters"]["G"] for _, run in runs]
    if max(sizes)/min(sizes) > 1.05:
        raise ValueError("Generator budgets differ by more than 5%")
    if len({label(run) for _, run in runs}) != len(runs):
        raise ValueError("Expected distinct generator/critic configurations, not seed repeats")
    runs.sort(key=lambda item: item[1]["final"]["test"]["joint_sw1"])
    preferences = {label(run): preference_audit(path, run) for path, run in runs}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Single-transition GAN results", "",
             "G1 -> st; G2 -> at; G3 -> st+1. All three receive the same z and context. "
             "Joint D(st, at, st+1) supplies feedback. The marginal arm adds D(st), D(at), D(st+1), "
             "each conditioned on the observed context. Action is displacement.", "",
             f"Matched {reference['config']['steps']:,} updates, {reference['real_draws']:,} real training draws, "
             f"seed {reference['config']['seed']}. Public {reference['recipe']['name']} recipe with "
             f"{reference['recipe']['num_particles']:,} particles, UCD and z_dim={reference['recipe']['z_dim']}. "
             "Training draws, initial joint D/prior, normalization, source hashes and evaluation settings are matched. "
             "Generator initializations differ with architecture. No seed-only repeats.", "",
             "Marginal critics add capacity and compute; only generator updates and real-data budgets are matched. "
             "Each critic has its own full-strength Rp/UCD/bcap objective. Generator feedback is "
             "L_joint + marginal_weight * mean(L_state, L_action, L_next), plus one prior regularizer. "
             "The joint critic's architecture is unchanged.", "",
             "Lower SW1 and residual are better. Distances use frozen training normalization; "
             "residuals are in physical coordinates. Ranking is by aggregate held-out joint SW1.", "",
             "| Model | G parameters | D parameters | Test joint SW1 | Interp. | Extrap. | State SW1 | Action SW1 | Next SW1 | Residual mean / p95 | Train seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path, run in runs:
        m = run["final"]["test"]
        name = label(run)
        lines.append(f"| {name} | {run['parameters']['G']:,} | {run['parameters']['D']:,} | {m['joint_sw1']:.4f} | "
                     f"{run['final']['interpolation']['joint_sw1']:.4f} | {run['final']['extrapolation']['joint_sw1']:.4f} | "
                     f"{m['state_sw1']:.4f} | {m['action_sw1']:.4f} | {m['next_state_sw1']:.4f} | "
                     f"{m['consistency_mean']:.5f} / {m['consistency_p95']:.5f} | {run['train_seconds']:.1f} |")
    floor = runs[0][1]["reference_floor"]["test"]
    control = runs[0][1]["controls"]["test"]["shuffled_real"]
    with np.load(runs[0][0]/"test_samples.npz") as samples:
        mean_step = float(np.linalg.norm(samples["real"][:, 2:4], axis=1).mean())
    lines += ["", f"Real-vs-real floor: joint SW1 **{floor['joint_sw1']:.4f}**, "
              f"residual **{floor['consistency_mean']:.2g}**. Shuffling reference branches within context "
              f"preserves all three empirical marginals but raises joint SW1 to **{control['joint_sw1']:.4f}** "
              f"and residual to **{control['consistency_mean']:.5f}**. "
              f"The mean reference displacement is **{mean_step:.5f}**, giving a physical scale for the residual.", "",
              "| Model | Coverage | Precision | Spread ratio | Shuffled generated joint SW1 | Shuffled generated residual |",
              "|---|---:|---:|---:|---:|---:|"]
    for path, run in runs:
        name = label(run)
        m, s = run["final"]["test"], run["controls"]["test"]["shuffled_generated"]
        lines.append(f"| {name} | {m['coverage']:.3f} | {m['precision']:.3f} | {m['spread_ratio']:.3f} | "
                     f"{s['joint_sw1']:.4f} | {s['consistency_mean']:.5f} |")
        for filename in ("viewer.html", "transitions.png"):
            shutil.copy2(path/filename, out/f"{name}_{filename}")
        if (path/"render_provenance.json").exists():
            shutil.copy2(path/"render_provenance.json", out/f"{name}_render_provenance.json")
    lines += ["", f"Reference coverage/precision/spread: {floor['coverage']:.3f} / "
              f"{floor['precision']:.3f} / {floor['spread_ratio']:.3f}. Coverage is the fraction of reference "
              "points with a generated neighbor inside the reference's 95th-percentile nearest-neighbor radius; "
              "precision reverses the direction. This is a strict six-dimensional support check. "
              "Spread is total conditional normalized variance divided by reference variance, target 1.", "",
              "## Interpretation", "",
              f"**{label(runs[0][1])}** has the lowest held-out joint SW1 in this comparison. "
              "Use residual and support coverage alongside that rank: good marginal distances or "
              "total variance do not establish a physically coherent transition.", ""]
    for _, run in runs:
        name = label(run)
        m = run["final"]["test"]
        shuffled = run["controls"]["test"]["shuffled_generated"]
        lines += [f"For **{name}**, shuffling raises joint SW1 from {m['joint_sw1']:.4f} "
                  f"to {shuffled['joint_sw1']:.4f} and residual from {m['consistency_mean']:.5f} "
                  f"to {shuffled['consistency_mean']:.5f}. The unshuffled residual is "
                  f"{100*m['consistency_mean']/mean_step:.1f}% of the mean reference step length. "
                  "This ratio compares aggregate means, rather than averaging per-sample ratios.", ""]
    by_arch = {run["config"]["architecture"]: run for _, run in runs if run["config"].get("critic_mode", "joint") == "joint"}
    if set(by_arch) == {"branches", "monolithic"}:
        branch, mono = (by_arch[k]["final"]["test"] for k in ("branches", "monolithic"))
        if mono["joint_sw1"] < branch["joint_sw1"] and mono["consistency_mean"] < branch["consistency_mean"]:
            lines += ["The monolithic generator wins on both joint distance and consistency. "
                      "The three branches coordinate, but there is no evidence of an advantage from "
                      "separating their parameters in this run. Shared intermediate features could "
                      "make the relation easier to represent; that explanation is a hypothesis, "
                      "not something this two-arm comparison isolates.", ""]
    variants = {label(run): run for _, run in runs}
    if {"branches_joint", "branches_joint_marginals"} <= set(variants):
        a, b = (variants[key]["final"]["test"] for key in ("branches_joint", "branches_joint_marginals"))
        lines += ["Adding marginal critics to the same three generators changes held-out distances as follows "
                  "(negative is better): " + ", ".join(f"{key}: {b[key]-a[key]:+.4f}" for key in
                  ("joint_sw1", "state_sw1", "action_sw1", "next_state_sw1", "consistency_mean")) + ". "
                  "This measures added marginal supervision and added critic compute together.", ""]
    if all(preferences.values()):
        lines += ["## Preference-class check", "",
                  "At the sampled time nearest the route midpoint, count states above the analytic "
                  "centerline, averaged over held-out geometries. Class 0 should favor the upper side "
                  "(target 0.8); class 1 should favor the lower side (upper target 0.3). This checks "
                  "class-dependent mixture weights, not whether a generated point is on valid support.", "",
                  "| Model | Class 0 upper fraction | Class 1 upper fraction |",
                  "|---|---:|---:|"]
        for name, audit in preferences.items():
            rows = audit["classes"]
            lines.append(f"| {name} | {rows[0]['generated_upper']:.3f} | {rows[1]['generated_upper']:.3f} |")
        rows = next(iter(preferences.values()))["classes"]
        lines += [f"| Reference samples | {rows[0]['real_upper']:.3f} | {rows[1]['real_upper']:.3f} |", "",
                  "Similar generated frequencies across the two classes indicate weak use of the preference label, "
                  "even if transition consistency is good.", ""]
        weak = [name for name, audit in preferences.items()
                if abs(audit["classes"][0]["generated_upper"]-audit["classes"][1]["generated_upper"]) < .15]
        if weak:
            lines += ["**Weak class separation:** " + ", ".join(weak) + ". Their upper-side frequencies "
                      "differ by less than 15 percentage points, versus the target's 50-point difference. "
                      "A class-agnostic sampler can fit the pooled upper probability (0.55) while missing both "
                      "conditional distributions.", ""]
    lines += [
              "This tests learning the joint transition distribution from complete records. "
              "It does not yet test missing-data recovery, benefit over marginal-only training, "
              "or arbitrary state/action queries and rollouts.", "",
              "## Recommendation", "",
              "Where the preference check fails, first compare explicitly feeding the class label "
              "into the critic against the current UCD class-head selection, holding the MoG, "
              "generator architecture and update budget fixed. This would test whether critic "
              "conditioning is responsible; the current results do not establish the cause.", "",
              "Keep the better joint sampler as the reference. Next, add a marginal-only baseline "
              "to test the motivating claim that learning the whole helps a part; compare its "
              "per-block distances and shuffled controls under a matched total budget. "
              "Resolve the remaining marginal/support errors before interpreting added marginal "
              "critics or missing observations. A better total variance ratio alone is insufficient.", "",
              "## Visuals", ""]
    lines += ["Plots can be regenerated from saved samples without retraining. The separate "
              "render provenance records display-code revisions; training source archives stay unchanged.", ""]
    for _, run in runs:
        name = label(run)
        lines += [f"[{name} interactive viewer]({name}_viewer.html)", "",
                  f"![{name} transitions]({name}_transitions.png)", ""]
    (out/"README.md").write_text("\n".join(lines))
    (out/"leaderboard.json").write_text(json.dumps([
        dict(run=str(path), architecture=run["config"]["architecture"], critic_mode=run["config"].get("critic_mode", "joint"), parameters=run["parameters"],
             final=run["final"], reference_floor=run["reference_floor"], controls=run["controls"],
             preference_audit=preferences[label(run)], train_seconds=run["train_seconds"]) for path, run in runs], indent=2, allow_nan=False)+"\n")
    return out/"README.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out-dir", default="reports/transition/mog_1024")
    args = parser.parse_args()
    print(analyze(args.runs, args.out_dir))

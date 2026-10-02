"""Plot every qualified guided-toy test checkpoint under all four fixed critics.

This reads scalar observations only: no model loading, training, smoothing,
output metric, head selection or best-checkpoint selection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/routed-convergence-guided-pair-v1"
ARMS = ("ordinary_native_game", "particle_native_game", "neutral_particle_native_game")
JUDGES = tuple(f"{arm}@{step}" for arm in ARMS[:2] for step in (800, 6400))
STEPS = tuple(sorted({0, 5120, *range(200, 6401, 200)}))
LABELS = ("Ordinary LoRA (native game)", "Original particles (sampled H/b)",
          "Neutral particles (H/b = 0 at init)")
COLORS = ("#2b5d9f", "#d17a17", "#218850")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def qualified_curves(run):
    paths = {
        "curves": run / "common-judge-curves.jsonl", "execution": run / "receipt.json",
        "execution_completion": run / "execution-completion.json",
        "review": run / "independent-review.json",
        "review_completion": run / "independent-review.json.completion.json",
    }
    hashes = {name: sha(path) for name, path in paths.items()}
    execution, completion, review, review_completion = (
        read(paths[name]) for name in ("execution", "execution_completion", "review", "review_completion"))
    if (execution["schema"] != "routed_convergence_guided_pair_execution_v1"
            or execution["status"] != "complete" or completion.get("complete") is not True
            or completion["receipt_sha256"] != hashes["execution"]
            or review["schema"] != "routed_convergence_guided_pair_independent_review_v1"
            or review.get("qualified") is not True or review["status"] != "qualified"
            or review["execution_receipt_sha256"] != hashes["execution"]
            or review_completion.get("complete") is not True
            or review_completion["review_sha256"] != hashes["review"]
            or set(review["judges"]) != set(JUDGES) or review["judges"] != execution["judges"]):
        raise ValueError("exact completed, independently qualified guided cohort required")
    rows = [json.loads(line) for line in paths["curves"].read_text().splitlines()]
    identities = {(row["arm"], row["step"]) for row in rows}
    expected = {(arm, step) for arm in ARMS for step in STEPS}
    if len(rows) != 102 or len(identities) != len(rows) or identities != expected:
        raise ValueError("all34 checkpoints for all3 arms, without duplicates, are required")
    values = {}
    for row in rows:
        if set(row["scores"]) != set(JUDGES):
            raise ValueError("all four fixed critics required in every checkpoint")
        for judge in JUDGES:
            score = row["scores"][judge]["test"]["paired_game"]
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("finite native test game required")
            values[row["arm"], row["step"], judge] = score
    return values, paths, hashes, review, execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=RUN / "figures")
    parser.add_argument("--receipt", type=Path, default=ROOT / "docs/e22_routed_convergence_guided_pair_plot.json")
    args = parser.parse_args()
    started = time.monotonic()
    values, paths, hashes, review, execution = qualified_curves(args.run)
    accepted_moves = {arm: execution["arms"][arm]["coverage"]["moves"] for arm in ARMS}
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/e22-guided-toy-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
        "svg.hashsalt": "e22-guided-toy-qualified-v1", "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 11.5, "axes.labelsize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.975, bottom=.17, top=.775, hspace=.27, wspace=.15)
    minimum, maximum = min(values.values()), max(values.values())
    margin = .055 * (maximum - minimum)
    for index, (axis, judge) in enumerate(zip(axes.flat, JUDGES)):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            axis.plot(STEPS, [values[arm, step, judge] for step in STEPS],
                color=color, linewidth=1.9, marker="o", markersize=2.8,
                markeredgewidth=0, label=label)
        for endpoint, color in ((5120, "#89919c"), (6400, "#525b68")):
            axis.axvline(endpoint, color=color, linestyle=(0, (4, 3)), linewidth=1.05, zorder=0)
            axis.text(endpoint, .985, str(endpoint), transform=axis.get_xaxis_transform(),
                      ha="right" if endpoint == 6400 else "center", va="top", fontsize=8.5,
                      color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.5))
        family, step = judge.split("@")
        family_label = "Ordinary LoRA" if family == ARMS[0] else "Original particle"
        axis.set_title(f"{chr(65+index)}   {family_label} critic, fixed at step {step}", loc="left", pad=11)
        axis.set_xlim(-100, 6550)
        axis.set_ylim(minimum-margin, maximum+margin)
        axis.set_xticks((0, 1600, 3200, 4800, 6400))
        axis.grid(axis="y", color="#dce1e6", linewidth=.7)
        axis.tick_params(length=3.5, color="#89919c")
        axis.set_axisbelow(True)
    axes[0,0].set_ylabel("Held-out paired RpGAN game ↓")
    axes[1,0].set_ylabel("Held-out paired RpGAN game ↓")
    axes[1,0].set_xlabel("Native training updates")
    axes[1,1].set_xlabel("Native training updates")
    fig.suptitle("Guided two-site toy: convergence under all four fixed critics", fontsize=17,
                 x=.085, y=.965, ha="left", fontweight="bold")
    fig.text(.085,.916,"CFG = 3  ·  clean FAST  ·  all 34 saved checkpoints  ·  lower game is better within each critic",
             fontsize=11, color="#4d5865")
    handles = [Line2D([0],[0],color=color,lw=2.4,marker="o",markersize=4,label=label)
               for color,label in zip(COLORS,LABELS)]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.078,.888),ncol=3,
               frameon=False,fontsize=10.5,handlelength=2.2,columnspacing=1.8)
    fig.text(.085,.095,"Dashed lines mark the fixed 5120 and 6400 endpoints. Every saved test point is plotted; no smoothing or state selection.",
             fontsize=9.2,color="#4d5865")
    moves_note = ("No structural moves were accepted; this does not establish a birth/death gain."
                  if sum(accepted_moves.values()) == 0 else
                  "Accepted structural moves: " + ", ".join(str(accepted_moves[arm]) for arm in ARMS) +
                  " (ordinary / original / neutral).")
    fig.text(.085,.064,"Qualified CPU toy cohort only. Critics have different learned score scales. " + moves_note,
             fontsize=9.2,color="#4d5865")
    args.output.mkdir(parents=True,exist_ok=True)
    outputs = {extension: args.output / f"guided-toy-convergence.{extension}" for extension in ("svg","png")}
    fig.savefig(outputs["svg"],metadata={"Date": None})
    fig.savefig(outputs["png"],dpi=180)
    plt.close(fig)
    if {name:sha(path) for name,path in paths.items()} != hashes:
        raise AssertionError("qualified input files changed while plotting")
    endpoints = {str(step): {judge: {arm: values[arm,step,judge] for arm in ARMS}
                             for judge in JUDGES} for step in (5120,6400)}
    receipt = dict(schema="guided_toy_convergence_figure_v1",input_paths={n:str(p) for n,p in paths.items()},
        input_sha256=hashes,plot_source_sha256=sha(__file__),review_checks=review["checks"],
        data_digest=review["data_digest"],critics=review["judges"],arms=list(ARMS),labels=list(LABELS),
        checkpoint_steps=list(STEPS),checkpoint_rows=102,plotted_values=408,panels=4,pool="test",
        metric="paired_game",head="clean FAST",smoothing=False,state_selection=False,output_metrics_used=False,
        fixed_endpoints=[5120,6400],fixed_endpoint_values=endpoints,
        accepted_structural_moves=accepted_moves,
        files={extension: {"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}
               for extension,path in outputs.items()},matplotlib_version=matplotlib.__version__,
        seconds=time.monotonic()-started,model_or_optimizer_updates=0,
        scope="Qualified guided two-site CPU cohort; within-critic comparisons only, no full-Supra or structural-move benefit claim")
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(receipt=str(args.receipt),files=receipt["files"],values=408,model_updates=0)),flush=True)


if __name__ == "__main__":
    main()

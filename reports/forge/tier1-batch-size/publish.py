"""Verify saved GPU evidence and render actual-training media; no model draws."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.toy_audit.api_run import render_gif
from benchmarks.toy_audit.tier1_batch_size import declaration, scorer, summarize
from experiments.forge.contracts import atomic_json, file_hash

DEST = Path(__file__).resolve().parent


def scalars(metrics):
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    protocol = declaration()
    run_results = json.loads((args.raw / "results.json").read_text())
    summaries, curves, verification, reference_states = [], {}, [], {}
    for receipt in run_results["results"]:
        task_id, batch = receipt["task"], receipt["batch"]
        name = f"b{batch}-{task_id}"
        raw = args.raw / name
        for artifact, expected in receipt["artifacts"].items():
            if file_hash(raw / artifact) != expected:
                raise ValueError("artifact mismatch: " + str(raw / artifact))
        task = json.loads((ROOT / protocol["tasks"][task_id]["path"]).read_text())
        spec = task["execution"]["host_definition"]
        target, score = scorer(task_id)
        observations = torch.load(raw / "observations.pt", weights_only=True, map_location="cpu")
        curve = json.loads((raw / "curve.json").read_text())
        for obs in observations:
            if score(obs["samples"], spec, obs["step"]) != obs["metrics"]:
                raise ValueError("saved samples no longer reproduce metrics")
        observed = {row["step"]: row["metrics"] for row in observations}
        for row in curve:
            if row["metrics"] != observed[row["step"]]:
                raise ValueError("curve/sample disagreement")
        grade = summarize(curve, task_id, protocol)
        for key, value in grade.items():
            if value != receipt[key]:
                raise ValueError("grade mismatch: " + key)
        curves[name] = curve
        reference_rng = torch.Generator(device="cpu").manual_seed(78013)
        reference = target(spec, 4096, reference_rng, 0)
        reference_states[name] = reference_rng.get_state()
        frames = []
        for index in np.linspace(0, len(observations) - 1, 9).round().astype(int):
            row = observations[index]
            samples = row["samples"]
            if samples.shape[1] == 1:
                edges = np.linspace(-2., 5., 57)
                density = lambda values: np.histogram(values[:, 0].numpy(), edges)[0] / len(values) / np.diff(edges)
                view = dict(kind="bar", title=f"Gaussian target and batch-{batch} output", target=density(reference),
                            samples=density(samples), bin_centers=(edges[:-1] + edges[1:]) / 2,
                            bin_width=float(edges[1] - edges[0]), xlim=[-2., 5.], ylim=[0., 8.5], xlabel="x", ylabel="density")
            else:
                view = dict(kind="scatter", title=f"Ring target and batch-{batch} output", target=reference,
                            samples=samples, xlim=[-4., 4.], ylim=[-4., 4.], xlabel="x", ylabel="y")
            view["caption"] = "Actual saved clean live GPU draws; acquisition and every hold check are graded separately."
            frames.append(dict(step=row["step"], metrics=scalars(row["metrics"]),
                               passed=next((r["full_pass"] for r in curve if r["step"] == row["step"]), False), views=[view]))
        media = DEST / (name + ".gif")
        render_gif(dict(id=name, goal="Acquire the law and retain quality under constant-rate continued training",
                        default_steps=4000), frames, media, full_budget=True, requested_steps=4000,
                   final_verdict=receipt["combined_verdict"])
        compact = {k: receipt[k] for k in ("task", "batch", "mode", "completed_updates", "additional_updates",
                    "continuation_loop_seconds", "total_real_training_examples", "new_real_training_examples",
                    "initial_proof", "prefix_receipt_sha256", "acquisition_verdict", "acquisition_suffix",
                    "hold_verdict", "hold_pass_checks", "hold_total_checks", "combined_verdict", "final_terminal_suffix",
                    "first_five_pass_window", "longest_pass_streak", "total_pass_checks", "total_checks", "artifacts")}
        compact.update(acquisition_metrics=scalars(receipt["acquisition_metrics"]),
                       final_metrics=scalars(receipt["final_metrics"]),
                       cuts=[{**row, "metrics": scalars(row["metrics"])} for row in receipt["cuts"]],
                       raw_receipt_sha256=file_hash(raw / "receipt.json"), source=json.loads((raw / "source.json").read_text()),
                       gif=media.name, gif_sha256=file_hash(media))
        # Full source maps are raw provenance, never bulk compact publication.
        compact["source"] = {k: v for k, v in compact["source"].items() if k != "files"}
        summaries.append(compact)
        verification.append(dict(task=task_id, batch=batch, hash_verified=True,
                                 saved_sample_recomputations=len(observations), grade_reproduced=True))
    plt.rcParams.update({"svg.hashsalt": "tier1-batch-size-v1", "font.size": 9})
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for col, task_id in enumerate(protocol["tasks"]):
        metric_names = ["mean_error_sigma", "std_ratio", "cdf_ks"] if task_id.startswith("gaussian") else ["component_covariance_error", "component_min_eigen_ratio", "hq"]
        bounds = [(.0, .2), (.8, 1.2), (.0, .05)] if col == 0 else [(.0, .85), (.15, 1.), (.85, 1.)]
        for row, (metric, bound) in enumerate(zip(metric_names, bounds)):
            ax = axes[row, col]
            ax.axhspan(*bound, color="#dff0de", label="Passing metric range")
            for batch, color in [(128, "#295888"), (512, "#aa5633")]:
                curve = curves[f"b{batch}-{task_id}"]
                ax.plot([r["step"] for r in curve], [r["metrics"][metric] for r in curve],
                        color=color, label=f"Batch {batch}", lw=1.1)
            ax.axvline(protocol["tasks"][task_id]["acquisition_steps"], color="#555555", linestyle="--", lw=.8)
            ax.set_ylabel(metric.replace("_", " "))
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(task_id)
            if row == 2:
                ax.set_xlabel("Completed updates; dashed line ends acquisition")
            if metric == "component_covariance_error":
                ax.set_yscale("log")
                ax.set_ylim(.1, 20.)
    axes[0, 0].legend(fontsize=8)
    svg = DEST / "comparison.svg"
    fig.savefig(svg, metadata={"Date": None})
    plt.close(fig)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    atomic_json(DEST / "results.json", dict(schema_version=1, id=protocol["id"], scope=protocol["scope"],
               qualification_input=False, protocol_sha256=file_hash(DEST / "protocol.json"),
               new_training_updates=run_results["new_training_updates"],
               new_training_loop_seconds=run_results["new_training_loop_seconds"], results=summaries))
    atomic_json(DEST / "verification.json", dict(training_updates=0, model_sampling_draws=0,
               inputs=verification, total_sample_recomputations=sum(r["saved_sample_recomputations"] for r in verification),
               new_batch_real_streams_match_original_prefix=json.loads((args.raw / "data-audit.json").read_text()),
               publication_source_sha256=file_hash(Path(__file__))))
    torch.save(dict(reference_seed=78013, states=reference_states), args.raw / "publication-rng.pt")
    print(json.dumps(dict(published=4, exact_sample_recomputations=sum(r["saved_sample_recomputations"] for r in verification))))


if __name__ == "__main__":
    main()

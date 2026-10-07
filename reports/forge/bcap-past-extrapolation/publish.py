"""Recompute saved metrics and publish actual-training media; no training."""
import argparse
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
from benchmarks.toy_audit.bcap_past_extrapolation import PROTOCOL, declaration, scorer, summarize
from experiments.forge.contracts import atomic_json, file_hash

DEST = Path(__file__).resolve().parent
COLORS = {"alternating": "#666666", "simultaneous": "#ca8032", "extrapolation_from_past": "#246ca0"}


def scalars(metrics):
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    protocol = declaration()
    results = json.loads((args.raw / "results.json").read_text())
    compact, verification, curves, reference_states = [], [], {}, {}
    for receipt in results["results"]:
        arm, task_id, phase = receipt["arm"], receipt["task"], receipt["phase"]
        name = f"{arm}-{task_id}-{phase}"
        raw = args.raw / name
        for artifact, digest in receipt["artifacts"].items():
            if file_hash(raw / artifact) != digest:
                raise ValueError("saved artifact changed: " + str(raw / artifact))
        task = json.loads((ROOT / protocol["tasks"][task_id]["path"]).read_text())
        spec = task["execution"]["host_definition"]
        if phase == "shift":
            spec["means"] = [[protocol["adaptation"]["mean_after"]]]
        target, score = scorer(task_id)
        snapshots = torch.load(raw / "observations.pt", weights_only=True, map_location="cpu")
        curve = json.loads((raw / "curve.json").read_text())
        recomputed = 0
        for row in snapshots:
            if score(row["samples"], spec, row["step"]) != row["metrics"]:
                raise ValueError("sample metrics do not reproduce")
            recomputed += 1
            if "frozen_samples" in row:
                if score(row["frozen_samples"], spec, row["step"]) != row["frozen_metrics"]:
                    raise ValueError("frozen metrics do not reproduce")
                recomputed += 1
        observations = {row["step"]: row for row in snapshots}
        for row in curve:
            if row["metrics"] != observations[row["step"]]["metrics"]:
                raise ValueError("curve and samples differ")
        grade = summarize(curve, task_id, phase, protocol)
        for key, value in grade.items():
            if receipt[key] != value:
                raise ValueError("saved verdict differs: " + key)
        if phase == "shift":
            frozen_curve = json.loads((raw / "frozen-curve.json").read_text())
            for row in frozen_curve:
                if row["metrics"] != observations[row["step"]]["frozen_metrics"]:
                    raise ValueError("frozen curve and samples differ")
            if summarize(frozen_curve, task_id, phase, protocol) != receipt["frozen_summary"]:
                raise ValueError("frozen grade differs")
        curves[name] = curve
        reference_rng = torch.Generator(device="cpu").manual_seed(78013)
        reference = target(spec, 4096, reference_rng, 0)
        reference_states[name] = reference_rng.get_state()
        frames = []
        for index in np.linspace(0, len(snapshots) - 1, 9).round().astype(int):
            row = snapshots[index]
            samples = row["samples"]
            if samples.shape[1] == 1:
                edges = np.linspace(-2., 6., 65)
                density = lambda values: np.histogram(values[:, 0].numpy(), edges)[0] / len(values) / np.diff(edges)
                view = dict(kind="bar", title=f"Gaussian: {arm}, {phase}", target=density(reference),
                            samples=density(samples), bin_centers=(edges[:-1] + edges[1:]) / 2,
                            bin_width=float(edges[1] - edges[0]), xlim=[-2., 6.], ylim=[0., 8.5], xlabel="x", ylabel="density")
            else:
                view = dict(kind="scatter", title=f"Ring: {arm}", target=reference, samples=samples,
                            xlim=[-4., 4.], ylim=[-4., 4.], xlabel="x", ylabel="y")
            view["caption"] = "Actual saved clean live GPU output; all declared scheduled checks determine the grade."
            frames.append(dict(step=row["step"], metrics=scalars(row["metrics"]),
                               passed=next((r["full_pass"] for r in curve if r["step"] == row["step"]), False), views=[view]))
        gif = DEST / (name + ".gif")
        stop = 4000 if phase == "stationary" else 6000
        render_gif(dict(id=name, goal="Acquire, retain and adapt with constant-rate live updates", default_steps=stop),
                   frames, gif, full_budget=True, requested_steps=stop, final_verdict=receipt["combined_verdict"])
        row = {key: receipt[key] for key in ("arm", "task", "phase", "mode", "completed_updates", "additional_updates",
                                            "loop_seconds", "recipe", "data_sha256", "artifacts")}
        row.update({key: scalars(value) if key.endswith("metrics") else value for key, value in grade.items()})
        row.update(gif=gif.name, gif_sha256=file_hash(gif), raw_receipt_sha256=file_hash(raw / "receipt.json"))
        for key in ("initial_proof", "baseline_receipt_sha256", "original_loop_seconds"):
            if key in receipt:
                row[key] = receipt[key]
        source = json.loads((raw / "source.json").read_text())
        row["source"] = {k: v for k, v in source.items() if k != "files"}
        if phase == "shift":
            row["frozen_summary"] = {k: scalars(v) if k.endswith("metrics") else v for k, v in receipt["frozen_summary"].items()}
        cuts = [1000, 2000, 3000, 4000] if phase == "stationary" else [5000, 6000]
        row["endpoint_cuts"] = [dict(step=r["step"], full_pass=r["full_pass"], metrics=scalars(r["metrics"]))
                                for r in curve if r["step"] in cuts]
        compact.append(row)
        verification.append(dict(arm=arm, task=task_id, phase=phase, artifact_hashes_verified=True,
                                 sample_recomputations=recomputed, live_grade_reproduced=True,
                                 frozen_grade_reproduced=phase == "shift"))
    plt.rcParams.update({"svg.hashsalt": protocol["id"], "font.size": 9})
    for phase, tasks in [("stationary", list(protocol["tasks"])), ("shift", [protocol["adaptation"]["task"]])]:
        fig, axes = plt.subplots(3, len(tasks), figsize=(6 * len(tasks), 8), sharex=True, squeeze=False, constrained_layout=True)
        for col, task_id in enumerate(tasks):
            metrics = ["mean_error_sigma", "std_ratio", "cdf_ks"] if task_id.startswith("gaussian") else ["component_covariance_error", "component_min_eigen_ratio", "hq"]
            bounds = [(0., .2), (.8, 1.2), (0., .05)] if task_id.startswith("gaussian") else [(0., .85), (.15, 1.), (.85, 1.)]
            for i, (metric, bound) in enumerate(zip(metrics, bounds)):
                ax = axes[i, col]
                ax.axhspan(*bound, color="#dff0de")
                for arm, color in COLORS.items():
                    curve = curves[f"{arm}-{task_id}-{phase}"]
                    ax.plot([r["step"] for r in curve], [r["metrics"][metric] for r in curve],
                            label=arm.replace("_", " "), color=color, linewidth=1.)
                cutoff = protocol["tasks"][task_id]["acquisition_steps"] if phase == "stationary" else 5000
                ax.axvline(cutoff, color="#555555", linestyle="--", linewidth=.8)
                ax.set_ylabel(metric.replace("_", " "))
                ax.grid(alpha=.2)
                if i == 0:
                    ax.set_title(f"{task_id}, {phase}")
                if i == 2:
                    ax.set_xlabel("Completed updates; dashed line ends acquisition")
                if metric == "component_covariance_error":
                    ax.set_yscale("log")
        axes[0, 0].legend(fontsize=8)
        svg = DEST / (phase + ".svg")
        fig.savefig(svg, metadata={"Date": None})
        plt.close(fig)
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    atomic_json(DEST / "results.json", dict(schema_version=1, id=protocol["id"], scope=protocol["scope"],
        qualification_input=False, protocol_sha256=file_hash(PROTOCOL), new_training_updates=results["new_training_updates"],
        new_training_loop_seconds=results["new_training_loop_seconds"], results=compact))
    atomic_json(DEST / "verification.json", dict(training_updates=0, model_sampling_draws=0, cells=verification,
        total_sample_recomputations=sum(r["sample_recomputations"] for r in verification), publication_source_sha256=file_hash(Path(__file__))))
    torch.save(dict(reference_seed=78013, states=reference_states), args.raw / "publication-rng.pt")
    print(json.dumps(dict(published=len(compact), sample_recomputations=sum(r["sample_recomputations"] for r in verification))))


if __name__ == "__main__":
    main()

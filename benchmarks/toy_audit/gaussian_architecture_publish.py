"""Verify and publish saved Gaussian architecture outputs without training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.gaussian_tasks import bounds, build, grade
from experiments.forge.state import state_digest
from .gaussian1d_quality import score_samples
from .reproducibility import reproducible_execution

ROOT = Path(__file__).resolve().parents[2]


def load(raw):
    result = json.loads((raw / "results.json").read_text())
    request = json.loads((raw / "request.json").read_text())
    tasks = request["tasks"]
    receipts = {phase: result[phase] for phase in ("smoke", "stability") if result.get(phase)}
    for phase, receipt in receipts.items():
        verify_artifacts(raw / phase, receipt["evidence"]["artifact_manifest"])
    return result, request, tasks, receipts


def recompute(raw, tasks, receipts):
    counts = dict(primary=0, confirmation=0, frozen=0)
    frames_by_phase = {}
    for phase, receipt in receipts.items():
        task = next(task for task in tasks.values() if task["evaluation"]["kind"] == "gaussian_" + phase)
        spec = task["execution"]["host_definition"]
        observed = torch.load(raw / phase / "observed-samples.pt", weights_only=True, map_location="cpu")
        frames_by_phase[phase] = observed
        expected = {row["step"]: row for row in receipt["evidence"]["observations"]}
        confirmed = {row["step"]: row for row in receipt["evidence"]["confirmations"]}
        frozen = {row["step"]: row for row in receipt["evidence"]["frozen_observations"]}
        for snapshot in observed:
            step = snapshot["step"]
            current = deepcopy(spec)
            if step > 4000:
                current["means"] = [[3.]]
            metrics = score_samples(snapshot["samples"], current, step)
            assert metrics == snapshot["metrics"], (phase, step)
            counts["primary"] += 1
            if step in expected:
                assert {"step": step, **metrics} == expected[step], (phase, step)
            if "confirmation_samples" in snapshot:
                confirm = snapshot["confirmation"]
                assert snapshot["training_state_sha256"] == confirm["primary_state_sha256"] == confirm["confirmed_state_sha256"]
                assert confirm["training_state_unchanged"] is True
                assert confirm["independent_stream"] == "eval/live/smoke_confirmation"
                actual = score_samples(snapshot["confirmation_samples"], current, step)
                assert actual == snapshot["confirmation"]["metrics"], (phase, step, "confirmation")
                if step in confirmed:
                    assert snapshot["confirmation"] == confirmed[step]
                counts["confirmation"] += 1
            if "frozen_samples" in snapshot:
                actual = score_samples(snapshot["frozen_samples"], current, step)
                assert actual == snapshot["frozen_metrics"]
                assert {"step": step, **actual} == frozen[step]
                counts["frozen"] += 1
        assert grade(task, receipt["evidence"]) == receipt["gaussian_grade"]
    return counts, frames_by_phase


@reproducible_execution
def restore(raw, *, device):
    assert torch.device(device).type == "cuda" and torch.cuda.is_available()
    _, request, tasks, receipts = load(raw)
    restored = []
    for phase, receipt in receipts.items():
        task = next(task for task in tasks.values() if task["evaluation"]["kind"] == "gaussian_" + phase)
        for path in sorted((raw / phase).glob("*state.pt")):
            state = torch.load(path, weights_only=True, map_location="cpu")
            cap = state["trainer"].get("max_steps", state["recipe"]["total_steps"])
            context, trainer, _ = build(request, task, device, max_steps=cap)
            context.load_state_dict(state)
            assert state_digest(context.state_dict()) == state_digest(state), path
            assert all(parameter.device.type == "cuda" for model in (trainer.G, trainer.D, trainer.prior)
                       for parameter in model.parameters())
            assert trainer.initial_lrs == [[.012, .012 * 2.5], [.012 * 1.5]]
            for optimizer in (trainer.opt_g, trainer.opt_d):
                counts = [int(optimizer.state[parameter]["step"]) for group in optimizer.param_groups
                          for parameter in group["params"] if "step" in optimizer.state[parameter]]
                assert all(count == trainer.completed_steps for count in counts)
            restored.append(dict(path=str(path.relative_to(raw)), sha256=file_hash(path),
                                 state_sha256=state_digest(state), completed_steps=trainer.completed_steps,
                                 exact=True, device=device))
    return dict(passed=True, training_updates=0, model_sampling_draws=0, states=restored)


def render(frames_by_phase, tasks, receipts, output):
    import numpy as np
    from scipy.special import ndtr
    from .api_run import render_gif
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    edges = np.linspace(-1., 5.5, 53)
    media = {}
    for phase, observed in frames_by_phase.items():
        task = next(task for task in tasks.values() if task["evaluation"]["kind"] == "gaussian_" + phase)
        frames = []
        for index in np.linspace(0, len(observed) - 1, 9).round().astype(int):
            snapshot = observed[index]
            step, metrics = snapshot["step"], snapshot["metrics"]
            mean = 3. if step > 4000 else 2.
            target = np.diff(ndtr((edges - mean) / .5)) / np.diff(edges)
            samples = np.histogram(snapshot["samples"].numpy().reshape(-1), edges)[0] / 4096 / np.diff(edges)
            view = dict(kind="bar", title="Target and actual Gaussian histograms", target=target, samples=samples,
                        bin_centers=(edges[:-1] + edges[1:]) / 2, bin_width=float(edges[1] - edges[0]),
                        xlim=[-1., 5.5], ylim=[0., 2.5], xlabel="x", ylabel="density",
                        caption="Actual clean live public samples; numerical gates use every scheduled observation.")
            frames.append(dict(step=step, metrics=metrics, passed=not bounds(metrics), views=[view]))
        path = output / (phase + ".gif")
        render_gif(dict(id=task["id"], goal=task["description"], default_steps=task["execution"]["steps"]),
                   frames, path, full_budget=True, requested_steps=task["execution"]["steps"],
                   final_verdict=receipts[phase]["gaussian_grade"]["gate_status"])
        media[phase] = dict(path=path.name, sha256=file_hash(path), frames=len(frames))
    rows = [row for phase in receipts.values() for row in phase["evidence"]["observations"]]
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for axis, name, limits in zip(axes, ("mean_error_sigma", "std_ratio", "cdf_ks"), ((0., .2), (.8, 1.2), (0., .05))):
        axis.axhspan(*limits, color="#dff0de")
        axis.plot([row["step"] for row in rows], [row[name] for row in rows], color="#324b8c", linewidth=1.)
        for cutoff in (1000, 4000, 5000):
            axis.axvline(cutoff, linestyle="--", color="#777777", linewidth=.7)
        axis.set_ylabel(name.replace("_", " "))
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Completed updates; acquisition 1,000, target shift 4,000, reacquisition 5,000")
    figure.tight_layout()
    figure.savefig(output / "metrics.svg", metadata={"Date": None})
    plt.close(figure)
    return media


def publish(raw, output, protocol_path):
    result, request, tasks, receipts = load(raw)
    protocol = json.loads(protocol_path.read_text())
    counts, snapshots = recompute(raw, tasks, receipts)
    output.mkdir(parents=True, exist_ok=True)
    media = render(snapshots, tasks, receipts, output)
    smoke_rows = receipts["smoke"]["evidence"]["observations"]
    suffix = 0
    for row in reversed(smoke_rows):
        if bounds(row):
            break
        suffix += 1
    compact = {}
    for phase, receipt in receipts.items():
        compact[phase] = dict(grade=receipt["gaussian_grade"], cost=receipt["cost"],
                              final_metrics=receipt["evidence"]["live"], host=receipt["evidence"]["host"],
                              data_sha256=receipt["evidence"]["data_sha256"], guards=receipt["evidence"]["guards"],
                              recipe=receipt["recipe"], recipe_sha256=stable_hash(receipt["recipe"]),
                              initialization=receipt["initialization"], checkpoint=receipt["evidence"]["checkpoint"],
                              receipt_sha256=file_hash(raw / phase / "adapter-receipt.json"), gif=media[phase])
    updates = receipts["smoke"]["cost"]["completed_steps"]
    if "stability" in receipts:
        updates += receipts["stability"]["cost"]["completed_steps"] - 1000
    assert updates == protocol["budget"]["new_training_updates"]
    published = dict(schema_version=1, id=protocol["id"], scope=protocol["scope"], qualification_input=False,
                     protocol_sha256=file_hash(protocol_path), source={key: result["source"][key] for key in ("digest", "origin_commit")},
                     new_training_updates=updates,
                     new_adapter_loop_seconds=sum(receipt["cost"]["adapter_loop_seconds"] for receipt in receipts.values()),
                     original_five_terminal_acquisition_verdict="PASS" if suffix >= 5 else "FAIL",
                     original_five_terminal_acquisition_suffix=suffix, results=compact)
    atomic_json(output / "results.json", published)
    verification = dict(passed=True, training_updates=0, model_sampling_draws=0,
                        saved_sample_recomputations=counts, grades_recomputed=len(receipts),
                        publisher_sha256=file_hash(Path(__file__)))
    atomic_json(output / "verification.json", verification)
    print(json.dumps(dict(published=protocol["id"], sample_recomputations=counts,
                          smoke=compact["smoke"]["grade"]["gate_status"],
                          stability=compact.get("stability", {}).get("grade", {}).get("gate_status"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--verify-state", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.verify_state:
        proof = restore(args.raw, device=args.device)
        atomic_json(args.output / "restore-proof.json", proof)
        print(json.dumps(dict(exact_cuda_restores=len(proof["states"]), passed=proof["passed"])), flush=True)
    else:
        publish(args.raw, args.output, args.protocol)


if __name__ == "__main__":
    main()

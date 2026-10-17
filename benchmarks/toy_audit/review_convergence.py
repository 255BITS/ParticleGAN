"""Review PR226/227's retained evidence and render actual training observations.

This reads a pinned proposal checkout and existing artifacts. It never calls a
training update. Run each proposal in a separate process so imports stay bound
to its source. Raw artifacts and review output belong outside Git.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


HEADS = {226: "0e69fa60bf151762f8edc49bb714c6b7f8cf4804",
         227: "b0e4f420856d7607baa98cbef488e569d20a3f31"}
BASE = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
COLORS = ["#287da8", "#c14b58", "#16866c", "#9570ad"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fafafa",
                     "axes.facecolor": "#fafafa"})


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def frame(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=128)


def save(frames, path, steps, law):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[260] * (len(frames) - 1) + [2400], loop=0, optimize=True)
    frames[-1].convert("RGB").save(path.with_suffix(".png"))
    return dict(gif=path.name, sha256=sha(path), bytes=path.stat().st_size,
                frames=len(frames), steps=steps, interpolation=False,
                poster=path.with_suffix(".png").name,
                poster_sha256=sha(path.with_suffix(".png")), sampling_law=law)


def sources(root, number):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != HEADS[number]:
        raise ValueError("proposal checkout differs from the reviewed head")
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE, head], cwd=root, text=True).splitlines()
    hashes = {name: sha(root / name) for name in changed}
    for name, digest in hashes.items():
        committed = subprocess.check_output(["git", "show", head + ":" + name], cwd=root)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError("reviewed proposal source changed: " + name)
    if subprocess.run(["git", "diff", "--quiet", BASE, "--", "particlegan"], cwd=root).returncode:
        raise ValueError("proposal changed the native package")
    return hashes


def paired(args):
    measured = read(args.source / "reports/paired-residual-toy/measurements.json")
    for name, expected in measured["source_hashes"].items():
        assert sha(args.source / name) == expected, name
    bindings = {name: sha(args.artifacts / name) for name in measured["raw_bindings"]}
    assert bindings == measured["raw_bindings"]
    result = read(args.artifacts / "result.json")
    assert result["status"] == "completed"
    names = ["current", "even_critic", "d_antithetic"]
    labels = ["Current", "Even critic", "D antithetic"]
    observations = [rows(args.artifacts / name / "metrics.jsonl") for name in names]
    updates = [rows(args.artifacts / name / "updates.jsonl") for name in names]
    steps = [r["step"] for r in observations[0]]
    assert steps == [0, 10, 25, 50, *range(100, 1201, 100)]
    assert all([r["step"] for r in arm] == steps for arm in observations)
    for i, name in enumerate(names):
        assert len(updates[i]) == 1200
        assert all(r["step"] == r["ka2_calls"] == j + 1 for j, r in enumerate(updates[i]))
        assert all(r["caller_draw_chain_sha256"] == updates[0][j]["caller_draw_chain_sha256"]
                   for j, r in enumerate(updates[i]))
        by_step = {r["step"]: r for r in observations[i]}
        for point in measured["profiles"][name]["curve"]:
            row = by_step[point["step"]]
            for key in ("served", "live"):
                close(row["pools"]["report"][key]["normalized_rmse"], point[key + "_rmse"])
            close(row["diagnostics"]["zero_residual_G_odd_force_norm"], point["zero_residual_odd_force"])
        assert all(r["all_owners_and_rng_unchanged"] for r in observations[i])
        half = next((r["step"] for r in observations[i] if r["report_rmse_ratio"]["served"] <= .5), None)
        assert half == measured["profiles"][name]["first_recorded_half_initial_rmse_step"]
    bias_bound = max(abs(v) for arm in observations for row in arm
                     for v in row["pools"]["report"]["served"]["bias"]) * 1.1
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), dpi=100)
    fig.subplots_adjust(left=.085, right=.97, top=.78, bottom=.23, hspace=.58, wspace=.27)
    frames = []
    for index, step in enumerate(steps):
        for ax in axes.flat:
            ax.clear(); ax.grid(alpha=.2); ax.set_xlim(0, 1200)
            ax.axvline(800, color="#777", ls=":", lw=1)
            ax.set_xlabel("Completed updates")
        for arm, label, color in zip(observations, labels, COLORS):
            prefix = arm[:index + 1]
            x = [r["step"] for r in prefix]
            errors = [r["pools"]["report"]["served"] for r in prefix]
            axes[0, 0].plot(x, [r["normalized_rmse"] for r in errors], color=color, label=label)
            for channel in range(2):
                style = "-" if channel == 0 else "--"
                axes[0, 1].plot(x, [r["bias"][channel] for r in errors], color=color, ls=style)
                axes[1, 1].plot(x, [r["diagnostics"]["energy_coefficients"][channel] for r in prefix], color=color, ls=style)
            axes[1, 0].plot(x, [r["diagnostics"]["zero_residual_G_odd_force_norm"] for r in prefix], color=color)
        axes[0, 0].axhline(measured["profiles"]["current"]["initial_report_rmse"] / 2, color="#777", ls="--", lw=1)
        axes[0, 0].set_ylim(0, .24); axes[0, 0].set_title("Clean reporting RMSE · dashed = half initial error")
        axes[0, 1].set_ylim(-bias_bound, bias_bound); axes[0, 1].axhline(0, color="#777", lw=.7)
        axes[0, 1].set_title("Residual mean · solid/dashed = channels 1/2")
        axes[1, 0].set_ylim(-.0001, .002); axes[1, 0].set_title("Generator odd force at zero residual · output L2")
        axes[1, 1].set_ylim(-.16, .01); axes[1, 1].set_title("Learned quadratic critic weights")
        fig.suptitle(f"PR226 · Can an asymmetric critic keep pushing a correct generator?\nRecorded evaluation {step} / 1,200 · paired GAN-only native E22", y=.98, fontsize=14)
        if index == 0:
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .88), ncol=3)
        now = [arm[index]["pools"]["report"]["served"]["normalized_rmse"] for arm in observations]
        text = fig.text(.5, .055, "RMSE: " + "  ·  ".join(f"{name} {value:.6f}" for name, value in zip(labels, now)) +
                        "\nClean served = live here. Dotted line: KA2 blend starts at 800. No absolute accuracy gate.\n"
                        "Actual recorded metrics; intermediate output tensors were not retained. Affine path can solve the task.",
                        ha="center", fontsize=9)
        frames.append(frame(fig)); text.remove()
    media = save(frames, args.media / "pr226.gif", steps, "clean stock-selected served output; live matches; disjoint reporting grid")
    plt.close(fig)
    return dict(pr=226, status="reviewed", retained_artifact_files_verified=len(bindings),
                artifact_bindings_sha256=identity(bindings), measured_runtime=measured["runtime"],
                retained_metric_rows=48, matched_update_records_per_profile=1200,
                software_tests="25 passed independently on Python3.12 / Torch2.13",
                final_metrics={name: measured["profiles"][name]["final_served"] for name in names},
                first_recorded_half_error={name: measured["profiles"][name]["first_recorded_half_initial_rmse_step"] for name in names},
                comparison=measured["comparison"], media=media, new_training_updates=0)


@torch.no_grad()
def routed(args):
    base = importlib.import_module("examples.e22_routed_convergence")
    neutral = importlib.import_module("examples.e22_routed_convergence_neutral")
    assert Path(base.__file__).resolve() == args.source / "examples/e22_routed_convergence.py"
    spec = importlib.util.spec_from_file_location("toy_audit_pr227_regression", args.source / "tests/test_e22_routed_convergence_long.py")
    regression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regression)
    parent, candidate = args.artifacts, args.candidate
    receipt = read(parent / "receipt.json")
    recovered = read(candidate / "recovered-evaluation-receipt.json")
    assert receipt["status"] == "complete" and recovered["status"] == "complete_recovered_evaluation"
    for name, expected in receipt["bindings"]["source_hashes"].items():
        assert sha(args.source / name) == expected, name
    for name, expected in recovered["original_artifact_sha256"].items():
        assert sha(candidate / name) == expected, name
    repair = importlib.import_module("examples.evaluate_e22_routed_convergence_neutral")
    proof = repair.observational_source_repair((candidate / "training-source-504df385.py").read_text(),
                                             (args.source / "examples/e22_routed_convergence_neutral.py").read_text())
    assert proof["training_ast_unchanged"]
    original = base.make_data()
    modified = neutral.make_neutral_data(original)
    assert original["digest"] == receipt["data_digest"] and modified["digest"] == recovered["data_digest"]
    panels = base.evaluation_panels(original)
    assert base.digest(panels) == receipt["private_panel_digest"] == recovered["private_panel_digest"]
    judges = {}
    bindings = {}
    for name, expected in receipt["judges"].items():
        arm, step = name.split("@")
        path = parent / arm / f"step-{int(step):04d}.pt"
        state = torch.load(path, weights_only=False)
        judge = base.ConditionalCritic(original["scale"]).eval().requires_grad_(False)
        judge.load_state_dict(state["training"]["models"]["critic"], strict=True)
        assert base.digest(judge.state_dict()) == expected
        judges[name] = judge
    actual = regression._assert_registered_game_regression(parent, candidate)
    original_game = regression._game
    def harmful_code_control(loop, judge, panel, *, ablate=False):
        return original_game(loop, judge, panel, ablate=False) - (.01 if ablate else 0)
    try:
        regression._game = harmful_code_control
        false_acceptance = regression._assert_registered_game_regression(parent, candidate)
        assert all(r["zero_code_minus_live"] < 0 for r in false_acceptance.values())
    finally:
        regression._game = original_game
    arm_names = [*base.ARMS[:2], "neutral_particle", base.ARMS[2]]
    labels = ["Ordinary native", "Original particles", "Particles: initial H/b = 0", "MSE reference"]
    recorded = {name: {} for name in arm_names}
    for row in rows(parent / "common-judge-curves.jsonl"):
        recorded[row["arm"]][row["step"]] = row["scores"]
    for row in rows(candidate / "recovered-common-judge-curves.jsonl"):
        recorded["neutral_particle"][row["step"]] = row["scores"]
    steps = list(range(0, 6401, 200))
    pictures, rmse, scores = {}, {}, {}
    max_difference = 0.
    for arm in arm_names:
        path = candidate if arm == "neutral_particle" else parent / arm
        law_state = torch.load(path / "step-0000.pt", weights_only=False)
        loop = (neutral.make_neutral_loop(modified, bindings=law_state["law"]["bindings"])
                if arm == "neutral_particle" else base.make_loop(arm, original, bindings=law_state["law"]["bindings"]))
        expected_files = recovered["checkpoints"] if arm == "neutral_particle" else receipt["arms"][arm]["checkpoints"]
        for entry in expected_files:
            assert sha(path / entry["file"]) == entry["sha256"]
        pictures[arm], rmse[arm], scores[arm] = [], [], []
        for step in steps:
            file = path / f"step-{step:04d}.pt"
            bindings[arm + "/" + file.name] = sha(file)
            state = torch.load(file, weights_only=False)
            base.restore(loop, state)
            assert base.digest(base.checkpoint(loop)) == base.digest(state)
            before = base.digest(base.checkpoint(loop))
            values = loop.data["test"]
            prediction = torch.cat([base.forward(loop, values["context"][i:i + base.BATCH_SIZE])
                                    for i in range(0, len(values["context"]), base.BATCH_SIZE)])
            residual = (prediction - values["targets"]) / original["scale"]
            condition = values["context"][:, 0, base.WIDTH:]
            measured = {}
            for name, judge in judges.items():
                score = float(torch.stack([F.softplus(judge(panel, condition) - judge(panel + residual, condition)).mean()
                                           for panel in panels["test"]]).mean())
                expected = recorded[arm][step][name]["test"]["paired_game"]
                close(score, expected); max_difference = max(max_difference, abs(score - expected))
                measured[name] = score
            error = float((prediction - values["targets"]).square().mean().sqrt())
            close(error, next(iter(recorded[arm][step].values()))["test"]["output_rmse_diagnostic"])
            target_edit = ((values["targets"] - values["base"]) / original["scale"]).flatten()[::128].numpy()
            model_edit = ((prediction - values["base"]) / original["scale"]).flatten()[::128].numpy()
            pictures[arm].append((target_edit.copy(), model_edit.copy()))
            rmse[arm].append(error); scores[arm].append(measured)
            assert base.digest(base.checkpoint(loop)) == before
        # The replay witness has no corresponding curve frame; still restore its actual state.
        extra = path / "step-0802.pt"
        state = torch.load(extra, weights_only=False)
        base.restore(loop, state)
        assert base.digest(base.checkpoint(loop)) == base.digest(state)
        bindings[arm + "/" + extra.name] = sha(extra)
    bound = max(abs(v).max() for arm in pictures.values() for pair in arm for v in pair) * 1.05
    maximum = max(v for arm in scores.values() for row in arm for v in row.values()) * 1.04
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 7.4), dpi=100)
    fig.subplots_adjust(left=.055, right=.985, top=.75, bottom=.21, hspace=.60, wspace=.30)
    frames = []
    for index, step in enumerate(steps):
        for i, (arm, label, color) in enumerate(zip(arm_names, labels, COLORS)):
            ax = axes[0, i]; ax.clear()
            ax.plot([-bound, bound], [-bound, bound], color="#888", lw=1, ls=":")
            ax.scatter(*pictures[arm][index], s=8, c=color, alpha=.48)
            ax.set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel="Required teacher edit", ylabel="Actual model edit",
                   title=label + f"\nRMSE {rmse[arm][index]:.6f}")
            ax.grid(alpha=.15)
        for j, judge in enumerate(judges):
            ax = axes[1, j]; ax.clear()
            for arm, label, color in zip(arm_names, labels, COLORS):
                ax.plot(steps[:index + 1], [r[judge] for r in scores[arm][:index + 1]], color=color,
                        ls="--" if arm == "ordinary_mse_reference" else "-", label=label)
            ax.axhline(np.log(2), color="#777", ls=":", lw=1)
            ax.set(xlim=(0, 6400), ylim=(.64, maximum), xlabel="Completed updates", ylabel="Paired generator game ↓",
                   title=judge.replace("_native_game", " critic ").replace("@", "@ step "))
            ax.grid(alpha=.2)
        fig.suptitle(f"PR227 · Does initial hidden modulation cause the particle convergence gap?\n"
                     f"Actual saved states {step} / 6,400 · fixed teacher-aligned target and four common judges", y=.98, fontsize=14)
        if index == 0:
            fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .88), ncol=4)
        text = fig.text(.5, .057, "Clean FAST predictions; fixed every-128th-coordinate display; metrics use all 96 held-out contexts.\n"
                        "Dotted diagonal = exact teacher edit. Dotted game line = teacher log(2) anchor, not an accuracy gate.\n"
                        "All four prespecified critics retained. MSE/AdamW is a separate historical objective. No Atlas substitution.",
                        ha="center", fontsize=9)
        frames.append(frame(fig)); text.remove()
    media = save(frames, args.media / "pr227.gif", steps, "clean FAST; no DV12/output noise in predictions; fixed private paired score panels sigma0.125")
    plt.close(fig)
    return dict(pr=227, status="reviewed", retained_checkpoint_files_verified=len(bindings),
                artifact_bindings_sha256=identity(bindings), measured_runtime=receipt["runtime"],
                independently_rescored_test_rows=132, mandatory_common_judges=4,
                maximum_absolute_game_score_difference=max_difference,
                software_tests="14 passed, 1 opt-in learned test skipped; tensor assertion separately passed retained artifacts",
                final_metrics=actual, observational_repair=proof,
                original_scoring_failure_preserved=True, owned_state_and_RNG_unchanged=True,
                synthetic_gate_control=dict(harmful_particle_codes_accepted=True,
                    zero_code_minus_live_by_judge={name: r["zero_code_minus_live"] for name, r in false_acceptance.items()},
                    note="Synthetic scoring control exposes abs(delta) accepting the wrong direction; not an alternate trained model."),
                media=media, new_training_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int, choices=[226, 227])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    args = parser.parse_args()
    if args.pr == 227 and args.candidate is None:
        parser.error("PR227 requires its retained intervention artifacts")
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = sources(args.source, args.pr)
    sys.path.insert(0, str(args.source))
    torch.set_num_threads(1)
    try:
        review = paired(args) if args.pr == 226 else routed(args)
        review.update(head_sha=HEADS[args.pr], base_sha=BASE, source_sha256=hashes,
                      review_runtime=dict(python=sys.version.split()[0], torch=torch.__version__, device="cpu"))
        (args.output / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in review.items() if k in ("pr", "status", "media", "new_training_updates")}), flush=True)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(dict(pr=args.pr, error=repr(error), new_training_updates=0), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()

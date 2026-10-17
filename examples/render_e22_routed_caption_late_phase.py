"""Render saved late-phase toy observations and curves on CPU, within60s.

CUDA_VISIBLE_DEVICES='' python -m examples.render_e22_routed_caption_late_phase --run-directory RUN
Optional Pillow/NumPy/Matplotlib. No models, ParticleGAN API calls or updates.
"""
import time
STARTED = time.monotonic()
import argparse
import json
import math
from pathlib import Path

import torch
from PIL import ImageDraw, ImageFont
from examples import render_e22_routed_caption_untied as frames

TASK = "routed_caption_late_phase_v1"
STEPS = (0, 256, 512, 768, 800, 1024, 1536, 2048)
ENDPOINTS = (512, 768, 800, 1024, 1536, 2048)
ARMS = frames.ARMS
LIMIT = 60
ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_routed_caption_late_phase_v1.json"
OUTPUTS = ("goal.gif", "goal-final.png", "metrics.png", "media-completion.json")


def budget():
    if time.monotonic() - STARTED > LIMIT:
        raise TimeoutError("saved-state CPU rendering startup-through-final-writes60s exceeded")


def finite_metric(metric):
    return (set(metric["by_source"]) == {str(s) for s in range(6)}
            and metric["contexts"] == 48 and metric["coordinates_per_context"] == 4096
            and metric["source_counts"] == {str(s): 8 for s in range(6)}
            and all(math.isfinite(v) and v >= 0 for v in (metric["rmse"], *metric["by_source"].values())))


def validate_observations(report, receipt, media, curves, hashes, card):
    """Pure retained-artifact validation; no policy, initializer or native import."""
    if (report["task"] != TASK or receipt["task"] != TASK or card["task"] != TASK
        or not report["complete"] or not receipt["complete"]
        or report["scientific_status"] not in ("PASS", "FAIL")
        or report["scientific_status"] != ("PASS" if report["gate"]["pass"] else "FAIL")
        or receipt["scientific_status"] != report["scientific_status"]
        or receipt["report_sha256"] != hashes["report_sha256"]
        or report["protocol_sha256"] != receipt["protocol_sha256"] or report["protocol_sha256"] != hashes["protocol_sha256"]
        or report["observed_media_sha256"] != hashes["observed_media_sha256"]
        or report["curve_residual_sha256"] != hashes["curve_residual_sha256"]
        or report["source_identity"] != receipt["source_identity"] or report["source_identity"] != card["sources"]
        or report["imported_package"] != receipt["imported_package"] or not report["imported_package_unchanged"]
        or card["steps"] != 2048 or report["quality_updates"] != 6144 or report["replay_updates"] != 0
        or tuple(card["media_steps"]) != STEPS or tuple(report["media_steps"]) != STEPS
        or tuple(card["endpoint_steps"]) != ENDPOINTS or tuple(report["endpoint_steps"]) != ENDPOINTS
        or report["live"]["denominator"] != 2047 or card["accuracy_thresholds"]["live_denominator"] != 2047):
        raise ValueError("completed hash-bound late-phase2048 protocol required")
    if (tuple(media["steps"]) != STEPS or tuple(media["indices"]) != tuple(card["media_indices"])
        or list(media["source_ids"][:6]) != list(range(6)) or not media["capture_native_state_rng_diagnostics_unchanged"]
        or tuple(media["target_residual"].shape) != (8, 256, 16) or media["target_residual"].count_nonzero()
        or set(media["actual_residuals"]) != set(ARMS) or set(report["curves"]) != set(ARMS)
        or tuple(curves["steps"]) != ENDPOINTS or set(curves["physical_residuals"]) != set(ARMS)):
        raise ValueError("fixed source/target/media/curve law differs")
    raw_ids = curves["source_ids"]
    if isinstance(raw_ids, torch.Tensor):
        if raw_ids.device.type != "cpu" or tuple(raw_ids.shape) != (48,):
            raise ValueError("CPU source IDs required")
        ids = raw_ids.tolist()
    elif isinstance(raw_ids, (list, tuple)):
        ids = list(raw_ids)
    else:
        raise ValueError("integer source ID sequence required")
    if (len(ids) != 48 or any(type(s) is not int for s in ids)
        or ids != [s for s in range(6) for _ in range(8)]):
        raise ValueError("TEST48/eight contexts per source required")
    source_ids = torch.as_tensor(ids, dtype=torch.long)
    for arm in ARMS:
        if (set(media["actual_residuals"][arm]) != {str(s) for s in STEPS}
            or set(report["curves"][arm]) != {str(s) for s in ENDPOINTS}
            or set(curves["physical_residuals"][arm]) != {str(s) for s in ENDPOINTS}):
            raise ValueError("all declared observations and full TEST endpoints mandatory")
        for value in media["actual_residuals"][arm].values():
            if value.device.type != "cpu" or tuple(value.shape) != (8, 256, 16) or not torch.isfinite(value).all():
                raise ValueError("finite CPU physical camera observations required")
        for step in ENDPOINTS:
            value = curves["physical_residuals"][arm][str(step)]
            metric = report["curves"][arm][str(step)]
            if value.device.type != "cpu" or tuple(value.shape) != (48, 256, 16) or not torch.isfinite(value).all() or not finite_metric(metric):
                raise ValueError("finite full TEST48 physical scores required")
            errors = value.double().square().flatten(1).mean(1)
            actual = {"rmse": float(errors.mean().sqrt()),
                      "by_source": {str(s): float(errors[source_ids == s].mean().sqrt()) for s in range(6)}}
            if (abs(actual["rmse"] - metric["rmse"]) > 2e-15
                or any(abs(actual["by_source"][s] - metric["by_source"][s]) > 2e-15 for s in actual["by_source"])
                or not torch.equal(media["actual_residuals"][arm][str(step)], value[list(media["indices"])])):
                raise ValueError("camera selection or full physical curve score differs")
        if report["accuracy"][arm] != report["curves"][arm]["2048"]:
            raise ValueError("terminal labels must use the fixed2048 full TEST score")


def read_traces(run, report):
    rows = {}
    hashes = {}
    if set(report["trace_sha256"]) != set(ARMS):
        raise ValueError("all three native trace hashes required")
    for arm in ARMS:
        path = run / f"{arm}.jsonl"
        hashes[arm] = frames.sha(path)
        if hashes[arm] != report["trace_sha256"][arm]:
            raise ValueError("native loss/phase trace bytes differ")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != 2048 or [r["step"] for r in records] != list(range(1, 2049)):
            raise ValueError("complete actual2048 native trace required")
        counts, first_blend = {}, None
        for row in records:
            if not all(math.isfinite(row[k]) for k in ("loss_g", "loss_d_game")):
                raise ValueError("finite native losses required")
            phase = row["phase_observation"]["penalty_last_stats"].get("phase")
            key = "unavailable" if phase is None else str(phase)
            counts[key] = counts.get(key, 0) + 1
            if phase == "blend" and first_blend is None:
                first_blend = row["step"]
        summary = report["phase_summary"][arm]
        if (counts != summary["phase_counts"] or first_blend != summary["first_blend_step"]
            or summary["actual_blend_observed"] != (first_blend is not None)
            or records[-1]["phase_observation"] != summary["last_observation"]):
            raise ValueError("recorded phase summary differs from actual native trace")
        rows[arm] = records
        budget()
    return rows, hashes


def goal_frame(*args, **kwargs):
    """Reuse immutable patch/color math; replace both512-specific captions."""
    image = frames.goal_frame(*args, **kwargs)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 42), fill="white")
    draw.text((20, 15), f"Toy late phase, not Supra | native update {kwargs['step']}/2048 per arm",
              fill="#111111", font=ImageFont.load_default(size=17))
    draw.rectangle((0, 800, image.width, image.height), fill="white")
    small = ImageFont.load_default(size=14)
    draw.text((20, 806), f"All frames: white=0, red={kwargs['vmax']:.6f} physical error. Six fixed TEST cameras; score uses48.",
              fill="#333333", font=small)
    t = kwargs["terminal"]
    label = (f"Fixed2048 {t['scientific_status']}: ordinary {t['ordinary']:.8f}, shared {t['shared']:.8f}, untied {t['untied']:.8f}"
             if kwargs["step"] == 2048 else "Visual observations only; numerical PASS/FAIL uses the fixed2048 endpoint.")
    draw.text((20, 832), label, fill="#111111", font=small)
    return image


def metrics_figure(report, traces):
    """Matplotlib is optional and loaded only by the actual plotting path."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    try:
        colors = ("#2a61b8", "#278351", "#b74437")
        labels = ("Ordinary BF16 LoRA", "Shared-Up particles", "Untied-Up particles")
        for arm, color, label in zip(ARMS, colors, labels):
            axes[0].plot(ENDPOINTS, [report["curves"][arm][str(s)]["rmse"] for s in ENDPOINTS], marker="o", color=color, label=label)
            steps = [r["step"] for r in traces[arm]]
            axes[1].plot(steps, [r["loss_g"] for r in traces[arm]], color=color, alpha=.65, linewidth=.7, label=label)
            axes[2].plot(steps, [r["loss_d_game"] for r in traces[arm]], color=color, alpha=.65, linewidth=.7, label=label)
        markers = {}
        for arm in ARMS:
            step = report["phase_summary"][arm]["first_blend_step"]
            if step is not None:
                markers.setdefault(step, []).append(arm)
        for step, arms in markers.items():
            name = "all arms" if len(arms) == len(ARMS) else ", ".join(labels[ARMS.index(a)] for a in arms)
            for axis in axes:
                axis.axvline(step, color="#555555", linestyle="--", linewidth=.8)
            axes[0].annotate(f"First observed KA2 blend: {step} ({name})", xy=(step, .99),
                             xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", fontsize=8, va="top")
        axes[0].set_title("Physical full TEST48 RMSE: offline only, lower is better")
        axes[0].set_ylabel("Physical RMSE")
        axes[0].legend(fontsize=8)
        axes[1].set_title("Recorded native generator game loss; no held-out feedback")
        axes[1].set_ylabel("Native G loss")
        axes[2].set_title("Recorded native critic game loss; penalty excluded")
        axes[2].set_ylabel("Native D game loss")
        axes[2].set_xlabel("Native editing update per arm")
        for axis in axes:
            axis.grid(alpha=.2)
            axis.set_xlim(0, 2048)
        detail = "Observed phase markers only" if markers else "Penalty phase unavailable; no transition marker inferred"
        fig.suptitle(f"Toy late phase, not Supra | fixed2048 {report['scientific_status']}\n{detail}")
        return fig
    except BaseException:
        plt.close(fig)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CARD)
    args = parser.parse_args(); run = args.run_directory
    source = frames.sha(__file__); before = torch.get_rng_state().clone()
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    writable = False; report = receipt = completion = error = None; code = 2
    try:
        if any((run / name).exists() for name in OUTPUTS):
            raise ValueError("preserve prior media; fresh unrendered completed run required")
        if not run.is_dir():
            raise ValueError("completed run directory required")
        writable = True
        paths = {"report_sha256": run / "report.json", "completion_sha256": run / "completion.json",
                 "observed_media_sha256": run / "observed-media.pt", "curve_residual_sha256": run / "curve-residuals.pt",
                 "protocol_sha256": args.protocol}
        hashes = {k: frames.sha(p) for k, p in paths.items()}
        report = json.loads(paths["report_sha256"].read_text()); receipt = json.loads(paths["completion_sha256"].read_text())
        card = json.loads(args.protocol.read_text())
        media = torch.load(paths["observed_media_sha256"], map_location="cpu", weights_only=True)
        curves = torch.load(paths["curve_residual_sha256"], map_location="cpu", weights_only=True)
        validate_observations(report, receipt, media, curves, hashes, card)
        if any(frames.sha(ROOT / name) != digest for name, digest in card["sources"].items()):
            raise ValueError("declared source/helper bytes changed")
        traces, trace_hashes = read_traces(run, report)
        vmax = max(float(frames.token_maps(media["actual_residuals"][arm]["0"]).max()) for arm in ARMS)
        terminal = {"scientific_status": report["scientific_status"],
                    **{k: report["accuracy"][arm]["rmse"] for k, arm in zip(("ordinary", "shared", "untied"), ARMS)}}
        observed = []
        for step in STEPS:
            observed.append(goal_frame(media["target_residual"], *(media["actual_residuals"][a][str(step)] for a in ARMS),
                                       step=step, vmax=vmax, terminal=terminal)); budget()
        gif, png, plot = (run / n for n in OUTPUTS[:3])
        with gif.open("xb") as handle:
            observed[0].save(handle, format="GIF", save_all=True, append_images=observed[1:], duration=900, loop=0, disposal=2)
        with png.open("xb") as handle:
            observed[-1].save(handle, format="PNG")
        figure = metrics_figure(report, traces)
        try:
            with plot.open("xb") as handle:
                figure.savefig(handle, format="png", dpi=130)
        finally:
            from matplotlib import pyplot as plt
            plt.close(figure)
        budget()
        if (frames.sha(__file__) != source or any(frames.sha(path) != hashes[key] for key, path in paths.items())
            or any(frames.sha(run / f"{arm}.jsonl") != digest for arm, digest in trace_hashes.items())
            or any(frames.sha(ROOT / name) != digest for name, digest in card["sources"].items())):
            raise AssertionError("source/card/retained observation/trace bytes changed")
        if torch.cuda.is_initialized():
            raise AssertionError("saved-state CPU renderer initialized CUDA")
        completion = {"task": TASK, "complete": True, **hashes, "trace_sha256": trace_hashes,
            "renderer_source_sha256": source, "pure_frame_helper_sha256": frames.sha(frames.__file__),
            "gif_sha256": frames.sha(gif), "png_sha256": frames.sha(png), "metrics_png_sha256": frames.sha(plot),
            "media_steps": list(STEPS), "curve_steps": list(ENDPOINTS), "fixed_initial_only_color_limit": vmax,
            "observed_first_blend_steps": {a: report["phase_summary"][a]["first_blend_step"] for a in ARMS},
            "producer_imported_package": report["imported_package"], "native_updates": 0, "host_forwards": 0,
            "model_constructions": 0, "ParticleGAN_API_calls": 0, "CUDA_initialized": False,
            "scientific_status_unchanged": report["scientific_status"], "scope": "Saved toy observations only; no Supra score or causal phase conclusion.",
            "seconds": time.monotonic()-STARTED, "limit_seconds": LIMIT}; code = 0
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        unchanged = torch.equal(before, torch.get_rng_state())
        torch.set_rng_state(before); torch.set_num_threads(threads)
        if not unchanged:
            error = {"type": "AssertionError", "message": "CPU renderer changed caller RNG"}
        if time.monotonic()-STARTED > LIMIT:
            error = {"type": "TimeoutError", "message": "CPU cleanup60s exceeded"}
        if error is not None:
            completion = {"task": TASK, "complete": False, "error": error, "seconds": time.monotonic()-STARTED,
                          "limit_seconds": LIMIT, "renderer_source_sha256": source}; code = 2
        if completion is not None:
            completion["global_CPU_RNG_unchanged"] = unchanged
        if writable:
            path = run / "media-completion.json"
            try:
                with path.open("x") as handle:
                    handle.write(json.dumps(completion, indent=2, allow_nan=False)+"\n")
            except FileExistsError:
                return 2
            if time.monotonic()-STARTED > LIMIT:
                completion.update(complete=False, error={"type": "TimeoutError", "message": "final receipt60s exceeded"}, seconds=time.monotonic()-STARTED)
                path.write_text(json.dumps(completion, indent=2, allow_nan=False)+"\n"); code = 2
        print(json.dumps(completion, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Render the new correlation-only caption toy's saved native observations, CPU<=60s.

python -m examples.render_e22_routed_caption_flow --run-directory RUN
Pure immutable PR240 frame math; no model/native API calls or CUDA.
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

TASK = "routed_caption_correlated_context_v1"
LIMIT = 60
CARD = Path(__file__).resolve().parents[1] / "docs/e22_routed_caption_flow_v1.json"


def budget():
    if time.monotonic() - STARTED > LIMIT: raise TimeoutError("saved-state CPU render60s exceeded")


def validate_observations(report, receipt, media, hashes, card):
    if (report["task"] != TASK or receipt["task"] != TASK or card["task"] != TASK
        or not report["complete"] or not receipt["complete"]
        or report["scientific_status"] not in ("PASS", "FAIL")
        or report["scientific_status"] != ("PASS" if report["gate"]["pass"] else "FAIL")
        or receipt["scientific_status"] != report["scientific_status"]
        or receipt["report_sha256"] != hashes["report_sha256"]
        or report["protocol_sha256"] != hashes["protocol_sha256"]
        or receipt["protocol_sha256"] != hashes["protocol_sha256"]
        or report["observed_media_sha256"] != hashes["observed_media_sha256"]
        or report["source_identity"] != receipt["source_identity"]
        or report["source_identity"] != card["sources"]
        or report["flow"]["law"] != card["flow"]
        or report["flow"]["normalization"]["law"] != card["normalization"]
        or report["flow"]["normalization"]["epsilon_floor_fraction"] != 0
        or not report["flow"]["scale_recomputed_from_new_untrained_fit"]):
        raise ValueError("completed hash-bound NEW correlated-context protocol required")
    if not all(math.isfinite(report["accuracy"][arm]["rmse"]) and report["accuracy"][arm]["rmse"] >= 0 for arm in frames.ARMS):
        raise ValueError("finite nonnegative physical terminal label metrics required")
    if (tuple(media["steps"]) != frames.STEPS or tuple(media["indices"]) != tuple(card["media_indices"])
        or list(media["source_ids"][:6]) != list(range(6))
        or not media["capture_native_state_rng_diagnostics_unchanged"]
        or tuple(media["target_residual"].shape) != (8, 256, 16)
        or media["target_residual"].count_nonzero()
        or set(media["actual_residuals"]) != set(frames.ARMS)):
        raise ValueError("fixed source/target/observation law differs")
    for arm in frames.ARMS:
        if set(media["actual_residuals"][arm]) != {str(s) for s in frames.STEPS}:
            raise ValueError("every declared native observation mandatory")
        for value in media["actual_residuals"][arm].values():
            if tuple(value.shape) != (8, 256, 16) or not torch.isfinite(value).all():
                raise ValueError("full finite physical observations required")


def goal_frame(*args, **kwargs):
    image = frames.goal_frame(*args, **kwargs)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1230, 42), fill="white")
    draw.text((20, 15), f"Correlated-context toy | native update {kwargs['step']}/512 per arm",
              fill="#111111", font=ImageFont.load_default(size=17))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CARD)
    args = parser.parse_args(); run = args.run_directory
    source = frames.sha(__file__); before = torch.get_rng_state().clone()
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    writable = False; completion = error = None; code = 2
    try:
        if any((run / name).exists() for name in ("goal.gif", "goal-final.png", "media-completion.json")):
            raise ValueError("preserve prior media; fresh unrendered completed run required")
        if not run.is_dir(): raise ValueError("completed run directory required")
        writable = True
        rp, cp, mp = run / "report.json", run / "completion.json", run / "observed-media.pt"
        report = json.loads(rp.read_text()); receipt = json.loads(cp.read_text())
        card = json.loads(args.protocol.read_text())
        hashes = {"report_sha256": frames.sha(rp), "completion_sha256": frames.sha(cp),
                  "observed_media_sha256": frames.sha(mp), "protocol_sha256": frames.sha(args.protocol)}
        media = torch.load(mp, map_location="cpu", weights_only=True)
        validate_observations(report, receipt, media, hashes, card)
        root = Path(__file__).resolve().parents[1]
        if any(frames.sha(root / name) != digest for name, digest in card["sources"].items()):
            raise ValueError("declared source/helper bytes changed")
        vmax = max(float(frames.token_maps(media["actual_residuals"][a]["0"]).max()) for a in frames.ARMS)
        terminal = {"scientific_status": report["scientific_status"],
                    **{k: report["accuracy"][a]["rmse"] for k, a in zip(("ordinary", "shared", "untied"), frames.ARMS)}}
        observed = []
        for step in frames.STEPS:
            observed.append(goal_frame(media["target_residual"], *(media["actual_residuals"][a][str(step)] for a in frames.ARMS),
                                       step=step, vmax=vmax, terminal=terminal)); budget()
        gif, png = run / "goal.gif", run / "goal-final.png"
        with gif.open("xb") as handle:
            observed[0].save(handle, format="GIF", save_all=True, append_images=observed[1:], duration=900, loop=0, disposal=2)
        with png.open("xb") as handle: observed[-1].save(handle, format="PNG")
        budget()
        if (frames.sha(__file__) != source or any(frames.sha(path) != hashes[key] for path, key in
             ((rp, "report_sha256"), (cp, "completion_sha256"), (mp, "observed_media_sha256"), (args.protocol, "protocol_sha256")))
             or any(frames.sha(root / name) != digest for name, digest in card["sources"].items())):
            raise AssertionError("source/card/retained observations changed")
        if torch.cuda.is_initialized(): raise AssertionError("CPU renderer initialized CUDA")
        completion = {"task": TASK, "complete": True, **hashes, "renderer_source_sha256": source,
                      "pure_frame_helper_sha256": frames.sha(frames.__file__), "gif_sha256": frames.sha(gif),
                      "png_sha256": frames.sha(png), "media_steps": list(frames.STEPS),
                      "fixed_initial_only_color_limit": vmax, "native_updates": 0, "host_forwards": 0,
                      "scientific_status_unchanged": report["scientific_status"],
                      "seconds": time.monotonic()-STARTED, "limit_seconds": LIMIT}; code = 0
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        torch.set_num_threads(threads)
        if not torch.equal(before, torch.get_rng_state()): error = {"type": "AssertionError", "message": "CPU renderer changed caller RNG"}
        if time.monotonic()-STARTED > LIMIT: error = {"type": "TimeoutError", "message": "CPU cleanup60s exceeded"}
        if error is not None:
            completion = {"task": TASK, "complete": False, "error": error, "seconds": time.monotonic()-STARTED, "limit_seconds": LIMIT}; code = 2
        if writable:
            path = run / "media-completion.json"
            try:
                with path.open("x") as handle: handle.write(json.dumps(completion, indent=2, allow_nan=False)+"\n")
            except FileExistsError: return 2
            if time.monotonic()-STARTED > LIMIT:
                completion.update(complete=False, error={"type": "TimeoutError", "message": "final media receipt60s exceeded"}, seconds=time.monotonic()-STARTED)
                path.write_text(json.dumps(completion, indent=2, allow_nan=False)+"\n"); code = 2
        print(json.dumps(completion, allow_nan=False), flush=True)
    return code


if __name__ == "__main__": raise SystemExit(main())

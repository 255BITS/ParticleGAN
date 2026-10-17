"""Render saved native-observed goal states on CPU; no model or GPU forwards.

One command: python -m examples.render_e22_routed_caption_accuracy --run-directory RUN
The separate60s rendering budget changes no training or scientific verdict.
Optional media dependencies: NumPy and Pillow>=10.1.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ARMS = ("ordinary_BF16", "particle_BF16")
STEPS = (0, 64, 128, 256, 384, 512)
LIMIT = 60


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget():
    if time.monotonic() - STARTED > LIMIT: raise TimeoutError("CPU rendering60s budget exceeded")


def token_maps(residual):
    """Each pixel summarizes all physical coordinates of one native patch."""
    if residual.ndim != 3 or len(residual) < 6 or not torch.isfinite(residual).all():
        raise ValueError("six finite observed residual maps are required")
    side = math.isqrt(residual.shape[1])
    if side * side != residual.shape[1]: raise ValueError("patch grid must be square")
    return residual[:6].double().square().mean(-1).sqrt().reshape(6, side, side).numpy()


def goal_frame(target, ordinary, particle, *, step, vmax, terminal):
    """Fixed axes/colors; no interpolation or fabricated training trajectory."""
    maps = [token_maps(value) for value in (target, ordinary, particle)]
    if not math.isfinite(vmax) or vmax <= 0: raise ValueError("fixed color limit must be positive")
    width, tile, height = 1180, 155, 680
    image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=17); small = ImageFont.load_default(size=14)
    draw.text((20, 15), f"Caption editing | native update {step}/512 per arm", fill="#111111", font=font)
    draw.text((20, 44), "Observed patch RMS velocity error; desired residual is zero", fill="#333333", font=small)
    for row, (name, values) in enumerate(zip(("Target = 0", "Ordinary LoRA", "Particles"), maps)):
        y = 94 + row * 174; draw.text((15, y + 65), name, fill="#111111", font=small)
        for source, values_at_source in enumerate(values):
            x = 165 + source * 165
            intensity = np.clip(values_at_source / vmax, 0, 1)
            rgb = np.stack((np.full_like(intensity, 255), 255 * (1-intensity), 255 * (1-intensity)), -1).astype(np.uint8)
            heat = Image.fromarray(rgb).resize((tile, tile), Image.Resampling.NEAREST)
            image.paste(heat, (x, y)); draw.rectangle((x, y, x+tile, y+tile), outline="#dddddd")
            if row == 0: draw.text((x + 44, 72), f"Source {source}", fill="#333333", font=small)
    draw.text((20, 627), f"All frames: white=0, red={vmax:.6f} physical error. Six fixed TEST queries shown; formal score uses48.", fill="#333333", font=small)
    if step == 512:
        draw.text((20, 651), f"Terminal {terminal['scientific_status']}: RMSE ordinary={terminal['ordinary']:.8f}, particles={terminal['particle']:.8f}", fill="#111111", font=small)
    else:
        draw.text((20, 651), "Visual observations only; numerical PASS/FAIL uses the fixed512 endpoint.", fill="#555555", font=small)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args(); run = args.run_directory; source = sha(__file__)
    before = torch.get_rng_state().clone(); old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    completion = None; error = None; result = 2; writable = False
    try:
        if any((run/name).exists() for name in ("goal.gif","goal-final.png","media-completion.json")):
            raise ValueError("preserve existing media outputs; choose an unrendered run")
        if not run.is_dir(): raise ValueError("an existing completed run directory is required")
        writable = True
        report_path, receipt_path, media_path = run/"report.json", run/"completion.json", run/"observed-media.pt"
        report = json.loads(report_path.read_text()); receipt = json.loads(receipt_path.read_text())
        bound = {"report_sha256": sha(report_path), "completion_sha256": sha(receipt_path), "observed_media_sha256": sha(media_path)}
        if (not receipt["complete"] or receipt["report_sha256"] != bound["report_sha256"]
            or report["task"] != "routed_caption_accuracy_v1" or not report["complete"]
            or report["observed_media_sha256"] != bound["observed_media_sha256"]
            or report["scientific_status"] not in ("PASS", "FAIL")):
            raise ValueError("complete bound actual-training artifacts are required")
        data = torch.load(media_path, map_location="cpu", weights_only=True)
        if (tuple(data["steps"]) != STEPS or list(data["source_ids"][:6]) != list(range(6))
            or not data["capture_native_state_rng_diagnostics_unchanged"] or data["target_residual"].count_nonzero()
            or set(data["actual_residuals"]) != set(ARMS)):
            raise ValueError("fixed target/source/media law differs")
        for arm in ARMS:
            if set(data["actual_residuals"][arm]) != {str(s) for s in STEPS}: raise ValueError("every observed media step is mandatory")
            for value in data["actual_residuals"][arm].values():
                if tuple(value.shape) != (8, 256, 16): raise ValueError("actual full-geometry media shape differs")
        vmax = max(float(token_maps(data["actual_residuals"][arm]["0"]).max()) for arm in ARMS)
        if not math.isfinite(vmax) or vmax <= 0: raise ValueError("initial observed error scale is degenerate")
        terminal = {"scientific_status": report["scientific_status"], "ordinary": report["accuracy"][ARMS[0]]["rmse"], "particle": report["accuracy"][ARMS[1]]["rmse"]}
        frames = []
        for step in STEPS:
            frames.append(goal_frame(data["target_residual"], *(data["actual_residuals"][arm][str(step)] for arm in ARMS), step=step, vmax=vmax, terminal=terminal)); budget()
        gif, png = run/"goal.gif", run/"goal-final.png"
        with gif.open("xb") as handle:
            frames[0].save(handle, format="GIF", save_all=True, append_images=frames[1:], duration=900, loop=0, disposal=2)
        with png.open("xb") as handle: frames[-1].save(handle, format="PNG")
        budget()
        if sha(__file__) != source or any(sha(run/name) != expected for name,expected in (("report.json",bound["report_sha256"]),("completion.json",bound["completion_sha256"]),("observed-media.pt",bound["observed_media_sha256"]))):
            raise ValueError("render source or actual observations changed")
        if torch.cuda.is_initialized(): raise AssertionError("CPU renderer initialized CUDA")
        completion = {"complete": True, **bound, "renderer_source_sha256": source, "gif_sha256": sha(gif), "png_sha256": sha(png),
            "media_steps": list(STEPS), "fixed_initial_only_color_limit": vmax, "native_updates": 0, "host_forwards": 0,
            "scientific_status_unchanged": report["scientific_status"], "seconds": time.monotonic()-STARTED, "limit_seconds": LIMIT}
        result = 0
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        torch.set_num_threads(old_threads)
        if not torch.equal(before,torch.get_rng_state()): error={"type":"AssertionError","message":"CPU renderer changed caller RNG"}
        if time.monotonic()-STARTED > LIMIT: error={"type":"TimeoutError","message":"CPU rendering cleanup exceeded60s"}
        if error is not None:
            completion={"complete":False,"error":error,"seconds":time.monotonic()-STARTED,"limit_seconds":LIMIT};result=2
        path=run/"media-completion.json"
        if writable:
            try:
                with path.open("x") as handle: handle.write(json.dumps(completion,indent=2,allow_nan=False)+"\n")
            except FileExistsError:
                print("preserved concurrently created media receipt",flush=True)
                return 2
            if time.monotonic()-STARTED>LIMIT:
                completion.update(complete=False,error={"type":"TimeoutError","message":"final media receipt write exceeded60s"},seconds=time.monotonic()-STARTED)
                path.write_text(json.dumps(completion,indent=2,allow_nan=False)+"\n");result=2
        print(json.dumps(completion,allow_nan=False),flush=True)
    return result


if __name__ == "__main__": raise SystemExit(main())

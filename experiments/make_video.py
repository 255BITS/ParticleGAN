#!/usr/bin/env python
"""
make_video.py

Render a training-evolution video for one arm config: the EMA read-out's fake
cloud, frame by frame, against a fixed real background, in the repo's scatter
style (`examples/100gaussians.py::save_fake_scatter` -- reals faint, fakes
opaque, axes fixed at +/-6, equal aspect).

The point of the video is qualitative: how a run *gets to* its final numbers,
not what those numbers are. So the trajectory has to be the real one. This
script does not reimplement the loop -- it calls `experiments.train_arm.train`
with its `frame_callback` observer hook, which hands over `(step, ema_G,
ema_prior)` at each frame boundary with the global RNG state snapshotted around
the call. Rendering (and `mode_coverage`, which draws on the default RNG) can
therefore consume as much randomness as it likes without shifting a single
training step. The run's `metrics.jsonl` comes out byte-identical to the
official run's, which is the check `--verify_against` performs.

Usage:

    python experiments/make_video.py --config configs/b_cap_c1p0_lr2p0_s1.yaml
    python experiments/make_video.py --config cfg.yaml --frame_interval 50 --fps 30
    python experiments/make_video.py --config cfg.yaml --gif --size 750 --fps 10
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.train_arm import load_config, train
from lib.particle_prior import ParticlePrior
from lib.toy_models import sample_100gaussians, mode_coverage

# Frame geometry / style, held to the repo's scatter conventions.
N_FAKE = 4096
N_REAL_BG = 8192
XLIM = (-6.0, 6.0)
YLIM = (-6.0, 6.0)
FIGSIZE = (6.0, 6.0)
DPI = 100

SCRATCH = Path(
    "/tmp/claude-1000/-home-martyn-dev-ParticleGAN/"
    "fb26c8a8-dc4d-4296-8279-4f671a9a83d0/scratchpad"
)
FRAME_ROOT = SCRATCH / "video_frames"


def arm_label(cfg: dict) -> str:
    """Human-readable 'which experiment is this' string for the frame title."""
    bits = [str(cfg["arm"]), str(cfg["loss_type"]), f"coeff={cfg['coeff']:g}"]
    if float(cfg["lr_mult"]) != 1.0:
        bits.append(f"lr x{float(cfg['lr_mult']):g}")
    # Only the non-default prior is called out, so the titles of videos rendered
    # before this existed stay exactly as they were.
    if str(cfg.get("prior", "particles")).lower() != "particles":
        bits.append("FIXED GAUSSIAN PRIOR (no particles)")
    return "  |  ".join(bits)


def render_frame(
    path: Path,
    ema_G: nn.Module,
    ema_prior: ParticlePrior,
    device: torch.device,
    real_bg: torch.Tensor,
    step: int,
    total_steps: int,
    label: str,
    dpi: int = DPI,
) -> Tuple[int, float]:
    """
    Draw one frame and return the (modes, hq) it reports.

    Matches `save_fake_scatter`: the fakes come from a *fixed* block of
    particles (`fixed_first_n=True`) so the cloud's motion across frames is the
    model moving, not the sample changing.
    """
    was_training = ema_G.training
    ema_G.eval()
    ema_prior.eval()
    with torch.no_grad():
        z_fake, _ = ema_prior.sample(N_FAKE, fixed_first_n=True)
        fake = ema_G(z_fake.to(device)).cpu()
    modes, hq = mode_coverage(ema_G, ema_prior, device)
    ema_G.eval()  # mode_coverage leaves the module in train mode
    real = real_bg.cpu()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(real[:, 0], real[:, 1], s=4, alpha=0.2, label="real")
    ax.scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.8, label="fake")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"{label}\nstep {step}/{total_steps}   modes {modes}/100   hq {hq:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

    if was_training:
        ema_G.train()
        ema_prior.train()
    return int(modes), float(hq)


def encode_gif(frame_dir: Path, out_path: Path, fps: int) -> Tuple[Path, str]:
    """
    Encode the PNG sequence as a GIF. Preferred path is ffmpeg's two-pass
    palettegen/paletteuse (one optimal 256-colour palette for the whole clip,
    Bayer-dithered so flat matplotlib backgrounds stay flat and the file stays
    small); pillow's adaptive per-frame palette is the fallback when ffmpeg is
    absent or the filter graph fails. Returns (written_path, encoder_used).
    """
    gif_path = out_path.with_suffix(".gif")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        vf = (
            "[0:v]split[a][b];"
            "[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        )
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%05d.png"),
            "-filter_complex", vf,
            "-loop", "0",
            str(gif_path),
        ]
        if subprocess.run(cmd).returncode == 0:
            return gif_path, "ffmpeg/palettegen+paletteuse"
        print("[warn] ffmpeg gif encode failed; falling back to pillow", flush=True)

    from PIL import Image

    frames = sorted(frame_dir.glob("frame_*.png"))
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frames]
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
        optimize=True,
    )
    return gif_path, "pillow/gif (adaptive palette)"


def encode(frame_dir: Path, out_path: Path, fps: int, n_frames: int) -> Tuple[Path, str]:
    """
    Encode the PNG sequence. H.264 mp4 via ffmpeg when it is on PATH, otherwise
    a pillow GIF next to it. Returns (written_path, encoder_used).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        # Both are H.264 in an mp4 with yuv420p at quality ~23; they differ only
        # in which one this ffmpeg build happens to ship. Fedora's default build
        # has no libx264, so libopenh264 is the one that actually runs here.
        candidates = [
            ("libx264", ["-crf", "23"]),
            ("libopenh264", ["-rc_mode", "quality", "-q", "23"]),
        ]
        available = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        ).stdout
        for enc, quality in candidates:
            if enc not in available:
                continue
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(frame_dir / "frame_%05d.png"),
                "-c:v", enc,
                "-pix_fmt", "yuv420p",
                *quality,
                "-r", str(fps),
                str(out_path),
            ]
            if subprocess.run(cmd).returncode == 0:
                return out_path, f"ffmpeg/{enc}"
        print("[warn] ffmpeg found but no usable H.264 encoder; falling back to GIF",
              flush=True)

    # Fallback: no ffmpeg (or no H.264 in it). Pillow can stitch a GIF.
    from PIL import Image

    gif_path = out_path.with_suffix(".gif")
    frames = sorted(frame_dir.glob("frame_*.png"))
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frames]
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
    )
    return gif_path, "pillow/gif (ffmpeg not found on PATH)"


def probe(path: Path) -> dict:
    """ffprobe summary of the encoded file, or {} when ffprobe is unavailable."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or path.suffix not in (".mp4", ".gif"):
        return {}
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    stream = (info.get("streams") or [{}])[0]
    return {
        "frames": int(stream.get("nb_read_frames", 0)),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": stream.get("avg_frame_rate"),
        "duration_sec": float(info.get("format", {}).get("duration", "nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a training-evolution video for one arm config.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output video path (default results/videos/{run_name}.mp4).")
    parser.add_argument("--frame_interval", type=int, default=25,
                        help="Render a frame every N steps (plus step 0 and the last step).")
    parser.add_argument("--fps", type=int, default=20, help="Playback frame rate.")
    parser.add_argument("--gif", action="store_true",
                        help="Write an animated GIF (ffmpeg palettegen/paletteuse, "
                             "pillow adaptive palette as fallback) instead of an mp4.")
    parser.add_argument("--size", type=int, default=int(FIGSIZE[0] * DPI),
                        help="Frame size in pixels (square). Default 600.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap on rendered frames: frame boundaries are strided so "
                             "at most this many are drawn (step 0 and the last step "
                             "are always kept).")
    parser.add_argument("--device", type=str, default=None, help="e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Where the run's own artifacts go. Defaults to a scratch "
                             "dir so an official results/runs/... is never clobbered.")
    parser.add_argument("--verify_against", type=str, default=None,
                        help="An existing metrics.jsonl to diff this run's against, "
                             "as a trajectory-parity check.")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Keep the PNG frames instead of reusing/overwriting them.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = Path(args.config).stem
    total_steps = int(cfg["total_steps"])

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Keep the official run directory untouched: the video run writes its own
    # metrics.jsonl / ckpt.pt / summary.json somewhere disposable.
    cfg["out_dir"] = args.run_dir or str(SCRATCH / "video_runs" / run_name)

    frame_dir = FRAME_ROOT / run_name
    if frame_dir.exists() and not args.keep_frames:
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    # The fixed real background: drawn once, before training, from its own
    # generator (seed + 1, the same stream train_arm uses for its viz reals) so
    # it never touches the default RNG the training loop draws from.
    bg_gen = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    bg_gen.manual_seed(int(cfg["seed"]) + 1)
    real_bg = sample_100gaussians(batch_size=N_REAL_BG, device=device, generator=bg_gen)

    label = arm_label(cfg)
    dpi = max(1, round(args.size / FIGSIZE[0]))
    state = {"n": 0, "seen": 0, "last": (0, 0.0)}

    # `--max_frames` strides the frame boundaries rather than the training loop:
    # every `stride`-th boundary is drawn, so the trajectory is unchanged and
    # only the rendering (the expensive part) is skipped.
    stride = 1
    if args.max_frames is not None and args.max_frames > 0:
        boundaries = total_steps // max(1, args.frame_interval) + 1
        stride = max(1, -(-boundaries // args.max_frames))

    def frame_callback(step: int, ema_G: nn.Module, ema_prior: ParticlePrior) -> None:
        seen = state["seen"]
        state["seen"] = seen + 1
        if stride > 1 and seen % stride != 0 and step != total_steps:
            return
        path = frame_dir / f"frame_{state['n']:05d}.png"
        modes, hq = render_frame(
            path, ema_G, ema_prior, device, real_bg, step, total_steps, label, dpi=dpi
        )
        state["n"] += 1
        state["last"] = (modes, hq)
        if state["n"] % 25 == 0 or step == total_steps:
            print(f"[frame {state['n']:04d}] step {step} modes {modes}/100 hq {hq:.3f}",
                  flush=True)

    print(f"[video] {run_name}: {total_steps} steps, frame every "
          f"{args.frame_interval} -> frames in {frame_dir}", flush=True)
    train(cfg, device, frame_callback=frame_callback, frame_interval=args.frame_interval)

    suffix = ".gif" if args.gif else ".mp4"
    out_path = (
        Path(args.out) if args.out
        else _REPO_ROOT / "results" / "videos" / f"{run_name}{suffix}"
    )
    if args.gif:
        written, encoder = encode_gif(frame_dir, out_path, args.fps)
    else:
        written, encoder = encode(frame_dir, out_path, args.fps, state["n"])
    info = probe(written)

    parity: Optional[str] = None
    if args.verify_against:
        ref = Path(args.verify_against).read_text()
        got = (Path(cfg["out_dir"]) / "metrics.jsonl").read_text()
        parity = "IDENTICAL" if ref == got else "DIFFERS"

    modes, hq = state["last"]
    print(json.dumps({
        "run": run_name,
        "video": str(written),
        "encoder": encoder,
        "bytes": written.stat().st_size,
        "frames_rendered": state["n"],
        "probe": info,
        "final_modes": modes,
        "final_hq": hq,
        "metrics_parity": parity,
        "frame_dir": str(frame_dir),
        "run_dir": cfg["out_dir"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
